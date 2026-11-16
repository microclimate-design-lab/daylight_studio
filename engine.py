"""
Vector bubble diagram -> graph (dataset format) -> daylight prediction.

Same pipeline as daylight_gnn/bubble_pipeline/predict.py, but the input is already vector data
(drawn in the browser), so no image parsing is needed:

    outline polygon (m) + circles (type, centre, radius) + links
        -> graph_features / node_features / edge_index / edge_attr   (bubble_to_graph.py conventions)
        -> window model p(win_N), p(win_S) per room
        -> room-level SDA / mean DA, unit-level SDA / mean DA, DA map on the 0.1 m grid

Coordinates sent by the browser: metres, x = west -> east, y = north -> south (screen convention).
"""
import json
import math
import os
import pickle
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("DAYLIGHT_MODELS", ROOT / "models"))
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "daylight"))  # inference code from daylight_gnn/bubble_pipeline
import features as F  # noqa: E402
from bubble_to_graph import (ROOM_TYPES, GRAPH_FEATURE_NAMES, NODE_FEATURE_NAMES,  # noqa: E402
                             AREA_SUM_RATIO, _ray_lengths)

PX = F.PX_PER_M
SDA_CUT = 127 / 255
TARGETS = ["sda", "mean_illum"]
MARGIN = 40  # px around the outline on the raster canvas
GEO = ["area", "area_ratio", "x", "y", "d_N", "d_S", "d_E", "d_W"]
# Singapore layouts have a fourth bedroom (code 13); the Shanghai data has none.
ALL_TYPES = list(ROOM_TYPES) + [(13, "Master room 3", (190, 126, 204))]
TYPE_NAME = {i: n for i, n, _ in ALL_TYPES}
REGIONS = {
    "shanghai": {"name": "Shanghai", "models": ["XGBoost", "MLP", "GNN"], "sides": "NS",
                 "wwr": {"S": [0.2, 0.5, 0.35], "N": [0.2, 0.35, 0.275]}, "types": list(range(9)),
                 "remap": {13: 3}, "library": ""},
    "singapore": {"name": "Singapore", "models": ["GNN"], "sides": "NSEW",
                  "wwr": {s: [0.2, 0.45, 0.33] for s in "NSEW"}, "types": [0, 1, 2, 3, 13, 5, 6, 8],
                  "remap": {4: 0, 7: 3}, "library": "sg_"},
}
MODEL_NAMES = tuple(dict.fromkeys(m for r in REGIONS.values() for m in r["models"]))


# ------------------------------------------------------------------------------------------- models
class XGB:
    """XGBoost model through the plain Booster (no scikit-learn import, less memory). Same output as the
    XGBRegressor / XGBClassifier wrappers: uses best_iteration when the model was trained with early stopping."""

    def __init__(self, path):
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        best = self.booster.attributes().get("best_iteration")
        self.iters = (0, int(best) + 1) if best is not None else (0, 0)

    def predict(self, X):
        return self.booster.inplace_predict(np.asarray(X, np.float32), iteration_range=self.iters)

    def predict_proba(self, X):
        p = self.predict(X)
        return np.column_stack([1 - p, p])


class Models:
    """backend "numpy" (default, no torch needed; weights from tools/export_numpy.py) or "torch" (original checkpoints)."""

    def __init__(self, backend=None):
        self.backend = backend or os.environ.get("DAYLIGHT_BACKEND", "numpy")
        if self.backend == "torch":
            import torch
            import gnn
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
            self.torch = torch
        else:
            import nn_numpy as gnn
        self.lock = threading.Lock()
        self.win = {side: XGB(MODELS / f"window_{side}.json") for side in "NS"}
        self.gnn = {region: gnn.Ensemble(MODELS / "gnn" / region) for region in REGIONS}
        self.gnn_map = {region: gnn.MapEnsemble(MODELS / "gnn" / region) for region in REGIONS
                        if (MODELS / "gnn" / region / "map_spec.json").exists()}
        self.sg_win = {side: XGB(MODELS / "gnn" / "singapore" / f"window_{side}.json") for side in "NSEW"}
        self.xgb = {name: XGB(MODELS / f"{name}.json")
                    for name in ["room_xgb_sda", "room_xgb_mean_illum", "unit_xgb_sda", "unit_xgb_mean_illum", "map_xgb"]}
        self.mlp, self.scaler = {}, {}
        for lvl in ["room", "unit", "map"]:
            if self.backend == "torch":
                self.mlp[lvl] = self.torch.load(MODELS / f"{lvl}_mlp.pt", weights_only=False, map_location="cpu").eval()
                self.scaler[lvl] = pickle.load(open(MODELS / f"{lvl}_scaler.pkl", "rb"))
            else:
                self.mlp[lvl] = gnn.MLP(MODELS / "np" / f"{lvl}_mlp.npz")

    def run_mlp(self, lvl, X):
        if self.backend != "torch":
            return self.mlp[lvl](X)
        with self.torch.no_grad():
            Xs = self.torch.tensor(self.scaler[lvl].transform(X), dtype=self.torch.float32)
            return self.mlp[lvl](Xs).numpy()

    def run_xgb(self, name, X):
        return np.clip(self.xgb[name].predict(X), 0, 1)


# ------------------------------------------------------------------------------------------- graph
def rasterise(outline_m):
    """Outline polygon (m) -> filled mask on a px canvas (1 m = 199.735 px), contours, offset."""
    pts = np.asarray(outline_m, float) * PX
    off = pts.min(0) - MARGIN
    pts = pts - off
    w, h = (pts.max(0) + MARGIN).astype(int) + 1
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [max(cs, key=cv2.contourArea)]
    return mask, cs, off


def build_graph(outline_m, rooms, links, ori, wwr_s, wwr_n):
    """rooms: [{id, type, x, y, r}] (m), links: [[id_a, id_b], ...].
    Returns graph dict (dataset format), mask, bbox_px, order (graph index -> client id), warnings."""
    if len(outline_m) < 3:
        raise ValueError("boundary needs at least 3 points")
    if not rooms:
        raise ValueError("place at least one room bubble")
    mask, contours, off = rasterise(outline_m)
    bx, by, W, H = cv2.boundingRect(mask)
    A_px = float(mask.sum())
    P_px = float(sum(cv2.arcLength(c, True) for c in contours))
    A_m2 = A_px / PX ** 2

    warnings = []
    items = []
    for r in rooms:
        cx, cy = r["x"] * PX - off[0], r["y"] * PX - off[1]
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]):
            warnings.append(f"room {r['id']} centre is outside the boundary – ignored")
            continue
        items.append(dict(id=r["id"], type=int(r["type"]), c=(cx, cy), r=max(float(r["r"]), 0.05)))
    if not items:
        raise ValueError("no room centre lies inside the boundary")

    # room areas from bubble-area proportions (as bubble_to_graph.py)
    r2 = sum(it["r"] ** 2 for it in items)
    for it in items:
        it["area"] = it["r"] ** 2 / r2 * AREA_SUM_RATIO * A_m2
    # dataset node order: room type, then north -> south, west -> east
    items.sort(key=lambda it: (it["type"], round(it["c"][1] / 50), it["c"][0]))
    A_tot = sum(it["area"] for it in items)
    nodes = []
    for it in items:
        cx, cy = it["c"]
        dN, dS, dE, dW = (d / PX for d in _ray_lengths(mask, cx, cy))
        nodes.append([it["type"], it["area"], it["area"] / A_tot, (cx - bx) / W, (by + H - cy) / H,
                      dN, dS, dE, dW, float("nan"), float("nan")])
    idx = {it["id"]: k for k, it in enumerate(items)}
    ei, ea, seen = [[], []], [], set()
    for a, b in links:
        if a not in idx or b not in idx or a == b:
            continue
        i, j = sorted((idx[a], idx[b]))
        if (i, j) in seen:
            continue
        seen.add((i, j))
        for p, q in ((i, j), (j, i)):
            ei[0].append(p); ei[1].append(q)
            ea.append([nodes[q][3] - nodes[p][3], nodes[q][4] - nodes[p][4]])
    g = {
        "num_nodes": len(nodes), "num_edges": len(ea),
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "graph_features": [W / PX, H / PX, W / H, A_m2, P_px / PX, 4 * math.pi * A_px / P_px ** 2,
                           int(ori), float(wwr_s), float(wwr_n), float("nan")],
        "node_feature_names": NODE_FEATURE_NAMES, "node_features": nodes,
        "edge_feature_names": ["dx", "dy"], "edge_index": ei, "edge_attr": ea,
    }
    bbox_m = [(bx + off[0]) / PX, (by + off[1]) / PX, W / PX, H / PX]
    return g, mask, (bx, by, W, H), [it["id"] for it in items], bbox_m, warnings


# ------------------------------------------------------------------------------------------- predict
def predict(M, payload):
    region = payload.get("region", "shanghai")
    if region not in REGIONS:
        raise ValueError(f"unknown region {region}")
    reg = REGIONS[region]
    model = payload.get("model", reg["models"][0])
    if model not in reg["models"]:
        raise ValueError(f"model {model} is not available for {reg['name']}")
    ori = int(payload.get("ori", 0))
    wwr = {s: float(np.clip(payload.get(f"wwr_{s.lower()}", d), lo, hi)) for s, (lo, hi, d) in reg["wwr"].items()}
    rooms, warnings = [], []
    for r in payload["rooms"]:
        t = int(r["type"])
        if t not in reg["types"]:
            t = reg["remap"].get(t, t)
            warnings.append(f"{TYPE_NAME.get(int(r['type']), r['type'])} #{r['id']} is not a {reg['name']} room type – "
                            f"treated as {TYPE_NAME[t]}")
        rooms.append({**r, "type": t})
    g, mask, bbox, order, bbox_m, w = build_graph(payload["outline"], rooms, payload.get("links", []), ori, wwr["S"], wwr["N"])
    warnings += w
    if model == "GNN":
        return _predict_gnn(M, region, g, order, bbox_m, wwr, payload.get("windows", {}) or {}, warnings, mask, bbox)
    wwr_s, wwr_n = wwr["S"], wwr["N"]
    n = g["num_nodes"]
    win_override = payload.get("windows", {}) or {}

    with M.lock:
        # ---- windows (not drawn in a bubble diagram -> predicted, unless fixed by the user)
        Xw = F.window_features(g)
        pwin = np.column_stack([M.win[s].predict_proba(Xw)[:, 1] for s in "NS"])
        win_src = _override(pwin, "NS", order, win_override)
        for k in range(n):
            g["node_features"][k][9], g["node_features"][k][10] = int(pwin[k, 0] > 0.5), int(pwin[k, 1] > 0.5)

        # ---- room level
        Xr = F.room_features(g, pwin)
        if model == "XGBoost":
            room = np.column_stack([M.run_xgb(f"room_xgb_{t}", Xr) for t in TARGETS])
        else:
            room = M.run_mlp("room", Xr)
        # ---- unit (layout) level
        Xu = F.unit_features(g, pwin)[None]
        if model == "XGBoost":
            unit = [float(M.run_xgb(f"unit_xgb_{t}", Xu)[0]) for t in TARGETS]
        else:
            unit = [float(v) for v in M.run_mlp("unit", Xu)[0]]
        # ---- DA map
        Xc, inside, cell_room = F.cell_features(g, list(bbox), mask, pwin)
        mp = M.run_xgb("map_xgb", Xc) if model == "XGBoost" else M.run_mlp("map", Xc)[:, 0]
    mp = np.clip(mp.reshape(inside.shape), 0, 1)

    rooms_out = []
    for k, cid in enumerate(order):
        nf = g["node_features"][k]
        sel = cell_room == k
        rooms_out.append({
            "id": cid, "index": k, "type": int(nf[0]), "name": ROOM_TYPES[int(nf[0])][1],
            "area": nf[1], "area_ratio": nf[2],
            "p_win_N": float(pwin[k, 0]), "p_win_S": float(pwin[k, 1]), "win_source": win_src[k],
            "sda": float(np.clip(room[k, 0], 0, 1)), "mean_da": float(np.clip(room[k, 1], 0, 1)),
            "sda_map": float((mp[sel] > SDA_CUT).mean()) if sel.any() else None,
            "mean_da_map": float(mp[sel].mean()) if sel.any() else None,
            "cells": int(sel.sum()),
        })
    vals = np.where(inside, np.round(mp, 3), -1).ravel().tolist()
    for r, k in zip(rooms_out, range(n)):
        r["p_win"] = {"N": float(pwin[k, 0]), "S": float(pwin[k, 1])}
    return {
        "model": model, "region": region,
        "inputs": {"ori": ori, "wwr_s": wwr_s, "wwr_n": wwr_n},
        "unit": {"sda": unit[0], "mean_da": unit[1],
                 "sda_map": float((mp[inside] > SDA_CUT).mean()), "mean_da_map": float(mp[inside].mean())},
        "rooms": rooms_out,
        "da_map": {"rows": inside.shape[0], "cols": inside.shape[1], "cell_m": F.CELL_M,
                   "bbox_m": bbox_m, "values": vals, "room": cell_room.ravel().tolist()},
        "graph": _graph_json(g, order),
        "warnings": warnings,
    }


def _override(pwin, sides, order, win_override):
    """User-fixed windows per room ("N", "SE", "-" for none) replace the window-model probabilities."""
    src = ["model"] * len(order)
    for k, cid in enumerate(order):
        o = win_override.get(str(cid))
        if o and o != "auto":
            pwin[k] = [float(s in o) for s in sides]
            src[k] = "user"
    return src


def gnn_graph(region, g, win, wwr, vlt=None):
    """Studio graph (build_graph) + windows (n x sides, 0/1) -> GNN graph schema (gnn.py)."""
    n = g["num_nodes"]
    nf = np.array(g["node_features"], float)
    node = {name: nf[:, j] for j, name in enumerate(NODE_FEATURE_NAMES[:9])}
    for k, s in enumerate(REGIONS[region]["sides"]):
        node[f"win_{s}"] = np.asarray(win, float)[:, k]
    glob = dict(zip(GRAPH_FEATURE_NAMES, g["graph_features"]), n_rooms=n)
    if vlt is not None:
        glob["VLT"] = vlt
    if region == "singapore":
        glob.update({f"WWR_{s}": wwr[s] for s in "NSEW"})
    return {"node": node, "glob": glob, "edge_index": g["edge_index"], "edge_attr": g["edge_attr"]}


def gnn_cell_features(region, g, bbox, mask, win, wwr, shape=None):
    """features.cell_features for the pixel-level GNN. Singapore adds the E / W windows of the cell's room and
    WWR E / W; Master room 3 is one-hot encoded as Master room 2 here (the room embedding keeps the difference)."""
    win = np.asarray(win, float)
    if region == "singapore":
        g = {**g, "node_features": [[3 if r[0] == 13 else r[0]] + list(r[1:]) for r in g["node_features"]]}
    X, inside, room = F.cell_features(g, list(bbox), mask, win[:, :2], shape=shape)
    if region == "singapore":
        k = np.clip(room.ravel(), 0, None)
        X = np.c_[X, win[k, 2:4], np.tile([wwr["E"], wwr["W"]], (len(X), 1))].astype(np.float32)
    return X, inside, room


def sg_window_probs(M, g):
    nf = np.array(g["node_features"], float)
    n = len(nf)
    levels = M.gnn["singapore"].spec["categorical"]["room_function"]
    gf = dict(zip(GRAPH_FEATURE_NAMES, g["graph_features"]))
    Xw = np.column_stack([(nf[:, 0][:, None] == np.array(levels)[None]).astype(float), nf[:, 1:9]]
                         + [np.full(n, v) for v in (gf["W_bbox"], gf["H_bbox"], gf["A_boundary"], n)])
    return np.column_stack([M.sg_win[s].predict_proba(Xw)[:, 1] for s in "NSEW"])


def _predict_gnn(M, region, g, order, bbox_m, wwr, win_override, warnings, mask=None, bbox=None):
    """GATv2 ensemble (gnn.py): room and layout sDA / mean DA from the graph, including the links;
    DA map from the pixel-level GNN (map_s*.pt) when it has been trained."""
    reg, ens = REGIONS[region], M.gnn[region]
    spec = ens.spec
    n = g["num_nodes"]
    nf = np.array(g["node_features"], float)
    sides = reg["sides"]
    with M.lock:
        if region == "shanghai":
            pwin = np.column_stack([M.win[s].predict_proba(F.window_features(g))[:, 1] for s in "NS"])
        else:
            pwin = sg_window_probs(M, g)
        win_src = _override(pwin, sides, order, win_override)
        win = (pwin > 0.5).astype(float)
        graph = gnn_graph(region, g, win, wwr, vlt=spec["train_mean"].get("VLT"))  # VLT: no effect in the data
        room, unit, room_std, unit_std, _ = ens.predict([graph])
        mp = map_std = None
        if region in M.gnn_map and mask is not None:
            X, inside, cell_room = gnn_cell_features(region, g, bbox, mask, win, wwr)
            cell_room = np.where(inside, cell_room, -1)
            flat = np.flatnonzero(inside.ravel())
            mp, map_std = np.full(inside.size, -1.0), np.zeros(inside.size)
            mean, std = M.gnn_map[region].predict_map(gnn_graph(region, g, win, wwr), X[flat], cell_room.ravel()[flat])
            mp[flat], map_std[flat] = mean, std
            mp, map_std = mp.reshape(inside.shape), map_std.reshape(inside.shape)
        for k in range(n):
            g["node_features"][k][9], g["node_features"][k][10] = int(win[k, 0]), int(win[k, 1])
    glob = graph["glob"]
    for name in ("A_boundary", "n_rooms"):
        lo, hi = spec["train_range"][name]
        if not lo <= glob[name] <= hi:
            label = {"A_boundary": "boundary area", "n_rooms": "room count"}[name]
            warnings.append(f"{label} {glob[name]:.0f} is outside the {reg['name']} training range {lo:.0f}–{hi:.0f}")
    if not g["edge_index"][0]:
        warnings.append("no links drawn – the GNN reads links as shared walls")
    if region == "singapore" and any(s == "model" for s in win_src):
        warnings.append("windows of some rooms come from the window model, which is weak for east / west windows – "
                        "fix them in the room editor for a more reliable sDA")
    rooms_out = [{
        "id": cid, "index": k, "type": int(nf[k, 0]), "name": TYPE_NAME[int(nf[k, 0])],
        "area": nf[k, 1], "area_ratio": nf[k, 2],
        "p_win": {s: float(pwin[k, j]) for j, s in enumerate(sides)},
        "p_win_N": float(pwin[k, 0]), "p_win_S": float(pwin[k, 1]), "win_source": win_src[k],
        "sda": float(room[k, 0]), "mean_da": float(room[k, 1]), "sda_std": float(room_std[k, 0]),
        **_map_room_stats(mp, cell_room if mp is not None else None, k),
    } for k, cid in enumerate(order)]
    out_g = _graph_json(g, order)
    out_g["node_feature_names"] = NODE_FEATURE_NAMES[:9] + [f"win_{s}" for s in sides]
    out_g["node_features"] = [row[:9] + [int(pwin[k, j] > 0.5) for j in range(len(sides))]
                              for k, row in enumerate(out_g["node_features"])]
    if region == "singapore":
        out_g["graph_feature_names"] = GRAPH_FEATURE_NAMES[:6] + [f"WWR_{s}" for s in sides]
        out_g["graph_features"] = out_g["graph_features"][:6] + [round(wwr[s], 2) for s in sides]
    return {
        "model": "GNN", "region": region,
        "inputs": {"ori": 0, **{f"wwr_{s.lower()}": v for s, v in wwr.items()}},
        "unit": {"sda": float(unit[0, 0]), "mean_da": float(unit[0, 1]), "sda_std": float(unit_std[0, 0]),
                 "sda_map": float((mp[inside] > SDA_CUT).mean()) if mp is not None else None,
                 "mean_da_map": float(mp[inside].mean()) if mp is not None else None},
        "rooms": rooms_out, "graph": out_g, "warnings": warnings,
        "da_map": None if mp is None else {
            "rows": mp.shape[0], "cols": mp.shape[1], "cell_m": F.CELL_M, "bbox_m": bbox_m,
            "values": np.where(inside, np.round(mp, 3), -1).ravel().tolist(),
            "std": np.where(inside, np.round(map_std, 3), -1).ravel().tolist(), "room": cell_room.ravel().tolist()},
    }


def _map_room_stats(mp, cell_room, k):
    if mp is None:
        return {"sda_map": None, "mean_da_map": None, "cells": 0}
    sel = cell_room == k
    return {"sda_map": float((mp[sel] > SDA_CUT).mean()) if sel.any() else None,
            "mean_da_map": float(mp[sel].mean()) if sel.any() else None, "cells": int(sel.sum())}


def _graph_json(g, order):
    def fix(v):
        if isinstance(v, (float, np.floating)):
            return None if math.isnan(v) else round(float(v), 4)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    return {
        "graph_feature_names": g["graph_feature_names"],
        "graph_features": [fix(v) for v in g["graph_features"]],
        "node_feature_names": g["node_feature_names"],
        "node_features": [[fix(v) for v in row] for row in g["node_features"]],
        "edge_index": g["edge_index"], "edge_attr": [[fix(v) for v in e] for e in g["edge_attr"]],
        "order": order,
    }


# ------------------------------------------------------------------------------------------- library
class Library:
    """Boundary / sample library packed by tools/build_data.py (data/library.json, samples.json, da_grids.npz)."""

    def __init__(self, data_dir=DATA, prefix=""):
        self.index = json.load(open(data_dir / f"{prefix}library.json"))
        self.samples = json.load(open(data_dir / f"{prefix}samples.json"))
        self.grids = np.load(data_dir / f"{prefix}da_grids.npz")

    def sample(self, i):
        s = self.samples.get(str(i))
        if s is None:
            raise FileNotFoundError(i)
        g = self.grids[f"g{i}"]
        vals = np.where(g == 255, -1, np.round(g / 254.0, 3))
        out = dict(s)
        out["truth"] = {**s["truth"], "rows": int(g.shape[0]), "cols": int(g.shape[1]),
                        "values": vals.ravel().tolist()}
        return out
