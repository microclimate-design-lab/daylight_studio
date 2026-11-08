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
MODEL_NAMES = ("XGBoost", "MLP")


# ------------------------------------------------------------------------------------------- models
class Models:
    def __init__(self):
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.torch = torch
        self.lock = threading.Lock()
        self.win = {}
        for side in "NS":
            c = xgb.XGBClassifier()
            c.load_model(MODELS / f"window_{side}.json")
            self.win[side] = c
        self.xgb = {}
        for name in ["room_xgb_sda", "room_xgb_mean_illum", "unit_xgb_sda", "unit_xgb_mean_illum", "map_xgb"]:
            m = xgb.XGBRegressor()
            m.load_model(MODELS / f"{name}.json")
            self.xgb[name] = m
        self.mlp, self.scaler = {}, {}
        for lvl in ["room", "unit", "map"]:
            self.mlp[lvl] = torch.load(MODELS / f"{lvl}_mlp.pt", weights_only=False).eval()
            self.scaler[lvl] = pickle.load(open(MODELS / f"{lvl}_scaler.pkl", "rb"))

    def run_mlp(self, lvl, X):
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
    model = payload.get("model", "XGBoost")
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown model {model}")
    ori = int(payload.get("ori", 0))
    wwr_s = float(np.clip(payload.get("wwr_s", 0.35), 0.2, 0.5))
    wwr_n = float(np.clip(payload.get("wwr_n", 0.275), 0.2, 0.35))
    g, mask, bbox, order, bbox_m, warnings = build_graph(
        payload["outline"], payload["rooms"], payload.get("links", []), ori, wwr_s, wwr_n)
    n = g["num_nodes"]
    win_override = payload.get("windows", {}) or {}

    with M.lock:
        # ---- windows (not drawn in a bubble diagram -> predicted, unless fixed by the user)
        Xw = F.window_features(g)
        pwin = np.column_stack([M.win[s].predict_proba(Xw)[:, 1] for s in "NS"])
        win_src = ["model"] * n
        for k, cid in enumerate(order):
            o = win_override.get(str(cid))
            if o and o != "auto":
                pwin[k] = [float("N" in o), float("S" in o)]
                win_src[k] = "user"
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
    return {
        "model": model,
        "inputs": {"ori": ori, "wwr_s": wwr_s, "wwr_n": wwr_n},
        "unit": {"sda": unit[0], "mean_da": unit[1],
                 "sda_map": float((mp[inside] > SDA_CUT).mean()), "mean_da_map": float(mp[inside].mean())},
        "rooms": rooms_out,
        "da_map": {"rows": inside.shape[0], "cols": inside.shape[1], "cell_m": F.CELL_M,
                   "bbox_m": bbox_m, "values": vals, "room": cell_room.ravel().tolist()},
        "graph": _graph_json(g, order),
        "warnings": warnings,
    }


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

    def __init__(self, data_dir=DATA):
        self.index = json.load(open(data_dir / "library.json"))
        self.samples = json.load(open(data_dir / "samples.json"))
        self.grids = np.load(data_dir / "da_grids.npz")

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
