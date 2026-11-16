"""
Package the GATv2 checkpoints for the studio and re-run them on their test sets (run once, results are committed).

    python tools/build_gnn.py --gnn D:\\course\\bps5231\\daylight_gnn --sg D:\\course\\bps5231\\daylight\\hdb_sg\\sg_layout_dataset

Shanghai   graphs/json + sample_parameters (train split -> scaler), results_gat/model_gatv2_R6_environ_s*.pt
Singapore  sg_layout_dataset/meta (train split -> scaler), results_gat_sg/model_gatv2_R6_window_s*.pt
           plus window classifiers (N / S / E / W) trained on the SG train + val rooms

Every checkpoint must reproduce the predictions saved next to it (pred_graph_gatv2_*.csv) before it is copied.
Writes models/gnn/<region>/{spec.json, model_s*.pt, test_predictions.csv}, models/gnn/singapore/window_*.json,
data/sg_library.json, data/sg_samples.json, data/sg_da_grids.npz.
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import engine  # noqa: E402
import features as F  # noqa: E402
import gnn as G  # noqa: E402
import sg_graphs  # noqa: E402

NODE_SH = ["room_function", "area", "area_ratio", "x", "y", "d_N", "d_S", "d_E", "d_W", "win_N", "win_S"]
GLOB_SH = ["W_bbox", "H_bbox", "AR_bbox", "A_boundary", "P_boundary", "compactness", "ORI", "WWR_S", "WWR_N", "VLT"]
BOUNDARY = ["n_rooms", "W_bbox", "H_bbox", "AR_bbox", "A_boundary", "P_boundary", "compactness"]
GEO = ["area", "area_ratio", "x", "y", "d_N", "d_S", "d_E", "d_W"]
SG_LEVELS = [0, 1, 2, 3, 5, 6, 8, 13]
WIN_FEATURES = GEO + ["W_bbox", "H_bbox", "A_boundary", "n_rooms"]
SPECS = {
    "shanghai": dict(rung="R6_environ", node=NODE_SH, glob=BOUNDARY + ["ORI", "WWR_S", "WWR_N", "VLT"], edge=["dx", "dy"],
                     categorical={"room_function": list(range(9)), "ORI": [0, 1, 2]}),
    "singapore": dict(rung="R6_window", node=["room_function"] + GEO + ["win_N", "win_S", "win_E", "win_W"],
                      glob=BOUNDARY + ["WWR_N", "WWR_S", "WWR_E", "WWR_W"], edge=["dx", "dy"],
                      categorical={"room_function": SG_LEVELS}),
}


def sh_graph(path):
    g = json.load(open(path, encoding="utf-8"))
    nf = np.array(g["node_features"], np.float32)
    glob = dict(zip(GLOB_SH, g["graph_features"]), n_rooms=nf.shape[0])
    return {"id": int(g["id"]), "node": {k: nf[:, j] for j, k in enumerate(NODE_SH)}, "glob": glob,
            "edge_index": np.array(g["edge_index"], np.int64).reshape(2, -1),
            "edge_attr": np.array(g["edge_attr"], np.float32).reshape(-1, 2),
            "y_graph": g["graph_labels"], "y_node": g["node_labels"]}


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    return {"mae": float(np.abs(e).mean()), "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), "n": len(y)}


def package(region, src, graphs, train_ids, test_ids):
    spec = dict(SPECS[region])
    train = [graphs[i] for i in train_ids]
    spec["scaler"] = G.fit_scaler(train, spec)
    spec["train_mean"] = {k: float(np.mean([g["glob"][k] for g in train])) for k in spec["glob"]}
    spec["train_range"] = {k: [float(min(g["glob"][k] for g in train)), float(max(g["glob"][k] for g in train))]
                           for k in spec["glob"] if k not in spec["categorical"]}
    out = ROOT / "models" / "gnn" / region
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("model_*.pt"):
        old.unlink()
    test = [graphs[i] for i in test_ids]
    for seed in range(5):
        ckpt = src / f"model_gatv2_{spec['rung']}_s{seed}.pt"
        shutil.copy(ckpt, out / f"model_s{seed}.pt")
    json.dump(spec, open(out / "spec.json", "w"), indent=1)
    ens = G.Ensemble(out)
    # guard: each seed must reproduce the predictions saved by the training run
    for seed, m in enumerate(ens.models):
        one = G.Ensemble.__new__(G.Ensemble)
        one.spec, one.models = spec, [m]
        _, yg, _, _, _ = one.predict(test)
        ref = pd.read_csv(src / f"pred_graph_gatv2_{spec['rung']}_s{seed}.csv").set_index("id").loc[test_ids]
        diff = float(np.abs(ref.pred_sda.values - yg[:, 0]).max())
        assert diff < 1e-4, f"{region} seed {seed}: checkpoint does not reproduce saved predictions ({diff:.2e})"
        print(f"  {region} seed {seed}: max |pred - saved| = {diff:.1e}")
    yn, yg, _, gstd, batch = ens.predict(test)
    saved = pd.read_csv(src / f"pred_node_gatv2_{spec['rung']}_s0.csv")
    true_node = pd.concat([saved[saved.id == g["id"]] for g in test])
    assert len(true_node) == len(yn), "room labels do not line up with the test graphs"
    rows =[{"id": g["id"], "true_sda": g["y_graph"][0], "pred_sda": float(yg[k, 0]), "seed_std_sda": float(gstd[k, 0]),
             "true_mean_illum": g["y_graph"][1], "pred_mean_illum": float(yg[k, 1])} for k, g in enumerate(test)]
    pd.DataFrame(rows).to_csv(out / "test_predictions.csv", index=False)
    lay = pd.DataFrame(rows)
    node_true = true_node.true_sda.values
    seeds = []
    for seed, m in enumerate(ens.models):
        one = G.Ensemble.__new__(G.Ensemble)
        one.spec, one.models = spec, [m]
        yn1, yg1, _, _, _ = one.predict(test)
        seeds.append({"layout_sda": metrics(lay.true_sda, yg1[:, 0]), "room_sda": metrics(node_true, yn1[:, 0])})
    spec["metrics"] = {
        "test_layouts": len(test), "test_rooms": int(len(node_true)),
        "ensemble": {"layout_sda": metrics(lay.true_sda, lay.pred_sda), "room_sda": metrics(node_true, yn[:, 0]),
                     "layout_mean_illum": metrics(lay.true_mean_illum, lay.pred_mean_illum)},
        "seed_mean": {"layout_sda_mae": float(np.mean([s["layout_sda"]["mae"] for s in seeds])),
                      "layout_sda_r2": float(np.mean([s["layout_sda"]["r2"] for s in seeds])),
                      "room_sda_mae": float(np.mean([s["room_sda"]["mae"] for s in seeds])),
                      "room_sda_r2": float(np.mean([s["room_sda"]["r2"] for s in seeds]))},
    }
    json.dump(spec, open(out / "spec.json", "w"), indent=1)
    print(region, json.dumps(spec["metrics"]["ensemble"]), "| seed mean", json.dumps(spec["metrics"]["seed_mean"]))
    return spec


def sg_windows(graphs, split):
    """XGBoost classifiers for win_N/S/E/W per room (windows are not drawn in a bubble diagram)."""
    def table(ids):
        X, Y = [], []
        for i in ids:
            g = graphs[i]
            n = len(g["node"]["area"])
            onehot = (g["node"]["room_function"][:, None] == np.array(SG_LEVELS)[None]).astype(np.float32)
            cols = [g["node"][k] for k in GEO] + [np.full(n, g["glob"][k]) for k in WIN_FEATURES[len(GEO):]]
            X.append(np.column_stack([onehot] + cols))
            Y.append(np.column_stack([g["node"][f"win_{s}"] for s in sg_graphs.SIDES]))
        return np.concatenate(X), np.concatenate(Y)
    Xtr, Ytr = table([i for i in graphs if split[i] in ("train", "val")])
    Xte, Yte = table([i for i in graphs if split[i] == "test"])
    out, auc = ROOT / "models" / "gnn" / "singapore", {}
    for k, s in enumerate(sg_graphs.SIDES):
        c = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=0)
        c.fit(Xtr, Ytr[:, k])
        c.save_model(out / f"window_{s}.json")
        p = c.predict_proba(Xte)[:, 1]
        auc[s] = {"auc": float(roc_auc_score(Yte[:, k], p)), "accuracy": float(((p > 0.5) == Yte[:, k]).mean())}
    print("SG window classifiers (test rooms):", {s: round(v["auc"], 3) for s, v in auc.items()})
    spec_path = out / "spec.json"
    spec = json.load(open(spec_path))
    spec["window_model"] = {"features": [f"type_{t}" for t in SG_LEVELS] + WIN_FEATURES, "test": auc}
    json.dump(spec, open(spec_path, "w"), indent=1)


def sg_library(sg_dir, graphs, split):
    """Sample library in the format of tools/build_data.py, outline in m with y pointing south (screen)."""
    lib, samples, grids = [], {}, {}
    for i, g in graphs.items():
        meta = json.load(open(sg_dir / "meta" / f"{i}.json", encoding="utf-8"))
        minx, miny, maxx, maxy = g["bounds"]
        to_screen = lambda x, y: [round(x - minx, 3), round(maxy - y, 3)]  # noqa: E731
        outline = [to_screen(x, y) for x, y in g["outline"]]
        n = len(meta["rooms"])
        lib.append({"id": i, "outline": outline, "n_rooms": n, "area": round(g["glob"]["A_boundary"], 1),
                    "W": round(g["glob"]["W_bbox"], 2), "H": round(g["glob"]["H_bbox"], 2), "split": split[i]})
        node_lab = room_labels(sg_dir, meta)
        rooms = []
        for k, r in enumerate(meta["rooms"]):
            cx, cy = g["centres"][k]
            win = "".join(s for s in sg_graphs.SIDES if g["node"][f"win_{s}"][k])
            rooms.append({"id": k + 1, "type": int(r["room_function"]), "x": to_screen(cx, cy)[0], "y": to_screen(cx, cy)[1],
                          "r": round(0.55 * math.sqrt(r["area_m2"] / math.pi), 3),
                          "true_sda": node_lab[k][0], "true_mean_da": node_lab[k][1], "true_win": win or "-"})
        links = sorted({(min(int(e["i"]), int(e["j"])) + 1, max(int(e["i"]), int(e["j"])) + 1) for e in meta["edges"]})
        w = meta["wwr"]
        samples[str(i)] = {"id": i, "outline": outline, "rooms": rooms, "links": [list(l) for l in links],
                           "params": {"ori": 0, "wwr_n": w["N"], "wwr_s": w["S"], "wwr_e": w["E"], "wwr_w": w["W"]},
                           "truth": {"sda": g["y_graph"][0], "mean_da": g["y_graph"][1]}}
        grids[f"g{i}"] = da_grid(sg_dir, meta, samples[str(i)], minx, maxy)
    data = ROOT / "data"
    json.dump(lib, open(data / "sg_library.json", "w"), separators=(",", ":"))
    json.dump(samples, open(data / "sg_samples.json", "w"), separators=(",", ":"))
    np.savez_compressed(data / "sg_da_grids.npz", **grids)
    print(f"SG library: {len(lib)} samples")


def room_labels(sg_dir, meta):
    """Per-room sDA / mean DA from the daylight image (meta only has layout labels), as in the SG_Test notebook."""
    img = cv2.imread(str(sg_dir / "images" / "daylight" / f"{meta['id']}.jpg"), cv2.IMREAD_UNCHANGED)
    gray, alpha = img[..., 2], img[..., 3]
    out = []
    for r in meta["rooms"]:
        m = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(r["polygon_px"])).astype(np.int32)], 1)
        vals = gray[(m > 0) & (alpha > 0)]
        out.append((float((vals > 127).mean()), float(vals.mean() / 255)) if vals.size else (None, None))
    return out


def da_grid(sg_dir, meta, sample, minx, maxy):
    """Daylight image -> DA grid on exactly the 0.1 m cells the studio uses for this outline
    (engine.build_graph + features.cell_features), uint8 value * 254, 255 = outside."""
    img = cv2.imread(str(sg_dir / "images" / "daylight" / f"{meta['id']}.jpg"), cv2.IMREAD_UNCHANGED)
    s = meta.get("px_per_m", 200.0)
    p0, m0 = meta["rooms"][0]["polygon_px"][0], meta["rooms"][0]["polygon_m"][0]
    ax, ay = p0[0] - s * m0[0], p0[1] + s * m0[1]                     # px = ax + s * x, py = ay - s * y
    g, _, _, _, bbox_m, _ = engine.build_graph(sample["outline"], sample["rooms"], [], 0, 0.3, 0.3)
    rows, cols = F.grid_shape(g)
    xs = minx + bbox_m[0] + (np.arange(cols) + 0.5) * F.CELL_M        # studio x (m) -> SG metres
    ys = maxy - (bbox_m[1] + (np.arange(rows) + 0.5) * F.CELL_M)
    px = np.clip(np.round(ax + s * xs).astype(int), 0, img.shape[1] - 1)
    py = np.clip(np.round(ay - s * ys).astype(int), 0, img.shape[0] - 1)
    patch = img[py[:, None], px[None, :]]
    gray, alpha = patch[..., 2].astype(float), patch[..., 3]
    return np.where(alpha > 0, np.round(gray / 255 * 254), 255).astype(np.uint8)


def main(gnn, sg_dir):
    split_sh = {s: pd.read_excel(gnn / "sample_parameters" / f"parameters_{s}.xlsx").ID.astype(int).tolist()
                for s in ("train", "test")}
    sh = {i: sh_graph(gnn / "graphs" / "json" / f"{i}.json") for i in split_sh["train"] + split_sh["test"]}
    package("shanghai", gnn / "results_gat", sh, split_sh["train"], split_sh["test"])
    graphs, split, _ = sg_graphs.load_all(sg_dir)
    package("singapore", gnn / "results_gat_sg", graphs, [i for i in graphs if split[i] == "train"],
            [i for i in graphs if split[i] == "test"])
    sg_windows(graphs, split)
    sg_library(sg_dir, graphs, split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default=r"D:\course\bps5231\daylight_gnn-20261112T101517Z-1-001\daylight_gnn")
    ap.add_argument("--sg", default=r"D:\course\bps5231\daylight\hdb_sg\sg_layout_dataset")
    a = ap.parse_args()
    main(Path(a.gnn), Path(a.sg))
