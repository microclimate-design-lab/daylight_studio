"""
Train the pixel-level GNN (gnn.MapGATv2): GATv2 room embeddings + a cell decoder -> DA on the 0.1 m grid.

    python tools/train_gnn_map.py --region shanghai
    python tools/train_gnn_map.py --region singapore

Inputs are built exactly as in the studio: library sample (outline, bubbles, links, WWR, ORI) -> engine.build_graph
-> GNN graph + features.cell_features; targets = the simulated DA grid of the sample (data/*da_grids.npz), the room
labels and the layout label. Windows = the plan's windows during training. Split = the library split
(train / val / test, the same split as the GNN and XGBoost / MLP models). Early stopping on the validation cells.

Writes models/gnn/<region>/map_s*.pt, map_spec.json (architecture, scalers, test metrics) and
models/gnn/<region>/map_test_metrics.csv: GNN map vs the XGBoost / MLP map models on the same test cells,
once with the plan's windows and once with the windows the studio predicts.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine  # noqa: E402
import gnn as G  # noqa: E402

CUT = engine.SDA_CUT
BOUNDARY = ["n_rooms", "W_bbox", "H_bbox", "AR_bbox", "A_boundary", "P_boundary", "compactness"]


def spec_for(region):
    base = json.load(open(ROOT / "models" / "gnn" / region / "spec.json"))  # same inputs as the reproduced GNN
    glob = [k for k in base["glob"] if k != "VLT"]                      # VLT is not a design input
    return {"node": base["node"], "glob": glob, "edge": ["dx", "dy"], "categorical": base["categorical"]}


POOL = {"train": 2500, "val": 1500}   # cells kept per layout (test: all cells)


def load_layouts(region):
    """Masks are large (200 px / m), so only the cell pools are kept; evaluate() rebuilds the test geometry."""
    reg = engine.REGIONS[region]
    lib = engine.Library(prefix=reg["library"])
    sides = reg["sides"]
    rng = np.random.default_rng(0)
    out = []
    for x in lib.index:
        s = lib.samples[str(x["id"])]
        p = s["params"]
        wwr = {k: p.get(f"wwr_{k.lower()}") for k in "NSEW"}
        g, mask, bbox, order, _, _ = engine.build_graph(s["outline"], s["rooms"], s["links"], p.get("ori", 0), p["wwr_s"], p["wwr_n"])
        by_id = {r["id"]: r for r in s["rooms"]}
        win = np.array([[float(c in (by_id[cid]["true_win"] or "")) for c in sides] for cid in order])
        grid = lib.grids[f"g{x['id']}"]
        X, inside, room = engine.gnn_cell_features(region, g, bbox, mask, win, wwr, shape=grid.shape)
        y = grid.ravel().astype(np.float32) / 254.0
        ok = np.flatnonzero(inside.ravel() & (grid.ravel() != 255) & (room.ravel() >= 0))
        if x["split"] in POOL and len(ok) > POOL[x["split"]]:
            ok = np.sort(rng.choice(ok, POOL[x["split"]], replace=False))
        y_node = np.array([[by_id[c]["true_sda"] if by_id[c]["true_sda"] is not None else np.nan,
                            by_id[c]["true_mean_da"] if by_id[c]["true_mean_da"] is not None else np.nan] for c in order], np.float32)
        out.append(dict(id=x["id"], split=x["split"], sample=s, order=order, wwr=wwr, win=win,
                        graph=engine.gnn_graph(region, g, win, wwr), X=X[ok], y=y[ok], room=room.ravel()[ok],
                        cells=ok, shape=grid.shape, y_node=y_node,
                        y_graph=np.array([s["truth"]["sda"], s["truth"]["mean_da"]], np.float32)))
    return out


def batches(items, spec, cells_per, rng, bs):
    order = rng.permutation(len(items))
    for i in range(0, len(order), bs):
        chunk = [items[k] for k in order[i:i + bs]]
        yield collate(chunk, spec, cells_per, rng)


def collate(chunk, spec, cells_per, rng):
    batch = Batch.from_data_list([it["data"] for it in chunk])
    xs, nodes, ys = [], [], []
    for j, it in enumerate(chunk):
        idx = np.arange(len(it["y"])) if cells_per is None else rng.choice(len(it["y"]), min(cells_per, len(it["y"])), replace=False)
        xs.append(it["Xs"][idx]); ys.append(it["y"][idx]); nodes.append(it["room"][idx] + int(batch.ptr[j]))
    y_node = torch.tensor(np.concatenate([it["y_node"] for it in chunk]))
    y_graph = torch.tensor(np.stack([it["y_graph"] for it in chunk]))
    return batch, torch.cat(xs), torch.tensor(np.concatenate(nodes)), torch.tensor(np.concatenate(ys)), y_node, y_graph


def loss_fn(out, y_cell, y_node, y_graph, aux=0.5):
    yn, yg, yc = out
    ok = ~torch.isnan(y_node)
    return TF.mse_loss(yc, y_cell) + aux * (TF.mse_loss(yn[ok], y_node[ok]) + TF.mse_loss(yg, y_graph))


def train_seed(items, spec, arch, seed, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = G.MapGATv2(arch["in_node"], arch["in_edge"], arch["in_glob"], arch["in_cell"], hidden=arch["hidden"],
                   heads=arch["heads"], cell_hidden=tuple(arch["cell_hidden"]))
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 2)
    train = [it for it in items if it["split"] == "train"]
    val = [it for it in items if it["split"] == "val"]
    vrng = np.random.default_rng(123)
    val_batches = [collate(val[i:i + 64], spec, 1500, vrng) for i in range(0, len(val), 64)]
    best, best_state, bad = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        m.train()
        t0 = time.time()
        for b, xc, nc, yc, yn, yg in batches(train, spec, args.cells, rng, args.batch):
            opt.zero_grad()
            loss = loss_fn(m(b, xc, nc), yc, yn, yg, args.aux)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            se = n = 0.0
            for b, xc, nc, yc, yn, yg in val_batches:
                se += float(((m(b, xc, nc)[2] - yc) ** 2).sum()); n += len(yc)
        v = se / n
        sched.step(v)
        if v < best - 1e-6:
            best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
        if epoch % 5 == 0 or bad == 0:
            print(f"  seed {seed} epoch {epoch:3d} val cell MSE {v:.5f} (best {best:.5f}) {time.time() - t0:.1f} s/epoch", flush=True)
        if bad >= args.patience:
            break
    m.load_state_dict(best_state)
    return m.eval(), best, epoch


def evaluate(region, items, ens, models_ref, windows):
    """Test metrics of the GNN map and (Shanghai) the XGBoost / MLP map on identical cells.
    windows = "plan" (the sample's windows) or "studio" (window model, as in the app)."""
    rows = {}
    room_rows = []
    for it in items:
        s, win = it["sample"], it["win"]
        p = s["params"]
        g, mask, bbox, _, _, _ = engine.build_graph(s["outline"], s["rooms"], s["links"], p.get("ori", 0), p["wwr_s"], p["wwr_n"])
        if windows == "studio":
            if region == "shanghai":
                pw = np.column_stack([models_ref.win[s].predict_proba(engine.F.window_features(g))[:, 1] for s in "NS"])
            else:
                pw = engine.sg_window_probs(models_ref, g)
            win = (pw > 0.5).astype(float)
        X, inside, room = engine.gnn_cell_features(region, g, bbox, mask, win, it["wwr"], shape=it["shape"])
        Xc, rc = X[it["cells"]], room.ravel()[it["cells"]]
        preds = {"GNN": ens.predict_map(engine.gnn_graph(region, g, win, it["wwr"]), Xc, rc)[0]}
        if region == "shanghai":
            pwin = win if windows == "plan" else pw
            Xs, _, _ = engine.F.cell_features(g, list(bbox), mask, pwin, shape=it["shape"])
            Xs = Xs[it["cells"]]
            preds["XGBoost"] = models_ref.run_xgb("map_xgb", Xs)
            preds["MLP"] = np.clip(models_ref.run_mlp("map", Xs)[:, 0], 0, 1)
        for name, p in preds.items():
            r = rows.setdefault(name, {"y": [], "p": [], "lay_t": [], "lay_p": []})
            r["y"].append(it["y"]); r["p"].append(p)
            r["lay_t"].append(it["y_graph"][0]); r["lay_p"].append(float((p > CUT).mean()))
            for k in range(len(it["order"])):
                sel = rc == k
                if sel.sum() >= 5 and not np.isnan(it["y_node"][k, 0]):
                    room_rows.append((name, it["y_node"][k, 0], float((p[sel] > CUT).mean())))
    out = []
    rm = pd.DataFrame(room_rows, columns=["model", "t", "p"])
    for name, r in rows.items():
        y, p = np.concatenate(r["y"]), np.concatenate(r["p"])
        lt, lp = np.array(r["lay_t"]), np.array(r["lay_p"])
        q = rm[rm.model == name]
        out.append(dict(model=name, windows=windows, cells=len(y), map_MAE=float(np.abs(p - y).mean()),
                        map_R2=float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                        cell_sDA_accuracy=float(((p > CUT) == (y > CUT)).mean()),
                        layout_sDA_from_map_MAE=float(np.abs(lp - lt).mean()),
                        layout_sDA_from_map_R2=float(1 - ((lp - lt) ** 2).sum() / ((lt - lt.mean()) ** 2).sum()),
                        room_sDA_from_map_MAE=float(np.abs(q.p - q.t).mean()),
                        room_sDA_from_map_R2=float(1 - ((q.p - q.t) ** 2).sum() / ((q.t - q.t.mean()) ** 2).sum())))
    return out


def main(args):
    region = args.region
    folder = Path(args.folder) if args.folder else ROOT / "models" / "gnn" / region
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = load_layouts(region)
    print(f"{region}: {len(items)} layouts, {sum(len(it['y']) for it in items)} cells ({time.time() - t0:.0f} s)", flush=True)
    spec = spec_for(region)
    train = [it for it in items if it["split"] == "train"]
    spec["scaler"] = G.fit_scaler([it["graph"] for it in train], spec)
    Xtr = np.concatenate([it["X"] for it in train])
    spec["cell_scaler"] = [Xtr.mean(0).tolist(), (Xtr.std(0) + 1e-6).tolist()]
    for it in items:
        it["data"] = G.to_data(it["graph"], spec)
        it["Xs"] = G.cell_inputs(it.pop("X"), spec)
    ex = items[0]["data"]
    spec["arch"] = {"in_node": ex.x.shape[1], "in_edge": ex.edge_attr.shape[1], "in_glob": ex.u.shape[1],
                    "in_cell": items[0]["Xs"].shape[1], "hidden": args.hidden, "heads": 4, "cell_hidden": [256, 128]}
    for old in folder.glob("map_s*.pt"):
        old.unlink()
    spec["training"] = {"seeds": [], "cells_per_layout_per_step": args.cells, "lr": args.lr, "batch": args.batch,
                        "split": {s: sum(it["split"] == s for it in items) for s in ("train", "val", "test")},
                        "windows": "plan's windows", "targets": "simulated DA grid (0.1 m), room labels, layout label"}
    spec["training"]["aux_weight"] = args.aux
    for seed in range(args.seeds):
        m, best, epochs = train_seed(items, spec, spec["arch"], seed, args)
        torch.save(m.state_dict(), folder / f"map_s{seed}.pt")
        spec["training"]["seeds"].append({"seed": seed, "val_cell_mse": best, "epochs": epochs})
    json.dump(spec, open(folder / "map_spec.json", "w"))
    ens = G.MapEnsemble(folder)
    if args.folder:                                          # tuning run: validation score only
        print("val cell MSE", [round(t["val_cell_mse"], 5) for t in spec["training"]["seeds"]])
        return
    models_ref = engine.Models()
    test = [it for it in items if it["split"] == "test"]
    res = evaluate(region, test, ens, models_ref, "plan") + evaluate(region, test, ens, models_ref, "studio")
    df = pd.DataFrame(res)
    df.to_csv(folder / "map_test_metrics.csv", index=False)
    spec["metrics"] = res
    json.dump(spec, open(folder / "map_spec.json", "w"))
    print(df.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=list(engine.REGIONS), required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--cells", type=int, default=384, help="cells per layout per step")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--aux", type=float, default=0.5, help="weight of the room / layout losses")
    ap.add_argument("--folder", help="output folder (default models/gnn/<region>); the base spec.json is read from there")
    main(ap.parse_args())
