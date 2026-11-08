"""
Pack the boundary / sample library from a daylight_gnn checkout into data/ (run once, results are committed).

    python tools/build_data.py --gnn D:\\course\\bps5231\\daylight_gnn

Reads   graphs/debug/<id>.json      outline polygon (px)
        bubble_pipeline/parsed/<id>.json   parsed bubble diagram (rooms, links, bubble scale)
        graphs/json/<id>.json       simulated labels (layout / room sDA, DA grid)
        sample_parameters/parameters_{train,val,test}.xlsx   split
Writes  data/library.json   outline + summary per sample (library grid)
        data/samples.json   bubbles, links, parameters, simulated sDA per sample
        data/da_grids.npz   simulated DA grid per sample, uint8 (value * 254, 255 = outside)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PX = 199.735
OUT = Path(__file__).resolve().parents[1] / "data"


def main(gnn):
    split = {}
    for s in ("train", "val", "test"):
        for i in pd.read_excel(gnn / "sample_parameters" / f"parameters_{s}.xlsx").ID.astype(int):
            split[int(i)] = s
    lib, samples, grids = [], {}, {}
    for p in sorted((gnn / "graphs" / "debug").glob("*.json"), key=lambda q: int(q.stem)):
        i = int(p.stem)
        d = json.load(open(p))
        bx, by = d["bbox_px"][:2]
        poly = np.round((np.asarray(d["outline_px"][0], float) - [bx, by]) / PX, 3).tolist()
        g = json.load(open(gnn / "bubble_pipeline" / "parsed" / f"{i}.json"))
        t = json.load(open(gnn / "graphs" / "json" / f"{i}.json", encoding="utf-8"))
        gf = g["graph_features"]
        lib.append({"id": i, "outline": poly, "n_rooms": g["num_nodes"], "area": round(gf[3], 1),
                    "W": round(gf[0], 2), "H": round(gf[1], 2), "split": split.get(i, "?")})

        W, H = gf[0], gf[1]
        scale = g.get("bubble_scale") or 0.55
        true_node = g.get("true_node") or list(range(g["num_nodes"]))
        rooms = []
        for k, nf in enumerate(g["node_features"]):
            tn = true_node[k] if k < len(true_node) else None
            ok = tn is not None and 0 <= tn < len(t["node_labels"])
            lab = t["node_labels"][tn] if ok else None
            tnf = t["node_features"][tn] if ok else None
            rooms.append({
                "id": k + 1, "type": int(nf[0]),
                "x": round(nf[3] * W, 3), "y": round((1 - nf[4]) * H, 3),
                "r": round(scale * math.sqrt(nf[1] / math.pi), 3),
                "true_sda": lab[0] if lab else None, "true_mean_da": lab[1] if lab else None,
                "true_win": (("N" if tnf[9] else "") + ("S" if tnf[10] else "")) if tnf else None,
            })
        ei = g["edge_index"]
        links = sorted({(min(a, b) + 1, max(a, b) + 1) for a, b in zip(ei[0], ei[1])})
        tg = t["graph_features"]
        samples[str(i)] = {
            "id": i, "outline": poly, "rooms": rooms, "links": [list(l) for l in links],
            "params": {"ori": int(tg[6]), "wwr_s": round(tg[7], 2), "wwr_n": round(tg[8], 2), "vlt": tg[9]},
            "truth": {"sda": t["graph_labels"][0], "mean_da": t["graph_labels"][1]},
        }
        grid = np.array([[np.nan if v is None else v for v in r] for r in t["da_grid"]], float)
        grids[f"g{i}"] = np.where(np.isnan(grid), 255, np.round(np.clip(grid, 0, 1) * 254)).astype(np.uint8)
    OUT.mkdir(exist_ok=True)
    json.dump(lib, open(OUT / "library.json", "w"), separators=(",", ":"))
    json.dump(samples, open(OUT / "samples.json", "w"), separators=(",", ":"))
    np.savez_compressed(OUT / "da_grids.npz", **grids)
    print(f"{len(lib)} samples ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default=r"D:\course\bps5231\daylight_gnn")
    main(Path(ap.parse_args().gnn))
