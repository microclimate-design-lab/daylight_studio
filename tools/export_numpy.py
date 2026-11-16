"""
Export the PyTorch models to NumPy weight files used by nn_numpy.py (the web app then needs no torch).

    python tools/export_numpy.py            # export
    python tools/export_numpy.py --check    # export, then compare NumPy and PyTorch predictions

Needs torch + torch_geometric (requirements.txt); the deployed app does not.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))


def export():
    (MODELS / "np").mkdir(exist_ok=True)
    for lvl in ["room", "unit", "map"]:
        m = torch.load(MODELS / f"{lvl}_mlp.pt", weights_only=False, map_location="cpu")
        sc = pickle.load(open(MODELS / f"{lvl}_scaler.pkl", "rb"))
        arrays, acts, layers = {}, [], list(m)
        for k, layer in enumerate(layers):
            if isinstance(layer, torch.nn.Linear):
                i = len(acts)
                arrays[f"W{i}"] = layer.weight.detach().numpy()
                arrays[f"b{i}"] = layer.bias.detach().numpy()
                nxt = [type(x).__name__ for x in layers[k + 1:k + 3]]
                acts.append("relu" if "ReLU" in nxt[:1] else "sigmoid" if "Sigmoid" in nxt[:1] else "none")
        np.savez(MODELS / "np" / f"{lvl}_mlp.npz", acts=np.array(acts), mean=sc.mean_, scale=sc.scale_, **arrays)
        print(lvl, "MLP", acts)
    for folder in sorted((MODELS / "gnn").iterdir()):
        for pt in sorted(folder.glob("*_s*.pt")):
            sd = torch.load(pt, map_location="cpu", weights_only=True)
            np.savez(pt.with_suffix(".npz"), **{k: v.numpy() for k, v in sd.items()})
            print(pt.relative_to(MODELS), "->", pt.with_suffix(".npz").name)


def check(n_samples=40):
    """End to end: the app's predict() with the torch backend vs the numpy backend on library samples."""
    import engine
    Mt, Mn = engine.Models("torch"), engine.Models("numpy")
    for region, reg in engine.REGIONS.items():
        lib = engine.Library(prefix=reg["library"])
        for model in reg["models"]:
            worst = {"unit": 0.0, "room": 0.0, "map": 0.0}
            for item in lib.index[:n_samples]:
                s = lib.sample(item["id"])
                body = dict(outline=s["outline"], rooms=s["rooms"], links=s["links"], model=model, region=region,
                            wwr_s=s["params"].get("wwr_s", 0.35), wwr_n=s["params"].get("wwr_n", 0.25))
                a, b = engine.predict(Mt, dict(body)), engine.predict(Mn, dict(body))
                worst["unit"] = max(worst["unit"], abs(a["unit"]["sda"] - b["unit"]["sda"]))
                worst["room"] = max(worst["room"], max(abs(x["sda"] - y["sda"]) for x, y in zip(a["rooms"], b["rooms"])))
                va, vb = np.asarray(a["da_map"]["values"], float), np.asarray(b["da_map"]["values"], float)
                worst["map"] = max(worst["map"], float(np.abs(va - vb).max()))
            print(f"{region:9s} {model:7s} max |torch - numpy|  layout sDA {worst['unit']:.1e}  "
                  f"room sDA {worst['room']:.1e}  DA map {worst['map']:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    export()
    if a.check:
        check()
