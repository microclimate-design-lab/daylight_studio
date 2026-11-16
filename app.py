"""
Daylight Studio – interactive bubble diagram -> graph -> daylight prediction (MLP / XGBoost / GNN; Shanghai and Singapore).

    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:5231
"""
import csv
import os
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

import engine

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")

print("loading models from", engine.MODELS)
M = engine.Models()
LIB = {region: engine.Library(prefix=r["library"]) for region, r in engine.REGIONS.items()}
print("library:", {region: len(lib.index) for region, lib in LIB.items()})


def _metrics():
    metrics = {}
    try:
        for row in csv.DictReader(open(engine.DATA / "metrics_sda_test.csv")):
            metrics.setdefault(row["model"], {})[row["level"] + "_sda_R2"] = float(row["sda_R2_mean"])
            metrics[row["model"]][row["level"] + "_sda_MAE"] = float(row["sda_MAE_mean"])
        for row in csv.DictReader(open(engine.DATA / "metrics_map_test.csv")):
            metrics.setdefault(row["model"], {})["map_R2"] = float(row["R2"])
            metrics[row["model"]]["map_MAE"] = float(row["MAE"])
    except FileNotFoundError:
        pass
    for region, ens in M.gnn.items():
        m = ens.spec["metrics"]
        metrics[f"{region}:GNN"] = {
            "unit_sda_R2": m["ensemble"]["layout_sda"]["r2"], "unit_sda_MAE": m["ensemble"]["layout_sda"]["mae"],
            "room_sda_R2": m["ensemble"]["room_sda"]["r2"], "room_sda_MAE": m["ensemble"]["room_sda"]["mae"],
            "test_layouts": m["test_layouts"], "test_rooms": m["test_rooms"], "seed_mean": m["seed_mean"],
            "window_auc": {s: v["auc"] for s, v in ens.spec.get("window_model", {}).get("test", {}).items()},
        }
        mm = [r for r in M.gnn_map[region].spec.get("metrics", []) if r["model"] == "GNN" and r["windows"] == "studio"]             if region in M.gnn_map else []
        if mm:                                                       # map scored with the windows the app predicts
            metrics[f"{region}:GNN"].update(map_R2=mm[0]["map_R2"], map_MAE=mm[0]["map_MAE"])
    return metrics


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "room_types": [{"id": i, "name": n, "rgb": list(c)} for i, n, c in engine.ALL_TYPES],
        "models": list(engine.MODEL_NAMES), "metrics": _metrics(),
        "regions": [{"id": region, "name": r["name"], "models": r["models"], "sides": r["sides"], "types": r["types"],
                     "wwr": r["wwr"], "library_size": len(LIB[region].index)} for region, r in engine.REGIONS.items()],
        "map_metrics": {region: m.spec.get("metrics", []) for region, m in M.gnn_map.items()},
        "ranges": {"wwr_s": [0.2, 0.5, 0.01], "wwr_n": [0.2, 0.35, 0.01]},
        "sda_cut": engine.SDA_CUT, "models_dir": str(engine.MODELS),
    })


def _lib():
    return LIB.get(request.args.get("region", "shanghai"), LIB["shanghai"])


@app.get("/api/library")
def library():
    return jsonify(_lib().index)


@app.get("/api/library/<int:i>")
def library_item(i):
    try:
        return jsonify(_lib().sample(i))
    except FileNotFoundError:
        return jsonify({"error": f"sample {i} not found"}), 404


@app.post("/api/predict")
def predict():
    t0 = time.perf_counter()
    try:
        out = engine.predict(M, request.get_json(force=True))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(out)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5231)),
            threaded=True, debug=False)
