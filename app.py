"""
Daylight Studio – interactive bubble diagram -> graph -> daylight prediction (MLP / XGBoost).

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
LIB = engine.Library()
print(f"library: {len(LIB.index)} boundaries")


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
    return metrics


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "room_types": [{"id": i, "name": n, "rgb": list(c)} for i, n, c in engine.ROOM_TYPES],
        "models": list(engine.MODEL_NAMES), "metrics": _metrics(),
        "ranges": {"wwr_s": [0.2, 0.5, 0.01], "wwr_n": [0.2, 0.35, 0.01]},
        "sda_cut": engine.SDA_CUT, "models_dir": str(engine.MODELS),
    })


@app.get("/api/library")
def library():
    return jsonify(LIB.index)


@app.get("/api/library/<int:i>")
def library_item(i):
    try:
        return jsonify(LIB.sample(i))
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
