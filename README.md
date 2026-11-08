# Daylight Studio

Interactive bubble-diagram editor with live daylight prediction for residential units.
Draw (or pick) a boundary, place room bubbles, set the window-to-wall ratios, and the
MLP / XGBoost surrogates return the DA map, room-level sDA and layout-level sDA in ~40 ms.

```
boundary + bubbles (browser) ─► graph (dataset format) ─► window model ─► DA map · room sDA · layout sDA
```

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5231. Use `HOST=0.0.0.0` to serve on the local network.

## Use

| Area | Function |
|---|---|
| Boundary | draw an orthogonal polygon (`B`), type a rectangle, or pick one of the 1691 dataset outlines (Library, filter train / val / test) |
| Bubbles | 9 room types; place (`R`), move / resize (`V`), link (`L`), delete (`D` tool, right-click, `Del`, × in the Rooms list) |
| Parameters | model XGBoost / MLP · WWR south 0.20–0.50 · WWR north 0.20–0.35 (step 0.01); orientation fixed to 0° |
| Output | layout sDA (layout model and from the DA map), room sDA, DA map (grey / heat / pass-fail), graph features, `graph.json` export |
| Reference | an unchanged library sample at its simulated WWR shows the simulated sDA and DA map, plus an error map |

Room area is read from the bubble proportions (r² / Σr² × 1.0017 × boundary area). Windows are not drawn:
two XGBoost classifiers predict north / south windows per room unless fixed in the room editor.
A cell counts towards sDA when its DA value is above 127/255.

## Repository layout

| Path | Content |
|---|---|
| `app.py` | Flask server: `/api/meta`, `/api/library`, `/api/library/<id>`, `/api/predict` |
| `engine.py` | vector bubble diagram → graph, inference pipeline, library access |
| `daylight/` | `features.py`, `bubble_to_graph.py` copied unchanged from `daylight_gnn/bubble_pipeline` (feature definitions shared with training) |
| `models/` | trained models from `daylight_gnn/bubble_pipeline/models` (window, room, unit, DA map; MLP + XGBoost) |
| `data/` | packed sample library (`library.json`, `samples.json`, `da_grids.npz`) and test metrics |
| `tools/build_data.py` | rebuilds `data/` from a `daylight_gnn` checkout |
| `static/` | single-page front end (no build step) |

Models and features must stay in sync with the training code: after retraining in `daylight_gnn`, copy the new
`bubble_pipeline/models/*` and `features.py` here. Training code and the full dataset live in the `daylight_gnn` project.

## Model performance (test set, 253 layouts)

| | layout sDA R² | room sDA R² | DA map R² |
|---|---|---|---|
| XGBoost | 0.883 | 0.930 | 0.874 |
| MLP | 0.840 | 0.915 | 0.854 |

Known limitation: the DA map model assigns cells to bubbles by distance to the bubble centre, so predicted
maps show circular patterns instead of orthogonal room boundaries (bubble diagrams carry no wall positions).
