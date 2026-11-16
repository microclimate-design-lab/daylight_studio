# Daylight Studio

Interactive bubble-diagram editor with live daylight prediction for residential units.
Draw (or pick) a boundary, place room bubbles, set the window-to-wall ratios, and the
MLP / XGBoost surrogates return the DA map, room-level sDA and layout-level sDA in ~40 ms.

```
boundary + bubbles (browser) ─► graph (dataset format) ─► window model ─► DA map · room sDA · layout sDA
```

**Live demo:** https://huggingface.co/spaces/Yinerrr/daylight-studio (runs in the browser, nothing to install)

New users: see **[HANDOVER.md](HANDOVER.md)** for installation, a walkthrough, limits and troubleshooting.
HANDOVER.md was written for the first release (Shanghai, MLP / XGBoost); the GNN models and the Singapore
dataset follow the same workflow.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5231. Use `HOST=0.0.0.0` to serve on the local network.

With Docker (same image as the Hugging Face Space):

```bash
docker build -t daylight-studio .
docker run -p 7860:7860 daylight-studio
```

Then open http://localhost:7860. The Space is deployed from this repository: `space/README.md` becomes the
Space's README, `Dockerfile` + `requirements-space.txt` define the image.

## Use

| Area | Function |
|---|---|
| Boundary | draw an orthogonal polygon (`B`), type a rectangle, or pick one of the 1691 dataset outlines (Library, filter train / val / test) |
| Bubbles | 9 room types; place (`R`), move / resize (`V`), link (`L`), delete (`D` tool, right-click, `Del`, × in the Rooms list) |
| Region | **Shanghai** (1691-layout library; XGBoost / MLP / GNN) or **Singapore** (117 HDB layouts; GNN) |
| Parameters | Shanghai: WWR south 0.20–0.50 · north 0.20–0.35 · Singapore: WWR N / S / E / W 0.20–0.45 (step 0.01); orientation fixed to 0° |
| Output | layout sDA (layout model and from the DA map), room sDA, DA map (grey / heat / pass-fail), graph features, `graph.json` export |
| Reference | an unchanged library sample at its simulated WWR shows the simulated sDA and DA map, plus an error map |

Room area is read from the bubble proportions (r² / Σr² × 1.0017 × boundary area). Windows are not drawn:
two XGBoost classifiers predict north / south windows per room unless fixed in the room editor.
A cell counts towards sDA when its DA value is above 127/255.

## Repository layout

| Path | Content |
|---|---|
| `app.py` | Flask server: `/api/meta`, `/api/library`, `/api/library/<id>`, `/api/predict` |
| `engine.py` | vector bubble diagram → graph, inference pipeline, library access, regions |
| `gnn.py` | GATv2 ensemble (5 seeds) from `daylight_gnn`, input encoding from `models/gnn/<region>/spec.json` |
| `daylight/` | `features.py`, `bubble_to_graph.py` copied unchanged from `daylight_gnn/bubble_pipeline` (feature definitions shared with training) |
| `models/` | trained models from `daylight_gnn/bubble_pipeline/models` (window, room, unit, DA map; MLP + XGBoost) |
| `data/` | packed sample library (`library.json`, `samples.json`, `da_grids.npz`) and test metrics |
| `tools/build_data.py` | rebuilds the Shanghai `data/` from a `daylight_gnn` checkout |
| `tools/build_gnn.py`, `tools/sg_graphs.py` | package the GNN checkpoints into `models/gnn/`, re-run them on their test sets (each seed must reproduce the saved predictions), train the Singapore window classifiers and build `data/sg_*` from `sg_layout_dataset` |
| `static/` | single-page front end (no build step) |

Models and features must stay in sync with the training code: after retraining in `daylight_gnn`, copy the new
`bubble_pipeline/models/*` and `features.py` here. Training code and the full dataset live in the `daylight_gnn` project.

## Model performance (test set, 253 layouts)

| | layout sDA R² | room sDA R² | DA map R² |
|---|---|---|---|
| XGBoost | 0.883 | 0.930 | 0.874 |
| MLP | 0.840 | 0.915 | 0.854 |

GNN (GATv2, mean of 5 seeds), re-run with `tools/build_gnn.py` on the dataset graphs:

| | test layouts | layout sDA R² | layout MAE | room sDA R² |
|---|---|---|---|---|
| Shanghai (`results_gat`, R6_environ) | 253 | 0.926 | 2.6 pt | 0.968 |
| Singapore (`results_gat_sg`, R6_window) | 17 | 0.647 | 7.4 pt | 0.902 |

The GNN reads links as shared walls. Its DA map comes from a pixel-level GNN trained here (`tools/train_gnn_map.py`,
`gnn.MapGATv2`): the GATv2 embedding of the room a 0.1 m cell belongs to, the globals and the 43 cell features of the
XGBoost / MLP map model go through a cell decoder; trained on the library samples as the studio builds them (plan's
windows, simulated DA grids, room and layout labels as auxiliary targets), 5 seeds each. Same test cells for every model:

| DA map, test layouts | windows | map R² | map MAE | layout sDA from map R² | MAE |
|---|---|---|---|---|---|
| Shanghai GNN | predicted | 0.882 | 0.062 | 0.912 | 2.7 pt |
| Shanghai XGBoost | predicted | 0.874 | 0.072 | 0.893 | 3.0 pt |
| Shanghai MLP | predicted | 0.854 | 0.067 | 0.892 | 3.0 pt |
| Shanghai GNN | plan | 0.902 | 0.056 | 0.945 | 2.1 pt |
| Singapore GNN | plan | 0.634 | 0.165 | 0.817 | 5.3 pt |
| Singapore GNN | predicted | 0.353 | 0.226 | 0.289 | 10.0 pt |

Full tables: `models/gnn/<region>/map_test_metrics.csv`. Through drawn bubbles the room / layout
numbers depend on the windows: on the Singapore test layouts the layout MAE is 7.4 pt with the plan's windows but
10.9 pt with the window classifiers (east / west windows are hard to infer from a bubble diagram), so Singapore
library samples load with their real windows fixed.

Known limitation: the DA map model assigns cells to bubbles by distance to the bubble centre, so predicted
maps show circular patterns instead of orthogonal room boundaries (bubble diagrams carry no wall positions).
