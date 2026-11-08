# Daylight Studio — Handover

A browser tool for testing daylight performance at the bubble-diagram stage of a residential layout.
You draw a unit outline and place one bubble per room. Trained machine-learning models then predict,
in about 40 ms:

- **Layout-level sDA**: the share of the whole unit that receives enough daylight.
- **Room-level sDA**: the same figure for each room.
- **DA map**: daylight on a 0.1 m grid across the plan.

The models (MLP and XGBoost) were trained on 1691 simulated layouts from the `daylight_gnn` project.
They give a fast early-stage estimate and **do not replace a daylight simulation**.

---

## 1. What you need

| | |
|---|---|
| Python | 3.10 – 3.13 (tested on 3.13) |
| Disk | about 1.5 GB after installing PyTorch; the repository itself is about 70 MB |
| OS | Windows, macOS or Linux |
| Browser | any recent Chrome, Edge, Firefox or Safari |

The app runs **on your own computer**. There is no public website: the browser page talks to a small Python
server that you start yourself.

## 2. Install (once)

```bash
git clone https://github.com/microclimate-design-lab/daylight_studio.git
cd daylight_studio
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

If PyTorch is slow to install, the CPU-only build is enough:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`, then run the command above again.

## 3. Start

```bash
python app.py
```

Wait for `Running on http://127.0.0.1:5231` (5–10 s while the models load), then open
**http://127.0.0.1:5231** in a browser. Stop the server with `Ctrl + C`.

To let colleagues on the same network open it from their own machines, start it with `HOST=0.0.0.0`
(PowerShell: `$env:HOST="0.0.0.0"; python app.py`) and share `http://<your-IP>:5231`.
It only stays reachable while your computer and the server are running.

---

## 4. The screen

```
┌──────────────┬──────────────────────────────┬────────────────────┐
│ 01 Boundary  │                              │ Layout-level sDA   │
│ 02 Bubbles   │      plan editor (canvas)    │  big number, gauge │
│ 03 Model&WWR │                              │ Rooms | DA map |   │
│ 04 Display   │                              │ Graph | Model tabs │
└──────────────┴──────────────────────────────┴────────────────────┘
```

- **Left panel**: the inputs, worked through from top to bottom.
- **Centre**: the plan. North is up, the grid is 1 m (0.5 m when zoomed in), and the predicted DA map is drawn under the bubbles.
- **Right panel**: the results. They update automatically after every change; there is no "Run" button.

When the app opens, it loads library sample #0 as an example.

## 5. Typical workflows

### A. Start from a dataset layout

1. **Library…** (01 Boundary) opens all 1691 outlines. Filter by sample id, split (train / val / test) or number of rooms.
2. Choose how much of the sample to load:
   - **Click a card** to load the full sample: the outline, its bubbles and the simulation result.
   - **Click the small "outline" button** on a card to load only the boundary.
3. The box under the big number compares the prediction with the simulation, for example *"simulated sDA 47.5 % · error +0.9 pt"*.
   It only shows while the layout is unchanged and the WWR sliders match the sample's simulated values.
   If they don't match, click **Use the sample's simulated WWR**.
   Samples simulated at a rotated orientation (30° E or W) have no reference, because orientation is fixed to 0° in this tool.

> Use **test**-split samples when judging accuracy. Train samples were seen during training and look better than the models really are.

### B. Draw your own layout

1. **Boundary**: choose one of two ways.
   - Press **Draw** (`B`) and click the corners. Lines snap to 0.1 m and stay orthogonal; hold `Alt` for a free angle.
     Close the shape by clicking the first point, pressing `Enter` or double-clicking.
   - Press **Rect**, type the width and height in metres, then click **Apply**.
2. **Rooms**: pick a room type in the palette (or press keys `1`–`9`), then on the plan:
   - click to drop a bubble at its default size, or
   - press and drag to set the radius.
3. **Links** (optional): press `L`, then click two bubbles to connect or disconnect them.
4. Adjust the design (see section 6). Results update as you go.

### C. Test façade options

Move the **WWR south** slider (0.20–0.50) and the **WWR north** slider (0.20–0.35) in steps of 0.01, and watch
the sDA change. Switch **XGBoost / MLP** to compare the two models on the same design.

## 6. Editing reference

| Action | How |
|---|---|
| Select / move a bubble | Select tool (`V`), drag the bubble |
| Resize a bubble | drag the square handle on its right edge, or use the *Bubble radius* slider in the room editor |
| Change room type | select the bubble, then use the *Type* dropdown or press `1`–`9` |
| Delete a bubble | Delete tool (`D`) + click · right-click the bubble · select it and press `Del` · **×** next to it in the Rooms list |
| Delete a link | Delete tool (`D`) + click the line, or Link tool + click the two bubbles again |
| Move a wall | Select tool, drag a boundary edge (it moves perpendicular to itself) |
| Move a corner | Select tool, drag the square at the corner |
| Fix windows of a room | select the bubble → *Windows*: auto / N / S / N+S / none |
| Pan / zoom | drag empty space · mouse wheel · **Fit** button or `F` |
| Cancel | `Esc` |
| Start again | **Clear all** (04 Display) |

A bubble drawn with a dashed red outline has its centre outside the boundary. It is ignored, and a warning appears
at the bottom-left of the plan.

## 7. Reading the results

| Item | Meaning |
|---|---|
| **Layout-level sDA** (big number) | prediction of the layout model for the whole unit |
| sDA from DA map | share of predicted map cells above the threshold; a second, independent estimate |
| mean DA | average daylight value (0–1) from the layout model and from the map |
| Gauge red mark | simulated sDA, when a reference is available |
| **Rooms tab** | room sDA (bar), the value from the map, mean DA, and window probabilities (`p N`, `p S`) or `fixed` if you set them; red `sim` = simulated value |
| **DA map tab** | the map on the 0.1 m grid; hover a cell for its value and room. *simulated* and *error* views appear for unchanged library samples |
| **Graph tab** | the graph the models actually receive: global features, node features per room, edges |
| **Model tab** | test-set accuracy of both models |
| **Export graph.json** | downloads the graph plus the current prediction |

The sDA threshold follows the dataset: a cell counts as daylit when its DA value is above 127/255 (0.498).
The colour ramp (grey / heat / pass-fail) only changes the display.

## 8. How it works (short)

1. **Boundary** → rasterised at 199.735 px per metre, the same scale as the dataset images.
2. **Bubbles** → one node per room:
   - Room area = the bubble's share of Σr² × 1.0017 × boundary area. Only the **relative** bubble sizes matter.
   - The node also stores its position and its N/S/E/W depth to the outline.
3. **Windows** are not drawn. Two XGBoost classifiers predict whether each room has north and south windows (AUC ≈ 0.98).
4. **Room** and **layout** regressors predict sDA and mean DA. A **per-cell** model predicts the DA map.

Test-set accuracy (253 layouts):

| | layout sDA R² | room sDA R² | DA map R² |
|---|---|---|---|
| XGBoost | 0.883 | 0.930 | 0.874 |
| MLP | 0.840 | 0.915 | 0.854 |

## 9. Limits — read before using the numbers

- **Valid range**: residential units like the dataset, 1–10 rooms of the 9 types, WWR south 0.20–0.50, WWR north 0.20–0.35,
  orientation 0° (south-facing). Results outside this range are extrapolations.
- **Circular patterns in the DA map** are expected. A bubble diagram has no wall positions, so each map cell is
  assigned to a bubble by distance to its centre. Read the map as a trend, not as room-by-room detail at the walls.
- The same bubble diagram can correspond to several real plans with different daylight, so expect differences of
  a few sDA points from a simulation of any particular plan.
- **Links do not change the prediction.** The MLP and XGBoost models do not use edges; links are kept for the graph export.
- Glazing light transmittance (VLT) is not an input; it had no effect in the simulation data.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Page shows *server unreachable* | the Python server is not running, or it crashed — check the terminal and run `python app.py` again |
| `Address already in use` | another copy is running; stop it, or start on another port: `PORT=5232 python app.py` (PowerShell: `$env:PORT="5232"`) |
| `ModuleNotFoundError` | the virtual environment is not activated, or `pip install -r requirements.txt` was skipped |
| Error loading `*_scaler.pkl` | scikit-learn version mismatch — install `scikit-learn==1.6.*` |
| Status says *draw or load a boundary* / *place room bubbles* | the boundary is not closed yet, or there are no bubbles |
| Numbers do not change when moving a link | expected (see section 9) |

## 11. Repository layout and maintenance

| Path | Content |
|---|---|
| `app.py` | web server and API |
| `engine.py` | bubble diagram → graph → prediction |
| `daylight/` | feature code copied from `daylight_gnn/bubble_pipeline` — must match the code the models were trained with |
| `models/` | trained models (window, room, unit, DA map; MLP + XGBoost) |
| `data/` | sample library and test metrics |
| `tools/build_data.py` | rebuilds `data/` from a `daylight_gnn` checkout |
| `static/` | the web page (plain HTML / CSS / JS, no build step) |

**After retraining in `daylight_gnn`:**

1. Copy `bubble_pipeline/models/*` to `models/` and `bubble_pipeline/features.py` to `daylight/`.
2. Copy `results/metrics_sda_test.csv` and `metrics_map_test.csv` to `data/`.
3. Restart the server and commit.

The training code and the full dataset are not in this repository; they are in the `daylight_gnn` project.

### Using the API from a script

The server can also be called without the web page:

```python
import requests
r = requests.post("http://127.0.0.1:5231/api/predict", json={
    "model": "XGBoost", "wwr_s": 0.35, "wwr_n": 0.25,
    "outline": [[0, 0], [10, 0], [10, 8], [0, 8]],               # metres, x east, y south
    "rooms": [{"id": 1, "type": 0, "x": 4, "y": 4, "r": 2.0},     # type: 0 Living, 1-3 Master 0-2,
              {"id": 2, "type": 1, "x": 8, "y": 2, "r": 1.2},     # 4 Dining, 5 Kitchen, 6 Bathroom,
              {"id": 3, "type": 6, "x": 8, "y": 6, "r": 0.7}],    # 7 Study, 8 Storage
    "links": [[1, 2], [1, 3]],
    "windows": {"3": "-"},                                        # optional: N, S, NS or - per room id
})
res = r.json()
print(res["unit"]["sda"], [(x["name"], round(x["sda"], 3)) for x in res["rooms"]])
```

The response also contains `da_map` (rows × cols values; −1 = outside) and `graph` (the model input).
