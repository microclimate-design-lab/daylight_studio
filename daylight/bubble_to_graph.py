"""
Bubble diagram (PNG) -> graph in the dataset format (graphs/json/<id>.json).

The bubble diagrams follow the drawing rules of draw_bubble_diagram() in graph_functions.ipynb:
  white canvas, outline = black polyline (14 px) on the wall centre-line, one bubble per room
  (fill = nominal room-type colour, dark-grey edge 8 px, radius = 0.55 * sqrt(area/pi), shrunk to stay
  inside the outline, at least 36 px), connections = dark-grey lines (10 px) between bubble centres,
  centre = black dot (radius 18 px). Scale: 1 m = 199.735 px (same canvas as the layout images).
  Expected input: images/bubble_diagrams_uniform/ (render_uniform_bubbles.py) where all bubbles of a
  diagram share one scale. Room areas are taken from the bubble-area PROPORTIONS, so any drawing scale works.

Recovered graph (same names / order / units as the dataset):
  graph_features : W_bbox, H_bbox, AR_bbox, A_boundary, P_boundary, compactness, ORI, WWR_S, WWR_N, VLT
                   (ORI / WWR are design inputs, not drawn -> passed as arguments; VLT is kept as a
                    placeholder only and is not used by the models)
  node_features  : room_function, area, area_ratio, x, y, d_N, d_S, d_E, d_W, win_N, win_S
                   win_N / win_S are not drawn either: they are left as NaN here and filled by the
                   window model (train_models.py) or given by the user.
  edge_index / edge_attr (dx, dy): from the connection lines.
Room area: area_k = r_k^2 / sum(r^2) * 1.0017 * A_boundary, using the dataset relation
sum(room areas) = 1.0017 * A_boundary (median, 5-95 % 1.0012-1.0025).

Usage
  python bubble_pipeline/bubble_to_graph.py path/to/bubble.png --ori 0 --wwr-s 0.3 --wwr-n 0.25
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

PX_PER_M = 199.735
ROOM_TYPES = [  # id, name, nominal RGB = bubble fill colour
    (0, "Living room", (143, 199, 196)), (1, "Master room 0", (87, 141, 202)),
    (2, "Master room 1", (139, 134, 190)), (3, "Master room 2", (0, 154, 142)),
    (4, "Dining room", (30, 59, 145)), (5, "Kitchen", (255, 188, 64)),
    (6, "Bathroom", (49, 156, 174)), (7, "Study room", (141, 195, 242)),
    (8, "Storage", (199, 200, 232)),
]
BUBBLE_SCALE, BUBBLE_FIT, BUBBLE_STROKE = 0.55, 0.95, 8
BUBBLE_LINE, CENTRE_RADIUS, OUTLINE_LINE = 10, 18, 14
MIN_R = 2 * CENTRE_RADIUS
AREA_SUM_RATIO = 1.0017
GRAPH_FEATURE_NAMES = ["W_bbox", "H_bbox", "AR_bbox", "A_boundary", "P_boundary",
                       "compactness", "ORI", "WWR_S", "WWR_N", "VLT"]
NODE_FEATURE_NAMES = ["room_function", "area", "area_ratio", "x", "y",
                      "d_N", "d_S", "d_E", "d_W", "win_N", "win_S"]


def _read(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:                      # composite RGBA onto white
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.int16)


def _outline(img):
    """Building outline -> filled mask on the wall centre-line, polygon, bbox."""
    black = (img.max(axis=2) < 30).astype(np.uint8)
    # connection lines (10 px, dark grey) drawn across the outline cut it: close those gaps first
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    # the outline is the black component touching the most area; centre dots are small blobs inside
    n, lab, st, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_WIDTH] * st[1:, cv2.CC_STAT_HEIGHT]))
    line = (lab == k).astype(np.uint8)
    contours, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = np.zeros_like(line)
    cv2.drawContours(outer, contours, -1, 1, thickness=cv2.FILLED)
    half = OUTLINE_LINE // 2
    mask = cv2.erode(outer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * half + 1, 2 * half + 1)))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [max(cs, key=cv2.contourArea)]
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, cs, -1, 1, thickness=cv2.FILLED)
    return mask, cs


def _centres(img, outline_mask):
    """Centre dots: black blobs that survive an opening wider than the outline line."""
    black = (img.max(axis=2) < 30).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (OUTLINE_LINE // 2 + 4) + 1,) * 2)
    blobs = cv2.morphologyEx(black, cv2.MORPH_OPEN, k)
    n, lab, st, cen = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    pts = []
    for j in range(1, n):
        area = st[j, cv2.CC_STAT_AREA]
        if 0.4 * math.pi * CENTRE_RADIUS ** 2 < area < 2.5 * math.pi * CENTRE_RADIUS ** 2:
            cx, cy = cen[j]
            if outline_mask[int(round(cy)), int(round(cx))]:
                pts.append((float(cx), float(cy)))
    return pts


def _room_type(img, c):
    """Majority fill colour in an annulus just outside the centre dot (lines / strokes ignored)."""
    cx, cy = c
    yy, xx = np.mgrid[-40:41, -40:41]
    rr = np.hypot(xx, yy)
    ring = (rr > CENTRE_RADIUS + 2) & (rr < CENTRE_RADIUS + 8)     # small bubbles: fill is only ~6 px wide
    ys, xs = (yy[ring] + int(round(cy))), (xx[ring] + int(round(cx)))
    ok = (ys >= 0) & (ys < img.shape[0]) & (xs >= 0) & (xs < img.shape[1])
    px = img[ys[ok], xs[ok]]
    cols = np.array([t[2] for t in ROOM_TYPES], np.int16)
    d = np.abs(px[:, None, :] - cols[None]).max(axis=2)           # (n_px, 9)
    best, dist = d.argmin(1), d.min(1)
    votes = np.bincount(best[dist <= 12], minlength=len(cols))
    if votes.sum() == 0:
        raise ValueError(f"no room colour around centre {c}")
    return int(votes.argmax())


def _radius(img, c, rgb):
    """Bubble radius: rays from the centre until the fill colour ends; occlusion only shortens rays,
    so a high percentile of the ray lengths gives the visible edge (inner side of the 8 px stroke)."""
    cx, cy = c
    col = np.array(rgb, np.int16)
    lens = []
    rs = np.arange(CENTRE_RADIUS + 2, 1500)
    for a in np.linspace(0, 2 * np.pi, 180, endpoint=False):
        xs = np.round(cx + rs * np.cos(a)).astype(int)
        ys = np.round(cy - rs * np.sin(a)).astype(int)
        inb = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
        xs, ys = xs[inb], ys[inb]
        fill = np.abs(img[ys, xs] - col).max(axis=1) <= 14
        # skip the dark-grey connection line that may start at the centre: first fill pixel, then first non-fill
        first = np.argmax(fill) if fill.any() else None
        if first is None:
            continue
        rest = ~fill[first:]
        lens.append(rs[first + (np.argmax(rest) if rest.any() else len(rest) - 1)])
    if not lens:
        return float(MIN_R)
    return float(np.percentile(lens, 90)) + BUBBLE_STROKE / 2 - 1


def _ray_lengths(mask, cx, cy):
    """Same as centroid_distances() in graph_functions: N, S, E, W distance (px) inside the outline."""
    c, r = int(round(cx)), int(round(cy))
    col, row = mask[:, c], mask[r, :]

    def run(line, i, step):
        j = i
        while 0 <= j + step < len(line) and line[j + step]:
            j += step
        return abs(j - i) + 1
    if not mask[r, c]:
        return 0.0, 0.0, 0.0, 0.0
    return float(run(col, r, -1)), float(run(col, r, +1)), float(run(row, c, +1)), float(run(row, c, -1))


def _edges(img, pts):
    """Pairs whose centre-to-centre segment is drawn as a dark-grey connection line."""
    grey = (np.abs(img - 60).max(axis=2) <= 35)
    E = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (x0, y0), (x1, y1) = pts[i], pts[j]
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 3 * CENTRE_RADIUS:
                continue
            t = np.linspace((CENTRE_RADIUS + 4) / L, 1 - (CENTRE_RADIUS + 4) / L, max(int(L / 3), 10))
            xs, ys = np.round(x0 + t * (x1 - x0)).astype(int), np.round(y0 + t * (y1 - y0)).astype(int)
            # tolerate 2 px of anti-aliasing: a sample counts if any pixel in a 3x3 patch is grey
            hit = np.zeros(len(t), bool)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    hit |= grey[np.clip(ys + dy, 0, grey.shape[0] - 1), np.clip(xs + dx, 0, grey.shape[1] - 1)]
            if hit.mean() < 0.9:
                continue
            # a straight chain i - k - j would also look like a line i - j: skip if a third centre sits on it
            on_line = False
            for k, (xk, yk) in enumerate(pts):
                if k in (i, j):
                    continue
                tk = ((xk - x0) * (x1 - x0) + (yk - y0) * (y1 - y0)) / L ** 2
                if 0 < tk < 1 and abs((x1 - x0) * (y0 - yk) - (x0 - xk) * (y1 - y0)) / L < CENTRE_RADIUS:
                    on_line = True
            if not on_line:
                E.append((i, j))
    return E


def parse_bubble_diagram(path, ori=0, wwr_s=0.35, wwr_n=0.275, win=None):
    """Return (graph dict in dataset format, debug dict). `win` optionally gives {node: (win_N, win_S)}."""
    img = _read(path)
    mask, contours = _outline(img)
    bx, by, W, H = cv2.boundingRect(mask)
    A_px = float(mask.sum())
    P_px = float(sum(cv2.arcLength(c, True) for c in contours))
    pts = _centres(img, mask)
    if not pts:
        raise ValueError("no bubble centres found")
    to_outline = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    rooms = []
    for c in pts:
        try:
            t = _room_type(img, c)
        except ValueError:          # black blob without a bubble around it -> not a room centre
            continue
        r = _radius(img, c, ROOM_TYPES[t][2])
        rooms.append(dict(type=t, c=c, r=r))

    # room areas from the bubble-area proportions (bubbles are drawn to one scale per diagram):
    #   area_k = r_k^2 / sum(r^2) * 1.0017 * A_boundary      -> independent of the drawing scale
    A_m2 = A_px / PX_PER_M ** 2
    r2 = sum(r["r"] ** 2 for r in rooms)
    for r in rooms:
        r["area"] = r["r"] ** 2 / r2 * AREA_SUM_RATIO * A_m2
    scale = math.sqrt(r2 / (AREA_SUM_RATIO * A_px / math.pi))   # drawing scale (0.55 if unconstrained)

    # node order as in the dataset: by room type, then north -> south, west -> east
    rooms.sort(key=lambda r: (r["type"], round(r["c"][1] / 50), r["c"][0]))
    A_tot = sum(r["area"] for r in rooms)
    nodes = []
    for k, r in enumerate(rooms):
        cx, cy = r["c"]
        dN, dS, dE, dW = (d / PX_PER_M for d in _ray_lengths(mask, cx, cy))
        wn, ws = (win.get(k, (np.nan, np.nan)) if win else (np.nan, np.nan))
        nodes.append([r["type"], r["area"], r["area"] / A_tot, (cx - bx) / W, (by + H - cy) / H,
                      dN, dS, dE, dW, wn, ws])
    E = _edges(img, [r["c"] for r in rooms])
    ei = [[], []]
    ea = []
    for i, j in E:
        for a, b in ((i, j), (j, i)):
            ei[0].append(a); ei[1].append(b)
            ea.append([nodes[b][3] - nodes[a][3], nodes[b][4] - nodes[a][4]])
    graph = {
        "num_nodes": len(nodes), "num_edges": len(ea),
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "graph_features": [W / PX_PER_M, H / PX_PER_M, W / H, A_m2, P_px / PX_PER_M,
                           4 * math.pi * A_px / P_px ** 2, int(ori), float(wwr_s), float(wwr_n), float("nan")],
        "node_feature_names": NODE_FEATURE_NAMES, "node_features": nodes,
        "edge_feature_names": ["dx", "dy"], "edge_index": ei, "edge_attr": ea,
        "nodes": [{"index": k, "room": ROOM_TYPES[r["type"]][1]} for k, r in enumerate(rooms)],
        "bubble_scale": scale,
    }
    debug = {"bbox_px": [bx, by, W, H], "centres_px": [r["c"] for r in rooms],
             "radius_px": [r["r"] for r in rooms], "outline_mask": mask}
    return graph, debug


def to_json(graph):
    """JSON-safe copy (NaN -> null)."""
    def fix(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        return v
    return fix(graph)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="bubble diagram -> graph JSON")
    ap.add_argument("image")
    ap.add_argument("--ori", type=int, default=0)
    ap.add_argument("--wwr-s", type=float, default=0.35)
    ap.add_argument("--wwr-n", type=float, default=0.275)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g, _ = parse_bubble_diagram(a.image, a.ori, a.wwr_s, a.wwr_n)
    out = a.out or str(Path(a.image).with_suffix(".graph.json"))
    json.dump(to_json(g), open(out, "w"), indent=1)
    print(f"{g['num_nodes']} rooms, {g['num_edges'] // 2} connections -> {out}")
