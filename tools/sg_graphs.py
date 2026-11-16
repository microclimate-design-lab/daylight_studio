"""
Singapore HDB layouts (sg_layout_dataset/meta/<ID>.json) -> graphs in the GNN schema.

Same definitions as sg_build_graph() in the SG_Test notebook: outline = union of the room polygons,
room centroid, rays to the outline in metres, x / y relative to the outline bbox (y measured from the
south edge), windows assigned to the nearest room, edges = rooms sharing a wall. Unlike SG_Test,
room code 13 (Master room 3) is kept, and east / west windows and all four WWRs are stored, because
the Singapore model was trained on them.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

SIDES = "NSEW"


def outline_of(polys):
    for r in (0.03, 0.1, 0.3, 0.6, 1.0):                       # close slivers across interior walls, then fill holes
        u = unary_union([p.buffer(r, join_style=2) for p in polys]).buffer(-r, join_style=2)
        if u.geom_type == "Polygon":
            return Polygon(u.exterior)
    return Polygon(max(u.geoms, key=lambda g: g.area).exterior)


def _ray(outline, cx, cy, dx, dy, reach):
    seg = LineString([(cx, cy), (cx + dx * reach, cy + dy * reach)]).intersection(outline.exterior)
    parts = [seg] if seg.geom_type in ("Point", "LineString") else list(getattr(seg, "geoms", []))
    ds = []
    for g in parts:
        pts = [(g.x, g.y)] if g.geom_type == "Point" else list(g.coords) if g.geom_type == "LineString" else []
        ds += [math.hypot(px - cx, py - cy) for px, py in pts]
    ds = [d for d in ds if d > 1e-6]
    return min(ds) if ds else 0.0


def build(meta):
    """Returns the GNN graph plus the outline polygon (m) and room polygons for the sample library."""
    rooms = meta["rooms"]
    n = len(rooms)
    polys = [Polygon(r["polygon_m"]).buffer(0) for r in rooms]
    outline = outline_of(polys)
    minx, miny, maxx, maxy = outline.bounds
    W, H = maxx - minx, maxy - miny
    A, P = outline.area, outline.exterior.length
    reach = 2 * (W + H) + 10
    win = {s: np.zeros(n) for s in SIDES}
    ppx = [Polygon(r["polygon_px"]).buffer(0) for r in rooms]
    for w in meta["windows"]:
        mid = Point((w["p0_px"][0] + w["p1_px"][0]) / 2, (w["p0_px"][1] + w["p1_px"][1]) / 2)
        win[w["ori"]][int(np.argmin([pp.distance(mid) for pp in ppx]))] = 1
    areas = np.array([p.area for p in polys])
    node = {k: [] for k in ("room_function", "x", "y", "d_N", "d_S", "d_E", "d_W")}
    centres = []
    for r, p in zip(rooms, polys):
        c = p.centroid
        centres.append((c.x, c.y))
        node["room_function"].append(int(r["room_function"]))
        node["x"].append((c.x - minx) / W)
        node["y"].append((c.y - miny) / H)
        for key, (dx, dy) in zip(("d_N", "d_S", "d_E", "d_W"), ((0, 1), (0, -1), (1, 0), (-1, 0))):
            node[key].append(_ray(outline, c.x, c.y, dx, dy, reach))
    node["area"], node["area_ratio"] = areas, areas / areas.sum()
    for s in SIDES:
        node[f"win_{s}"] = win[s]
    node = {k: np.asarray(v, np.float32) for k, v in node.items()}
    ei, ea = [], []
    for e in meta["edges"]:
        i, j = int(e["i"]), int(e["j"])
        dx, dy = node["x"][j] - node["x"][i], node["y"][j] - node["y"][i]
        ei += [[i, j], [j, i]]
        ea += [[dx, dy], [-dx, -dy]]
    wwr = meta["wwr"]
    glob = {"W_bbox": W, "H_bbox": H, "AR_bbox": W / H, "A_boundary": A, "P_boundary": P,
            "compactness": 4 * math.pi * A / P ** 2, "n_rooms": n,
            "WWR_N": wwr["N"], "WWR_S": wwr["S"], "WWR_E": wwr["E"], "WWR_W": wwr["W"]}
    lab = meta["labels"]
    return {"id": int(meta["id"]), "node": node, "glob": glob,
            "edge_index": np.array(ei, np.int64).T.reshape(2, -1), "edge_attr": np.array(ea, np.float32).reshape(-1, 2),
            "y_graph": [float(lab["sDA"]), float(lab["DA_mean"]) / 100.0],
            "outline": [list(xy) for xy in outline.exterior.coords[:-1]], "bounds": (minx, miny, maxx, maxy),
            "centres": centres}


def load_all(sg_dir):
    sg_dir = Path(sg_dir)
    params = pd.read_csv(sg_dir / "sample_parameters" / "parameters_all.csv")
    graphs = {}
    for i in params.ID.astype(int):
        graphs[i] = build(json.load(open(sg_dir / "meta" / f"{i}.json", encoding="utf-8")))
    split = dict(zip(params.ID.astype(int), params.split))
    return graphs, split, params
