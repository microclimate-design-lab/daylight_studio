"""Feature builders shared by training (train_models.py) and inference (predict.py).
All features are computed from a PARSED bubble-diagram graph (bubble_to_graph.py) plus ORI / WWR."""
import collections
import math

import numpy as np

PX_PER_M = 199.735
CELL_M = 0.1
CELL_PX = CELL_M * PX_PER_M
ROOM_NAMES = ["Living", "Master0", "Master1", "Master2", "Dining", "Kitchen", "Bath", "Study", "Storage"]
GEO = ["area", "area_ratio", "x", "y", "d_N", "d_S", "d_E", "d_W"]            # node feature cols 1..8
GLOBAL = ["W_bbox", "H_bbox", "AR_bbox", "A_boundary", "P_boundary", "compactness",
          "ORI_0", "ORI_1", "ORI_2", "WWR_S", "WWR_N"]                       # VLT is not used
SLOTS = [(t, 0) for t in range(9)] + [(6, 1)]


def global_vec(gf):
    W, H, AR, A, P, C, ORI, WS, WN = gf[:9]
    return np.r_[W, H, AR, A, P, C, np.eye(3)[int(ORI)], WS, WN].astype(float)


def gap_features(nf, gf):
    """Distance from the bubble's equivalent-circle edge to the N / S facade (m), relative depth."""
    r_eq = np.sqrt(nf[:, 1] / math.pi)
    H = gf[1]
    return np.c_[nf[:, 5] - r_eq, nf[:, 6] - r_eq, nf[:, 5] / H, nf[:, 6] / H]


GAP = ["gap_N", "gap_S", "rel_d_N", "rel_d_S"]


# ---------------------------------------------------------------- windows (per room)
def window_features(g):
    nf = np.asarray(g["node_features"], float)
    gv = global_vec(g["graph_features"])
    return np.c_[np.eye(9)[nf[:, 0].astype(int)], nf[:, 1:9], gap_features(nf, g["graph_features"]),
                 np.tile(gv, (len(nf), 1)), np.full(len(nf), len(nf))]


WINDOW_NAMES = [f"type_{n}" for n in ROOM_NAMES] + GEO + GAP + GLOBAL + ["num_rooms"]


# ---------------------------------------------------------------- room level
def room_features(g, pwin):
    """pwin: (N, 2) window probabilities (or 0/1 if given by the user)."""
    return np.c_[window_features(g), pwin]


ROOM_NAMES_FEAT = WINDOW_NAMES + ["p_win_N", "p_win_S"]


# ---------------------------------------------------------------- unit level (fixed-length, no edges)
def unit_features(g, pwin):
    nf = np.asarray(g["node_features"], float)
    per = np.c_[nf[:, 1:9], pwin]                                       # 10 values per room
    slots = np.zeros((len(SLOTS), 1 + per.shape[1]))
    seen = collections.Counter()
    for k in sorted(range(len(nf)), key=lambda k: (nf[k, 0], -nf[k, 1])):
        t = int(nf[k, 0])
        if (t, seen[t]) not in SLOTS:
            continue
        idx = SLOTS.index((t, seen[t]))
        seen[t] += 1
        slots[idx, 0], slots[idx, 1:] = 1.0, per[k]
    gv = global_vec(g["graph_features"])
    win_area = float((nf[:, 2] * pwin[:, 0]).sum()), float((nf[:, 2] * pwin[:, 1]).sum())
    return np.r_[gv, len(nf), win_area, slots.ravel()]


def unit_names():
    n = GLOBAL + ["num_rooms", "winN_area_share", "winS_area_share"]
    for t, j in SLOTS:
        tag = f"{ROOM_NAMES[t]}{'' if j == 0 else j + 1}"
        n += [f"{tag}_present"] + [f"{tag}_{x}" for x in GEO + ["p_win_N", "p_win_S"]]
    return n


# ---------------------------------------------------------------- DA map (per 0.1 m cell)
def _runs(inside):
    """Number of consecutive inside cells (incl. itself) towards N, S, E, W."""
    R, C = inside.shape
    n, s, e, w = (np.zeros((R, C), np.float32) for _ in range(4))
    for r in range(R):
        n[r] = (n[r - 1] + 1) * inside[r] if r else inside[r]
    for r in range(R - 1, -1, -1):
        s[r] = (s[r + 1] + 1) * inside[r] if r < R - 1 else inside[r]
    for c in range(C):
        w[:, c] = (w[:, c - 1] + 1) * inside[:, c] if c else inside[:, c]
    for c in range(C - 1, -1, -1):
        e[:, c] = (e[:, c + 1] + 1) * inside[:, c] if c < C - 1 else inside[:, c]
    return n, s, e, w


def grid_shape(g):
    W, H = g["graph_features"][0], g["graph_features"][1]
    return int(round(H / CELL_M)), int(round(W / CELL_M))


def cell_features(g, dbg_bbox, outline_mask, pwin, shape=None):
    """Features for every cell of the 0.1 m grid over the outline bbox (row 0 = north, col 0 = west).
    Returns X (R*C, F), inside (R, C) bool, room (R, C) int (assigned bubble, -1 outside)."""
    bx, by, W, H = dbg_bbox
    R, C = shape or grid_shape(g)
    rr, cc = np.mgrid[0:R, 0:C]
    py = np.clip(by + (rr + 0.5) * CELL_PX, 0, outline_mask.shape[0] - 1).astype(int)
    px = np.clip(bx + (cc + 0.5) * CELL_PX, 0, outline_mask.shape[1] - 1).astype(int)
    inside = outline_mask[py, px].astype(bool)
    n, s, e, w = (x * CELL_M for x in _runs(inside.astype(np.float32)))

    nf = np.asarray(g["node_features"], float)
    cx = bx + nf[:, 3] * W                                   # bubble centres in px
    cy = by + (1 - nf[:, 4]) * H
    r_eq = np.sqrt(nf[:, 1] / math.pi) * PX_PER_M
    # weighted nearest bubble: distance relative to the bubble's equivalent radius
    dist = np.hypot(px[..., None] - cx, py[..., None] - cy)  # (R, C, N)
    rel = dist / r_eq
    room = rel.argmin(-1)
    room_d = np.take_along_axis(rel, room[..., None], -1)[..., 0]
    ordered = np.sort(rel, -1)
    second = ordered[..., 1] if nf.shape[0] > 1 else np.full((R, C), 9.0)

    k = room.ravel()
    gv = global_vec(g["graph_features"])
    X = np.c_[
        (cc.ravel() + 0.5) / C, (rr.ravel() + 0.5) / R,                     # relative position (x: W->E, y: N->S)
        rr.ravel() * CELL_M, (R - rr.ravel()) * CELL_M,                     # metres from N / S edge of bbox
        n.ravel(), s.ravel(), e.ravel(), w.ravel(),                         # distance to outline N, S, E, W
        room_d.ravel(), second.ravel(), second.ravel() - room_d.ravel(),    # bubble membership
        (px.ravel() - cx[k]) / PX_PER_M, (cy[k] - py.ravel()) / PX_PER_M,   # offset from bubble centre (m, E / N)
        np.eye(9)[nf[k, 0].astype(int)], nf[k, 1:9], pwin[k],               # assigned room
        np.tile(gv, (R * C, 1)),
    ].astype(np.float32)
    room = np.where(inside, room, -1)
    return X, inside, room


CELL_NAMES = (["u", "v", "m_from_N", "m_from_S", "c_d_N", "c_d_S", "c_d_E", "c_d_W",
               "rel_dist", "rel_dist_2nd", "membership_margin", "off_E", "off_N"]
              + [f"room_type_{n}" for n in ROOM_NAMES] + [f"room_{x}" for x in GEO]
              + ["room_p_win_N", "room_p_win_S"] + GLOBAL)
