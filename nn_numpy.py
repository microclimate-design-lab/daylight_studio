"""
NumPy inference for the PyTorch models, so the web app runs without torch / torch_geometric (~300 MB less memory).

Weights are exported once from the PyTorch checkpoints by tools/export_numpy.py:
    models/np/{room,unit,map}_mlp.npz            MLP (Linear / ReLU / Sigmoid) + StandardScaler mean / scale
    models/gnn/<region>/model_s*.npz             LadderGATv2 state dicts   (gnn.Ensemble)
    models/gnn/<region>/map_s*.npz               MapGATv2 state dicts      (gnn.MapEnsemble)
The forward passes below reproduce gnn.py (PyTorch Geometric GATv2Conv: add_self_loops, fill_value="mean",
negative_slope 0.2, concat heads, share_weights False) in eval mode (dropout off).
tools/export_numpy.py --check compares both implementations.
"""
import json
from pathlib import Path

import numpy as np

TARGETS = ["sda", "mean_illum"]
EDGE_ALL = ["dx", "dy"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def linear(x, w, b=None):
    y = x @ w.T
    return y + b if b is not None else y


# ------------------------------------------------------------------------------------------- MLP
class MLP:
    """nn.Sequential of Linear / ReLU / Dropout / Sigmoid with its StandardScaler (models/np/<level>_mlp.npz)."""

    def __init__(self, path):
        z = np.load(path)
        self.acts = [str(a) for a in z["acts"]]
        self.W = [z[f"W{i}"] for i in range(len(self.acts))]
        self.b = [z[f"b{i}"] for i in range(len(self.acts))]
        self.mean, self.scale = z["mean"], z["scale"]

    def __call__(self, X, chunk=65536):
        out = []
        for k in range(0, len(X), chunk):
            h = ((np.asarray(X[k:k + chunk], np.float64) - self.mean) / self.scale).astype(np.float32)
            for W, b, a in zip(self.W, self.b, self.acts):
                h = linear(h, W, b)
                h = relu(h) if a == "relu" else sigmoid(h) if a == "sigmoid" else h
            out.append(h)
        return np.concatenate(out)


# ------------------------------------------------------------------------------------------- GNN
def _encode(columns, chosen, spec, rows):
    """One-hot for categoricals, (v - mean) / std otherwise (same as gnn._encode)."""
    out = []
    for name in chosen:
        col = np.asarray(columns[name], np.float32).reshape(rows)
        levels = spec["categorical"].get(name)
        if levels is not None:
            out.append((col[:, None].astype(int) == np.asarray(levels)[None, :]).astype(np.float32))
        else:
            mu, sd = spec["scaler"][name]
            out.append(((col - mu) / sd)[:, None].astype(np.float32))
    return np.concatenate(out, axis=1) if out else np.zeros((rows, 0), np.float32)


def to_arrays(g, spec):
    n = len(g["node"][spec["node"][0]])
    x = _encode(g["node"], spec["node"], spec, n)
    u = _encode({k: [v] for k, v in g["glob"].items()}, spec["glob"], spec, 1)
    ei = np.asarray(g["edge_index"], np.int64).reshape(2, -1)
    ea = np.asarray(g["edge_attr"], np.float32).reshape(-1, 2)
    if spec["edge"]:
        ea = _encode({k: ea[:, EDGE_ALL.index(k)] for k in spec["edge"]}, spec["edge"], spec, ea.shape[0])
    else:
        ea = np.zeros((ei.shape[1], 0), np.float32)
    return x, ei, ea, u


def batch_graphs(graphs, spec):
    xs, eis, eas, us, batch, off = [], [], [], [], [], 0
    for k, g in enumerate(graphs):
        x, ei, ea, u = to_arrays(g, spec)
        xs.append(x); eis.append(ei + off); eas.append(ea); us.append(u)
        batch.append(np.full(len(x), k)); off += len(x)
    return (np.concatenate(xs), np.concatenate(eis, axis=1), np.concatenate(eas), np.concatenate(us),
            np.concatenate(batch))


def gatv2(p, pre, h, ei, ea, heads):
    """PyG GATv2Conv (eval): self loops with mean incoming edge attribute, attention over incoming edges."""
    n = len(h)
    src, dst = ei
    keep = src != dst                                         # remove_self_loops
    src, dst, ea = src[keep], dst[keep], ea[keep]
    if ea.shape[1]:                                           # fill_value="mean": mean attr of edges into the node
        cnt = np.bincount(dst, minlength=n).astype(np.float32)
        loop = np.zeros((n, ea.shape[1]), np.float32)
        np.add.at(loop, dst, ea)
        loop /= np.maximum(cnt, 1)[:, None]
        ea = np.concatenate([ea, loop])
    loops = np.arange(n)
    src, dst = np.concatenate([src, loops]), np.concatenate([dst, loops])
    C = p[pre + "lin_l.weight"].shape[0] // heads
    xl = linear(h, p[pre + "lin_l.weight"], p[pre + "lin_l.bias"]).reshape(n, heads, C)
    xr = linear(h, p[pre + "lin_r.weight"], p[pre + "lin_r.bias"]).reshape(n, heads, C)
    e = xl[src] + xr[dst]
    if ea.shape[1] and pre + "lin_edge.weight" in p:
        e = e + linear(ea, p[pre + "lin_edge.weight"]).reshape(-1, heads, C)
    e = np.where(e > 0, e, 0.2 * e)
    a = (e * p[pre + "att"]).sum(-1)                          # (E, heads)
    amax = np.full((n, heads), -np.inf, np.float32)
    np.maximum.at(amax, dst, a)
    a = np.exp(a - amax[dst])
    den = np.zeros((n, heads), np.float32)
    np.add.at(den, dst, a)
    a = a / (den[dst] + 1e-16)
    out = np.zeros((n, heads, C), np.float32)
    np.add.at(out, dst, xl[src] * a[..., None])
    return out.reshape(n, heads * C) + p[pre + "bias"]


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _head(p, pre, x):
    return linear(relu(linear(x, p[pre + "0.weight"], p[pre + "0.bias"])), p[pre + "3.weight"], p[pre + "3.bias"])


def embed(p, x, ei, ea, u, batch, heads):
    u_node = u[batch]
    h = relu(linear(np.concatenate([x, u_node], 1), p["inp.weight"], p["inp.bias"]))
    k = 0
    while f"convs.{k}.att" in p:
        h = layer_norm(h + elu(gatv2(p, f"convs.{k}.", h, ei, ea, heads)), p[f"norms.{k}.weight"], p[f"norms.{k}.bias"])
        k += 1
    return h, u_node


def ladder_forward(p, x, ei, ea, u, batch, heads):
    h, u_node = embed(p, x, ei, ea, u, batch, heads)
    y_node = sigmoid(_head(p, "node_head.", np.concatenate([h, u_node], 1)))
    nb = int(batch.max()) + 1
    mean = np.stack([h[batch == b].mean(0) for b in range(nb)])
    mx = np.stack([h[batch == b].max(0) for b in range(nb)])
    y_graph = sigmoid(_head(p, "graph_head.", np.concatenate([mean, mx, u], 1)))
    return h, u_node, y_node, y_graph


def _load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


class Ensemble:
    """The five seeds of one region; predict() returns (room [n x 2], layout [2]) as seed means and seed std."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.load(open(folder / "spec.json"))
        self.models = [_load(p) for p in sorted(folder.glob("model_s*.npz"))]
        if not self.models:
            raise FileNotFoundError(f"no model_s*.npz in {folder} (run tools/export_numpy.py)")
        self.heads = self.models[0]["convs.0.att"].shape[1]

    def predict(self, graphs):
        x, ei, ea, u, batch = batch_graphs(graphs, self.spec)
        runs = [ladder_forward(p, x, ei, ea, u, batch, self.heads)[2:] for p in self.models]
        node = np.stack([r[0] for r in runs])
        graph = np.stack([r[1] for r in runs])
        return node.mean(0), graph.mean(0), node.std(0), graph.std(0), batch


class MapEnsemble:
    """Seeds of the pixel-level GNN (map_s*.npz + map_spec.json)."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.load(open(folder / "map_spec.json"))
        self.models = [_load(p) for p in sorted(folder.glob("map_s*.npz"))]
        if not self.models:
            raise FileNotFoundError(f"no map_s*.npz in {folder} (run tools/export_numpy.py)")
        self.heads = self.spec["arch"]["heads"]

    def predict_map(self, graph, X, cell_room, chunk=20000):
        x, ei, ea, u, batch = batch_graphs([graph], self.spec)
        mu = np.asarray(self.spec["cell_scaler"][0], np.float32)
        sd = np.asarray(self.spec["cell_scaler"][1], np.float32)
        xc = ((np.asarray(X, np.float32) - mu) / sd).astype(np.float32)
        idx = np.asarray(cell_room, np.int64)
        runs = []
        for p in self.models:
            h, u_node = embed(p, x, ei, ea, u, batch, self.heads)
            n_lin = sum(1 for k in p if k.startswith("cell_head.") and k.endswith(".weight"))
            parts = []
            for i in range(0, len(idx), chunk):
                c = np.concatenate([h[idx[i:i + chunk]], u_node[idx[i:i + chunk]], xc[i:i + chunk]], 1)
                for j in range(n_lin):
                    c = linear(c, p[f"cell_head.{2 * j}.weight"], p[f"cell_head.{2 * j}.bias"])
                    if j < n_lin - 1:
                        c = relu(c)
                parts.append(sigmoid(c)[:, 0])
            runs.append(np.concatenate(parts))
        r = np.stack(runs)
        return r.mean(0), r.std(0)
