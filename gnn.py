"""
GATv2 ladder models from daylight_gnn (graph neural network, one prediction per room and per layout).

    Shanghai   results_gat/model_gatv2_R6_environ_s{0-4}.pt     (GATv2_Ladder_Train notebook)
    Singapore  results_gat_sg/model_gatv2_R6_window_s{0-4}.pt   (SG HDB dataset, four-direction windows)

The checkpoints hold weights only. models/gnn/<region>/spec.json stores what the network needs besides the
weights: input features, one-hot levels and the training-set mean / std of every continuous feature
(written by tools/build_gnn.py). Predictions are the mean of the five seeds.

A graph is a dict: node {name: array (n rooms)}, glob {name: float}, edge_index (2 x e), edge_attr (e x 2, dx dy).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

TARGETS = ["sda", "mean_illum"]
EDGE_ALL = ["dx", "dy"]


class LadderGATv2(nn.Module):
    """Identical to LadderGATv2 in the GATv2_Ladder_Train notebook (layer names must match the checkpoints)."""

    def __init__(self, in_node, in_edge, in_glob, hidden=64, heads=4, layers=3, dropout=0.1, n_out=len(TARGETS)):
        super().__init__()
        self.in_edge, self.dropout = in_edge, dropout
        self.inp = nn.Linear(in_node + in_glob, hidden)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=in_edge or None, dropout=dropout)
                                    for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.node_head = nn.Sequential(nn.Linear(hidden + in_glob, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))
        self.graph_head = nn.Sequential(nn.Linear(2 * hidden + in_glob, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def embed(self, data):
        """Room embeddings after the message passing (n rooms x hidden)."""
        u_node = data.u[data.batch]
        h = F.relu(self.inp(torch.cat([data.x, u_node], dim=1)))
        ea = data.edge_attr if self.in_edge else None
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.elu(conv(h, data.edge_index, ea)))
            h = F.dropout(h, self.dropout, self.training)
        return h

    def forward(self, data):
        h, u_node = self.embed(data), data.u[data.batch]
        y_node = torch.sigmoid(self.node_head(torch.cat([h, u_node], dim=1)))
        pooled = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch), data.u], dim=1)
        return y_node, torch.sigmoid(self.graph_head(pooled))


class MapGATv2(LadderGATv2):
    """LadderGATv2 plus a pixel-level decoder: DA of a 0.1 m cell from the embedding of the room the cell
    belongs to (nearest bubble, as features.cell_features), the globals and the cell's own features."""

    def __init__(self, in_node, in_edge, in_glob, in_cell, hidden=64, heads=4, cell_hidden=(256, 128), dropout=0.1):
        super().__init__(in_node, in_edge, in_glob, hidden=hidden, heads=heads, dropout=dropout)
        dims, layers = [hidden + in_glob + in_cell, *cell_hidden], []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.cell_head = nn.Sequential(*layers, nn.Linear(dims[-1], 1))

    def forward(self, data, cell_x=None, cell_node=None):
        """cell_node: index of each cell's room in the batched node list."""
        h, u_node = self.embed(data), data.u[data.batch]
        y_node = torch.sigmoid(self.node_head(torch.cat([h, u_node], dim=1)))
        pooled = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch), data.u], dim=1)
        y_graph = torch.sigmoid(self.graph_head(pooled))
        if cell_x is None:
            return y_node, y_graph, None
        y_cell = torch.sigmoid(self.cell_head(torch.cat([h[cell_node], u_node[cell_node], cell_x], dim=1)))[:, 0]
        return y_node, y_graph, y_cell


def _encode(columns, chosen, spec, rows):
    """columns: {name: values}. One-hot for categoricals, (v - mean) / std otherwise, as _encode() in the notebook."""
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


def fit_scaler(graphs, spec):
    """Training-set mean / std (population std + 1e-8) of every continuous node, global and edge feature."""
    sc = {}
    for name in spec["node"]:
        if name not in spec["categorical"]:
            c = np.concatenate([g["node"][name] for g in graphs]).astype(np.float32)
            sc[name] = [float(c.mean()), float(c.std() + 1e-8)]
    for name in spec["glob"]:
        if name not in spec["categorical"]:
            c = np.array([g["glob"][name] for g in graphs], np.float32)
            sc[name] = [float(c.mean()), float(c.std() + 1e-8)]
    edges = np.concatenate([g["edge_attr"] for g in graphs]).astype(np.float32)
    for name in spec["edge"]:
        c = edges[:, EDGE_ALL.index(name)]
        sc[name] = [float(c.mean()), float(c.std() + 1e-8)]
    return sc


def to_data(g, spec):
    n = len(g["node"][spec["node"][0]])
    x = _encode(g["node"], spec["node"], spec, n)
    u = _encode({k: [v] for k, v in g["glob"].items()}, spec["glob"], spec, 1)
    ei = np.asarray(g["edge_index"], np.int64).reshape(2, -1)
    ea = np.asarray(g["edge_attr"], np.float32).reshape(-1, 2)
    if spec["edge"]:
        ea = _encode({k: ea[:, EDGE_ALL.index(k)] for k in spec["edge"]}, spec["edge"], spec, ea.shape[0])
    else:
        ea = np.zeros((ei.shape[1], 0), np.float32)
    return Data(x=torch.tensor(x), edge_index=torch.tensor(ei), edge_attr=torch.tensor(ea), u=torch.tensor(u))


class Ensemble:
    """The five seeds of one region; predict() returns (room [n x 2], layout [2]) as seed means and seed std."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.load(open(folder / "spec.json"))
        self.models = []
        for path in sorted(folder.glob("model_*.pt")):
            sd = torch.load(path, map_location="cpu", weights_only=True)
            hidden, heads = sd["inp.weight"].shape[0], sd["convs.0.att"].shape[1]
            in_glob = sd["graph_head.0.weight"].shape[1] - 2 * hidden
            in_edge = sd["convs.0.lin_edge.weight"].shape[1] if "convs.0.lin_edge.weight" in sd else 0
            m = LadderGATv2(sd["inp.weight"].shape[1] - in_glob, in_edge, in_glob, hidden=hidden, heads=heads)
            m.load_state_dict(sd)
            self.models.append(m.eval())
        if not self.models:
            raise FileNotFoundError(f"no model_*.pt in {folder}")

    @torch.no_grad()
    def predict(self, graphs):
        batch = Batch.from_data_list([to_data(g, self.spec) for g in graphs])
        runs = [m(batch) for m in self.models]
        node = torch.stack([r[0] for r in runs]).numpy()
        graph = torch.stack([r[1] for r in runs]).numpy()
        return node.mean(0), graph.mean(0), node.std(0), graph.std(0), batch.batch.numpy()


def cell_inputs(X, spec):
    """Raw cell features (features.cell_features, plus E / W window columns for Singapore) -> scaled tensor."""
    mu, sd = np.asarray(spec["cell_scaler"][0], np.float32), np.asarray(spec["cell_scaler"][1], np.float32)
    return torch.tensor(((np.asarray(X, np.float32) - mu) / sd).astype(np.float32))


class MapEnsemble:
    """Seeds of the pixel-level GNN (models/gnn/<region>/map_s*.pt + map_spec.json, tools/train_gnn_map.py)."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.load(open(folder / "map_spec.json"))
        a = self.spec["arch"]
        self.models = []
        for path in sorted(folder.glob("map_s*.pt")):
            m = MapGATv2(a["in_node"], a["in_edge"], a["in_glob"], a["in_cell"], hidden=a["hidden"], heads=a["heads"],
                         cell_hidden=tuple(a["cell_hidden"]))
            m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.models.append(m.eval())
        if not self.models:
            raise FileNotFoundError(f"no map_s*.pt in {folder}")

    @torch.no_grad()
    def predict_map(self, graph, X, cell_room, chunk=20000):
        """One layout: graph (gnn schema), X raw cell features, cell_room assigned room per cell. Returns mean, std."""
        data = Batch.from_data_list([to_data(graph, self.spec)])
        xc, idx = cell_inputs(X, self.spec), torch.tensor(np.asarray(cell_room, np.int64))
        runs = []
        for m in self.models:
            runs.append(torch.cat([m(data, xc[i:i + chunk], idx[i:i + chunk])[2] for i in range(0, len(idx), chunk)]))
        r = torch.stack(runs).numpy()
        return r.mean(0), r.std(0)
