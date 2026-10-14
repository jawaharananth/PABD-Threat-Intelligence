"""Graph layer: E-GraphSAGE over host-communication graphs (Lo et al., 2022), made inductive.

Every 10-second bin is a graph: nodes are hosts, edges are (src -> dst) aggregates of that bin's
flows. Messages carry edge features, so a node learns from *how* its neighbours talk to it, not
only who they are. No identity reaches the model: node features are behavioural (fan-out,
fan-in, internal reach, first-time contacts over the last five minutes), which is what lets the
same weights run on a different network (the 2018 test) and what the graph-NIDS audits ask for.

Lateral-movement signal: an internal host suddenly contacting many internal hosts it has never
talked to (the 2017 infiltration victim scanning 11 machines) shows up as high internal fan-out
plus a burst of first-time internal contacts.
"""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import is_internal, secs
from .units import BIN

NODE_F = 8


def build(flows: pd.DataFrame, num: np.ndarray, units: pd.DataFrame):
    """Returns node features, edge index/features/labels and unit->node mapping."""
    t = secs(flows["Timestamp"]).astype(np.int64); b = t // BIN
    f = pd.DataFrame({"day": flows["day"].to_numpy(), "bin": b, "src": flows["Src IP"].to_numpy(),
                      "dst": flows["Dst IP"].to_numpy(), "i": np.arange(len(flows))})
    fam = flows["family"].to_numpy(); att = flows["attempted"].to_numpy()
    f["lab"] = np.where((fam != "Benign") & ~att, fam, np.where(att, "Attempted", "Benign"))
    # edges
    g = f.groupby(["day", "bin", "src", "dst"], sort=False)
    e = g.agg(n=("i", "size"), first=("i", "min")).reset_index()
    sums = np.zeros((len(e), num.shape[1]), np.float32)
    eid = g.ngroup().to_numpy()
    np.add.at(sums, eid, num[f["i"].to_numpy()])
    ex = sums / e["n"].to_numpy()[:, None]
    lab = f.groupby(eid)["lab"].agg(lambda s: s.value_counts().index[0]).to_numpy()
    e_int = is_internal(e["dst"])
    ex = np.concatenate([ex, np.log1p(e["n"].to_numpy())[:, None].astype(np.float32), e_int[:, None].astype(np.float32)], 1)
    # nodes: every (day, bin, host) appearing as src or dst
    nodes = pd.concat([e[["day", "bin", "src"]].rename(columns={"src": "host"}),
                       e[["day", "bin", "dst"]].rename(columns={"dst": "host"})]).drop_duplicates().reset_index(drop=True)
    nid = pd.Series(np.arange(len(nodes)), index=pd.MultiIndex.from_frame(nodes))
    s_ix = nid.loc[pd.MultiIndex.from_frame(e[["day", "bin", "src"]].rename(columns={"src": "host"}))].to_numpy()
    d_ix = nid.loc[pd.MultiIndex.from_frame(e[["day", "bin", "dst"]].rename(columns={"dst": "host"}))].to_numpy()
    # behavioural node features
    out_deg = np.bincount(s_ix, minlength=len(nodes)); in_deg = np.bincount(d_ix, minlength=len(nodes))
    int_out = np.bincount(s_ix, weights=e_int, minlength=len(nodes))
    out_flows = np.bincount(s_ix, weights=e["n"].to_numpy(), minlength=len(nodes))
    in_flows = np.bincount(d_ix, weights=e["n"].to_numpy(), minlength=len(nodes))
    # first-time contacts in the last 5 minutes (30 bins), per source host
    e_sorted = e.assign(s=s_ix).sort_values(["day", "src", "bin"])
    first_new = np.zeros(len(e), np.float32); last = {}
    for k, (d, src, dst, bb) in zip(e_sorted.index, e_sorted[["day", "src", "dst", "bin"]].itertuples(index=False)):
        p = last.get((d, src, dst)); first_new[k] = 1.0 if p is None or bb - p > 30 else 0.0; last[(d, src, dst)] = bb
    new_out = np.bincount(s_ix, weights=first_new, minlength=len(nodes))
    new_int = np.bincount(s_ix, weights=first_new * e_int, minlength=len(nodes))
    h_int = is_internal(nodes["host"]).astype(np.float32)
    X = np.log1p(np.stack([out_deg, in_deg, int_out, out_flows, in_flows, new_out, new_int], 1)).astype(np.float32)
    X = np.concatenate([X, h_int[:, None]], 1)
    ex = np.concatenate([ex, first_new[:, None]], 1)
    # unit -> node
    u_ix = nid.reindex(pd.MultiIndex.from_frame(units[["day", "bin", "host"]])).to_numpy()
    return dict(X=X, s=s_ix, d=d_ix, ex=ex.astype(np.float32), lab=lab, unit_node=u_ix,
                edge_first=e["first"].to_numpy(), nodes=nodes)


class EGraphSAGE(nn.Module):
    def __init__(self, nf, ef, n_classes, d=64):
        super().__init__()
        self.l1 = nn.Linear(nf + ef + nf, d); self.l2 = nn.Linear(d + ef + d, d)
        self.out = nn.Sequential(nn.Linear(2 * d + ef, d), nn.ReLU(), nn.Linear(d, n_classes))

    @staticmethod
    def agg(h, ex, s, d, n):
        msg = torch.cat([ex, h[s]], 1)
        m = torch.zeros(n, msg.size(1)).index_add_(0, d, msg)
        cnt = torch.zeros(n).index_add_(0, d, torch.ones(len(d))).clamp(min=1)[:, None]
        # undirected view: sources also hear from their destinations
        m2 = torch.zeros(n, msg.size(1)).index_add_(0, s, torch.cat([ex, h[d]], 1))
        cnt2 = torch.zeros(n).index_add_(0, s, torch.ones(len(s))).clamp(min=1)[:, None]
        return (m + m2) / (cnt + cnt2)

    def forward(self, X, ex, s, d):
        n = X.size(0)
        h = F.relu(self.l1(torch.cat([X, self.agg(X, ex, s, d, n)], 1)))
        h = F.relu(self.l2(torch.cat([h, self.agg(h, ex, s, d, n)], 1)))
        return self.out(torch.cat([h[s], h[d], ex], 1)), h


def train(G, edge_train, classes, epochs=60, lr=3e-3, seed=0, log=print):
    torch.manual_seed(seed)
    X, ex = torch.from_numpy(G["X"]), torch.from_numpy(G["ex"])
    s, d = torch.from_numpy(G["s"]), torch.from_numpy(G["d"])
    y = np.array([classes.index(l) if l in classes else -1 for l in G["lab"]])
    m = torch.from_numpy(edge_train & (y >= 0)); yt = torch.from_numpy(y)
    cnt = np.bincount(y[edge_train & (y >= 0)], minlength=len(classes)).astype(float)
    w = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0), dtype=torch.float32)
    model = EGraphSAGE(X.size(1), ex.size(1), len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train(); logit, _ = model(X, ex, s, d)
        loss = F.cross_entropy(logit[m], yt[m], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0: log(f"  graph epoch {ep + 1}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def unit_scores(model, G, n_units):
    """Per unit: max non-benign edge probability over the host's incident edges, and the class."""
    model.eval()
    logit, h = model(torch.from_numpy(G["X"]), torch.from_numpy(G["ex"]), torch.from_numpy(G["s"]), torch.from_numpy(G["d"]))
    p = F.softmax(logit, -1).numpy()
    att = 1 - p[:, 0]
    node_att = np.zeros(len(G["X"]), np.float32); node_cls = np.zeros((len(G["X"]), p.shape[1]), np.float32)
    for ix in (G["s"], G["d"]):
        better = att > node_att[ix]
        np.maximum.at(node_att, ix, att)
    # class profile: probability-weighted mean over outgoing edges
    np.add.at(node_cls, G["s"], p)
    node_cls /= np.maximum(node_cls.sum(1, keepdims=True), 1e-9)
    un = G["unit_node"]; ok = ~np.isnan(un.astype(float))
    ui = np.where(ok, un, 0).astype(np.int64)
    score = np.where(ok, node_att[ui], 0.0); cls = np.where(ok[:, None], node_cls[ui], 0.0)
    return score.astype(np.float32), cls.astype(np.float32), h.numpy()
