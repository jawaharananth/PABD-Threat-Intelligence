"""Host-space evasion test (after "What is the Problem Space?", 2026, and Apostolaki-style
realism arguments): the attacker controls only their own machine.

Allowed, per attack flow in the attacker's sequence:
  slow   - stretch inter-arrival time by factor s           (rate limiting)
  pad    - add bytes to forward packets by factor s         (payload padding)
  chaff  - replace a fraction s of the sequence with benign flows sampled from benign hosts
           (decoy traffic generated from the same machine)
Not allowed: anything in the backward direction (the victim's responses), TCP flag semantics,
or removing the attack flows themselves.

We re-score the perturbed sequences with the transformer (representation + family head) and recompute
the fused score with the graph score held at its observed value -- conservative for slow/pad
(the graph sees the same edges) and optimistic for the attacker under chaff (decoys would add
new edges the graph layer could notice).

    python -m pabd2.adversarial            # robustness of the standard model
"""
import json, os, pickle
import numpy as np
import pandas as pd
import torch
from .features import N_NUM
from .prepare import CACHE
from .schema import FEATURES
from .seqmodel import FlowSeq, embed_proba

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v2")
NF = len(FEATURES)
FWD = [i for i, f in enumerate(FEATURES) if f.startswith(("Fwd Packet Length", "Total Length of Fwd"))]
RATE = [i for i, f in enumerate(FEATURES) if f in ("Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s")]
IAT = [i for i, f in enumerate(FEATURES) if "IAT" in f or f in ("Flow Duration", "Idle Mean", "Idle Max", "Total TCP Flow Time")]
DT = NF  # index of log-dt token column


def raw(sc, Z):
    L = Z * sc.iqr + sc.med
    return np.sign(L) * np.expm1(np.abs(L))


def perturb(num, seq_rows, kind, s, sc, benign_pool, rng):
    """Return a modified copy of the token rows referenced by seq_rows (flow indices)."""
    n = num[seq_rows].copy()
    if kind == "slow":
        R = raw(sc, n[:, :NF]); R[:, IAT] *= s; R[:, RATE] /= s
        n[:, :NF] = sc.transform(R)
        n[:, DT] = np.clip(n[:, DT] + np.log1p(s) / np.log1p(3600), 0, 1)
    elif kind == "pad":
        R = raw(sc, n[:, :NF]); R[:, FWD] *= s
        n[:, :NF] = sc.transform(R)
    elif kind == "chaff":
        k = int(round(s * len(n)))
        if k: n[rng.choice(len(n), k, replace=False)] = num[rng.choice(benign_pool, k)]
    return n


@torch.no_grad()
def run(budgets=None, tag=""):
    budgets = budgets or {"slow": [2, 5, 20], "pad": [1.5, 3, 10], "chaff": [0.25, 0.5, 0.75]}
    rng = np.random.default_rng(0)
    sc = pickle.load(open(os.path.join(CACHE, "scaler.pkl"), "rb"))
    z = np.load(os.path.join(CACHE, "tok_2017.npz")); num, cat = z["num"], z["cat"]
    seq = np.load(os.path.join(CACHE, "seq_2017.npy")).astype(np.int64)
    ck = torch.load(os.path.join(CACHE, f"flowseq{tag}.pt")); fams = ck["families"]
    model = FlowSeq(len(fams)); model.load_state_dict(ck["state"]); model.eval()
    u = pd.read_parquet(os.path.join(OUT, "units_scored_2017.parquet"))
    st = json.load(open(os.path.join(OUT, "fusion_state.json")))
    attack = np.flatnonzero((u.role == "test").to_numpy() & ~u.family.isin(["Benign", "Attempted"]).to_numpy())
    benign_pool = np.flatnonzero(pd.read_parquet(os.path.join(CACHE, "flows_2017.parquet"), columns=["family"]).family.to_numpy() == "Benign")
    cal = np.array(st["cal_benign_fused"]); thr = np.quantile(cal, 0.99)
    st["mu"], st["sd"] = np.array(st["mu"]), np.array(st["sd"])

    from .evaluate import Normality
    emb17 = np.load(os.path.join(CACHE, f"seqscores_2017{tag}.npz"))["emb"]
    norm = Normality().fit(emb17[((u.split == "train") & (u.family == "Benign")).to_numpy()])

    def fused(emb, att, graph, log_n, log_dst):
        x = (np.stack([norm.score(emb), att, graph, log_n, log_dst], 1) - st["mu"]) / st["sd"]
        return x @ np.array(st["w"]) + st["b"]

    base = u.iloc[attack]
    res = {"units": int(len(attack)), "threshold": "conformal alpha=0.01",
           "clean_detection": round(float((base.fused > thr).mean()), 4)}
    res["by_family_clean"] = base.assign(d=base.fused > thr).groupby("family").d.mean().round(4).to_dict()
    for kind, ss in budgets.items():
        res[kind] = {}
        for s in ss:
            S = seq[attack]; pad = S < 0
            n = np.zeros(S.shape + (N_NUM,), np.float32)
            for r in range(len(S)):
                rows = S[r][~pad[r]]
                n[r, ~pad[r]] = perturb(num, rows, kind, s, sc, benign_pool, rng)
            # score perturbed sequences: write them into a scratch token table
            flat = n.reshape(-1, N_NUM); idx = np.arange(len(flat)).reshape(S.shape)
            idx[pad] = -1
            c = np.zeros((len(flat), 2), np.int64)
            c[idx[~pad]] = cat[S[~pad]]
            emb, pr = embed_proba(model, idx, flat, c)
            att = 1 - pr[:, fams.index("Benign")]
            f = fused(emb, att, base.graph.to_numpy(), base.log_n.to_numpy(), base.log_dst.to_numpy())
            res[kind][str(s)] = round(float((f > thr).mean()), 4)
            print(kind, s, res[kind][str(s)], flush=True)
    json.dump(res, open(os.path.join(OUT, f"adversarial{tag}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    print(run())
