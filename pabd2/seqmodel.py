"""FlowSeq: a small transformer over a host's recent flows.

Pre-training (labels never used): masked-flow reconstruction -- hide 25% of the flows in a
host's sequence and predict their features from context -- plus a SimCLR-style contrastive loss
that pulls two noisy views of the same sequence together (after Self-Supervised Transformer-based
Contrastive Learning for IDS, 2025, and netFound's multi-objective pre-training).

Anomaly score: hide each of the last K flows in turn and measure how badly the model, trained
only on benign traffic, reconstructs them. Behaviour it has never seen reconstructs poorly.
A light classification head is then fine-tuned on top for the attack family.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import N_NUM, SEQ_LEN
from .schema import PORT_CATS

D = 96


class FlowSeq(nn.Module):
    def __init__(self, n_classes=0, d=D, layers=3, heads=4):
        super().__init__()
        self.inp = nn.Linear(N_NUM, d)
        self.port = nn.Embedding(len(PORT_CATS) + 1, d)
        self.proto = nn.Embedding(4, d)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.cls = nn.Parameter(torch.zeros(d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.recon = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_NUM))
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 64))
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_classes)) if n_classes else None

    def embed(self, num, cat, pad, mask=None):
        x = self.inp(num) + self.port(cat[..., 0]) + self.proto(cat[..., 1]) + self.pos
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_tok.expand_as(x), x)
        x = torch.cat([self.cls.expand(x.size(0), 1, -1), x], 1)
        kpm = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        h = self.norm(self.enc(x, src_key_padding_mask=kpm))
        return h[:, 0], h[:, 1:]

    def forward(self, num, cat, pad, mask=None):
        c, h = self.embed(num, cat, pad, mask)
        return c, self.recon(h)


def batches(idx, num, cat, bs, shuffle, rng):
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for s in range(0, len(order), bs):
        b = idx[order[s:s + bs]]
        pad = b < 0
        bb = np.where(pad, 0, b)
        n = num[bb]; c = cat[bb]
        n[pad] = 0; c[pad] = 0
        yield order[s:s + bs], torch.from_numpy(n), torch.from_numpy(c), torch.from_numpy(pad)


def nt_xent(a, b, t=0.2):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    z = torch.cat([a, b]); sim = z @ z.T / t
    sim.fill_diagonal_(-1e9)
    n = a.size(0); tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, tgt)


def pretrain(model, seq_idx, num, cat, epochs=4, bs=256, lr=2e-3, log=print, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(seq_idx) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    model.train(); k = 0
    for ep in range(epochs):
        tot = 0.0; k_ep = 0
        for _, n, c, pad in batches(seq_idx, num, cat, bs, True, rng):
            m = (torch.rand(pad.shape) < 0.25) & ~pad
            m[:, -1] |= ~pad[:, -1]                      # always practise predicting the newest flow
            c1, rec = model(n, c, pad, m)
            l_rec = ((rec - n) ** 2).mean(-1)[m].mean()
            # second view: jittered, unmasked; the masked pass above is the first view
            v2 = n * (1 + 0.1 * torch.randn_like(n)) + 0.05 * torch.randn_like(n)
            z2, _ = model.embed(v2, c, pad, None)
            loss = l_rec + 0.2 * nt_xent(model.proj(c1), model.proj(z2))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item(); k += 1; k_ep += 1
            if k % 200 == 0: log(f"    step {k}/{steps}  loss {loss.item():.4f}")
        log(f"  epoch {ep + 1}/{epochs}  mean loss {tot / max(1, k_ep):.4f}")
    return model


@torch.no_grad()
def anomaly_scores(model, seq_idx, num, cat, last_k=4, bs=1024):
    """Mean reconstruction error of the last `last_k` flows, each hidden in turn."""
    model.eval(); out = np.zeros(len(seq_idx), np.float32); emb = np.zeros((len(seq_idx), D), np.float32)
    for o, n, c, pad in batches(seq_idx, num, cat, bs, False, None):
        errs = []
        for j in range(1, last_k + 1):
            m = torch.zeros_like(pad); m[:, -j] = True; m &= ~pad
            cl, rec = model(n, c, pad, m)
            e = ((rec[:, -j] - n[:, -j]) ** 2).mean(-1)
            errs.append(torch.where(pad[:, -j], torch.full_like(e, float("nan")), e))
            if j == 1: emb[o] = cl.numpy()
        out[o] = np.nanmean(torch.stack(errs, 1).numpy(), 1)
    return out, emb


def finetune_head(model, n_classes, seq_idx, y, num, cat, epochs=3, bs=256, lr=1e-3, log=print, seed=0):
    """Class-balanced fine-tuning of the family head; the encoder keeps learning at 1/10 the rate."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model.head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, n_classes))
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0), dtype=torch.float32)
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": lr},
                             {"params": [p for n_, p in model.named_parameters() if not n_.startswith("head")], "lr": lr / 10}])
    yt = torch.from_numpy(y)
    for ep in range(epochs):
        model.train(); tot = 0; k = 0
        for o, n, c, pad in batches(seq_idx, num, cat, bs, True, rng):
            cl, _ = model.embed(n, c, pad)
            loss = F.cross_entropy(model.head(cl), yt[o], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); k += 1
        log(f"  head epoch {ep + 1}/{epochs}  loss {tot / k:.4f}")
    return model


@torch.no_grad()
def predict_proba(model, seq_idx, num, cat, bs=1024):
    model.eval(); out = []
    for _, n, c, pad in batches(seq_idx, num, cat, bs, False, None):
        cl, _ = model.embed(n, c, pad)
        out.append(F.softmax(model.head(cl), -1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_proba(model, seq_idx, num, cat, bs=1024):
    """One pass: representation and family probabilities (what fusion uses)."""
    model.eval(); emb = np.zeros((len(seq_idx), D), np.float32); out = []
    for o, n, c, pad in batches(seq_idx, num, cat, bs, False, None):
        cl, _ = model.embed(n, c, pad); emb[o] = cl.numpy()
        out.append(F.softmax(model.head(cl), -1).numpy())
    return emb, np.concatenate(out)
