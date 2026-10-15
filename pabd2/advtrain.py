"""Adversarial training against host-space evasion.

Continues training the main FlowSeq model on its original training mix plus disguised copies of
the attack sequences: each copy gets a random host-space perturbation (pad 1.5-10x, slow 2-20x,
or 25-75% chaff) -- exactly the attacker capabilities pabd2.adversarial tests. Saves
flowseq_adv.pt and its 2017 scores; then `python -m pabd2.adversarial --tag _adv` re-runs the
evasion test with a threshold recalibrated for this model.

    python -m pabd2.advtrain
"""
import os, pickle, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from .adversarial import perturb
from .features import N_NUM, SEQ_LEN, sequence_index
from .prepare import CACHE
from .seqmodel import FlowSeq, batches, embed_proba
from .train_seq import flow_split

torch.set_num_threads(max(1, os.cpu_count() - 1))


def main(per_family=15_000, n_adv=12_000, epochs=2, lr=3e-4, seed=1):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); t0 = time.time()
    sc = pickle.load(open(os.path.join(CACHE, "scaler.pkl"), "rb"))
    z = np.load(os.path.join(CACHE, "tok_2017.npz")); num, cat = z["num"], z["cat"]
    flows = pd.read_parquet(os.path.join(CACHE, "flows_2017.parquet"))
    units = pd.read_parquet(os.path.join(CACHE, "units_2017.parquet"))
    train = flow_split(flows, units) == "train"
    fam = flows["family"].to_numpy(); att = flows["attempted"].to_numpy()
    ck = torch.load(os.path.join(CACHE, "flowseq.pt")); fams = ck["families"]
    model = FlowSeq(len(fams)); model.load_state_dict(ck["state"])

    pick = []
    for f in fams:
        ix = np.flatnonzero(train & (fam == f) & ~att)
        if len(ix): pick.append(rng.choice(ix, min(per_family * (5 if f == "Benign" else 1), len(ix)), replace=False))
    pick = np.sort(np.concatenate(pick)); y = np.array([fams.index(f) for f in fam[pick]])
    seq = sequence_index(flows, pick)

    # disguised copies of attack sequences, appended to a private token table
    benign_pool = np.flatnonzero(fam == "Benign")
    a_rows = rng.choice(np.flatnonzero(y != fams.index("Benign")), n_adv, replace=False)
    extra_num = np.zeros((n_adv * SEQ_LEN, N_NUM), np.float32); extra_cat = np.zeros((n_adv * SEQ_LEN, 2), np.int64)
    adv_seq = np.full((n_adv, SEQ_LEN), -1, np.int64)
    for k, r in enumerate(a_rows):
        s_ = seq[r]; ok = s_ >= 0; rows = s_[ok]
        kind = rng.choice(["pad", "slow", "chaff"])
        mag = {"pad": rng.uniform(1.5, 10), "slow": rng.uniform(2, 20), "chaff": rng.uniform(0.25, 0.75)}[kind]
        base = k * SEQ_LEN + np.flatnonzero(ok)
        extra_num[base] = perturb(num, rows, kind, mag, sc, benign_pool, rng)
        extra_cat[base] = cat[rows]
        adv_seq[k, ok] = len(num) + base
    NUM = np.concatenate([num, extra_num]); CAT = np.concatenate([cat, extra_cat])
    all_seq = np.concatenate([seq, adv_seq]); all_y = np.concatenate([y, y[a_rows]])
    print(f"training on {len(seq):,} original + {n_adv:,} disguised sequences ({time.time() - t0:.0f}s)", flush=True)

    cnt = np.bincount(all_y, minlength=len(fams)).astype(float)
    w = torch.tensor(np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); yt = torch.from_numpy(all_y)
    for ep in range(epochs):
        model.train(); tot = 0; k = 0
        for o, n, c, pad in batches(all_seq, NUM, CAT, 256, True, rng):
            cl, _ = model.embed(n, c, pad)
            loss = F.cross_entropy(model.head(cl), yt[o], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); k += 1
        print(f"  adversarial epoch {ep + 1}/{epochs}  loss {tot / k:.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save({"state": model.state_dict(), "families": fams}, os.path.join(CACHE, "flowseq_adv.pt"))
    s17 = np.load(os.path.join(CACHE, "seq_2017.npy")).astype(np.int64)
    emb, proba = embed_proba(model, s17, num, cat)
    np.savez_compressed(os.path.join(CACHE, "seqscores_2017_adv.npz"), anom=np.full(len(s17), np.nan, np.float32),
                        emb=emb, proba=proba, families=np.array(fams))
    print(f"  scored 2017 ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
