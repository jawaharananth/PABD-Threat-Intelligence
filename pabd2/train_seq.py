"""Train FlowSeq on 2017: self-supervised pre-training on benign traffic, then the family head.

Training sequences are sampled at the flow level from the training split (many more examples than
there are units); scoring happens at the unit level. Outputs go to data_v2/cache/.

    python -m pabd2.train_seq [--leave-out FAMILY]
"""
import argparse, os, time
import numpy as np
import pandas as pd
import torch
from .features import secs, sequence_index
from .prepare import CACHE
from .schema import FAMILIES
from .seqmodel import FlowSeq, anomaly_scores, embed_proba, finetune_head, predict_proba, pretrain
from .units import BIN

torch.set_num_threads(max(1, os.cpu_count() - 1))


def flow_split(flows, units):
    b = (secs(flows["Timestamp"]).astype(np.int64) // BIN)
    key = pd.DataFrame({"day": flows["day"].to_numpy(), "host": flows["Src IP"].to_numpy(), "bin": b})
    m = key.merge(units[["day", "host", "bin", "split"]], on=["day", "host", "bin"], how="left")
    return m["split"].to_numpy()


def main(leave_out=None, n_pre=120_000, per_family=15_000, seed=0):
    rng = np.random.default_rng(seed); t0 = time.time()
    tag = "" if leave_out is None else "_lo_" + leave_out.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    z = np.load(os.path.join(CACHE, "tok_2017.npz")); num, cat = z["num"], z["cat"]
    flows = pd.read_parquet(os.path.join(CACHE, "flows_2017.parquet"))
    units = pd.read_parquet(os.path.join(CACHE, "units_2017.parquet"))
    fsplit = flow_split(flows, units)
    fam = flows["family"].to_numpy(); att = flows["attempted"].to_numpy()
    train = fsplit == "train"

    # 1) self-supervised pre-training: benign training flows only
    model = FlowSeq(); pre_path = os.path.join(CACHE, "flowseq_pretrained.pt")
    if os.path.exists(pre_path):
        model.load_state_dict(torch.load(pre_path), strict=False); print("loaded pre-trained encoder")
    else:
        ben = np.flatnonzero(train & (fam == "Benign"))
        pre = sequence_index(flows, np.sort(rng.choice(ben, min(n_pre, len(ben)), replace=False)))
        print(f"pre-training on {len(pre):,} benign sequences")
        pretrain(model, pre, num, cat, epochs=3)
        torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("head")}, pre_path)
        print(f"  pre-trained ({time.time() - t0:.0f}s)")

    # 2) family head: class-capped sample of successful attacks + benign (leave-out family removed)
    fam_list = [f for f in FAMILIES if f != leave_out]
    pick = []
    for f in fam_list:
        ix = np.flatnonzero(train & (fam == f) & ~att)
        cap = per_family * (5 if f == "Benign" else 1)
        if len(ix): pick.append(rng.choice(ix, min(cap, len(ix)), replace=False))
    pick = np.sort(np.concatenate(pick))
    y = np.array([fam_list.index(f) for f in fam[pick]])
    print("head training counts:", dict(zip(fam_list, np.bincount(y, minlength=len(fam_list)))))
    finetune_head(model, len(fam_list), sequence_index(flows, pick), y, num, cat, epochs=3)
    torch.save({"state": model.state_dict(), "families": fam_list}, os.path.join(CACHE, f"flowseq{tag}.pt"))

    # 3) score every 2017 unit (and 2018 if prepared)
    for year in ((2017,) if leave_out else (2017, 2018)):
        sp = os.path.join(CACHE, f"seq_{year}.npy")
        if not os.path.exists(sp): continue
        if leave_out:      # fusion only needs the representation and the head: one pass
            seq = np.load(sp).astype(np.int64); emb, proba = embed_proba(model, seq, num, cat)
            np.savez_compressed(os.path.join(CACHE, f"seqscores_{year}{tag}.npz"), anom=np.full(len(seq), np.nan, np.float32),
                                emb=emb, proba=proba, families=np.array(fam_list))
            print(f"  scored {len(seq):,} units of {year} ({time.time() - t0:.0f}s)"); continue
        if year == 2018:
            z = np.load(os.path.join(CACHE, "tok_2018.npz")); num, cat = z["num"], z["cat"]
        seq = np.load(sp).astype(np.int64)
        anom, emb = anomaly_scores(model, seq, num, cat)
        proba = predict_proba(model, seq, num, cat)
        np.savez_compressed(os.path.join(CACHE, f"seqscores_{year}{tag}.npz"), anom=anom, emb=emb,
                            proba=proba, families=np.array(fam_list))
        print(f"  scored {len(seq):,} units of {year} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--leave-out", default=None)
    a = ap.parse_args(); main(a.leave_out)
