"""Score a prepared year with the saved 2017 models (nothing is refitted).

    python -m pabd2.score_year 2018 [--tag _lo_Botnet]
"""
import argparse, os
import numpy as np
import pandas as pd
import torch
from . import graph
from .prepare import CACHE
from .seqmodel import FlowSeq, anomaly_scores, predict_proba

torch.set_num_threads(max(1, os.cpu_count() - 1))


def main(year, tag=""):
    z = np.load(os.path.join(CACHE, f"tok_{year}.npz")); num, cat = z["num"], z["cat"]
    seq = np.load(os.path.join(CACHE, f"seq_{year}.npy")).astype(np.int64)
    ck = torch.load(os.path.join(CACHE, f"flowseq{tag}.pt")); fams = ck["families"]
    m = FlowSeq(len(fams)); m.load_state_dict(ck["state"])
    anom, emb = anomaly_scores(m, seq, num, cat); proba = predict_proba(m, seq, num, cat)
    np.savez_compressed(os.path.join(CACHE, f"seqscores_{year}{tag}.npz"), anom=anom, emb=emb, proba=proba,
                        families=np.array(fams))
    print(f"sequence scores for {len(seq):,} units of {year}")
    flows = pd.read_parquet(os.path.join(CACHE, f"flows_{year}.parquet"))
    units = pd.read_parquet(os.path.join(CACHE, f"units_{year}.parquet"))
    gk = torch.load(os.path.join(CACHE, f"graph{tag}.pt"))
    gm = graph.EGraphSAGE(gk["nf"], gk["ef"], len(gk["classes"])); gm.load_state_dict(gk["state"])
    G = graph.build(flows, num, units)
    score, cls, _ = graph.unit_scores(gm, G, len(units))
    np.savez_compressed(os.path.join(CACHE, f"graphscores_{year}{tag}.npz"), score=score, cls=cls,
                        classes=np.array(gk["classes"]))
    print(f"graph scores for {len(units):,} units of {year}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("year", type=int); ap.add_argument("--tag", default="")
    a = ap.parse_args(); main(a.year, a.tag)
