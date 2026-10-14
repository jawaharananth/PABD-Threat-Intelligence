"""Train the E-GraphSAGE layer on 2017 and score units of every prepared year.

    python -m pabd2.train_graph [--leave-out FAMILY]
"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
import torch
from . import graph
from .prepare import CACHE
from .schema import FAMILIES

torch.set_num_threads(4)


def load_year(year):
    z = np.load(os.path.join(CACHE, f"tok_{year}.npz"))
    flows = pd.read_parquet(os.path.join(CACHE, f"flows_{year}.parquet"))
    units = pd.read_parquet(os.path.join(CACHE, f"units_{year}.parquet"))
    return flows, z["num"], units


def main(leave_out=None):
    t0 = time.time()
    tag = "" if leave_out is None else "_lo_" + leave_out.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    flows, num, units = load_year(2017)
    G = graph.build(flows, num, units)
    # an edge trains if its source host's unit in that bin is a training unit
    key = pd.MultiIndex.from_frame(units[["day", "bin", "host"]])
    split = pd.Series(units["split"].to_numpy(), index=key)
    src = G["nodes"].iloc[G["s"]]
    e_split = split.reindex(pd.MultiIndex.from_frame(src[["day", "bin", "host"]])).fillna("test").to_numpy()
    classes = [f for f in FAMILIES if f != leave_out]
    print(f"graph: {len(G['X']):,} nodes, {len(G['s']):,} edges ({time.time() - t0:.0f}s)")
    model = graph.train(G, e_split == "train", classes)
    torch.save({"state": model.state_dict(), "classes": classes, "nf": G["X"].shape[1], "ef": G["ex"].shape[1]},
               os.path.join(CACHE, f"graph{tag}.pt"))
    for year in (2017, 2018):
        if year == 2018:
            if not os.path.exists(os.path.join(CACHE, "units_2018.parquet")): continue
            flows, num, units = load_year(2018); G = graph.build(flows, num, units)
        score, cls, _ = graph.unit_scores(model, G, len(units))
        np.savez_compressed(os.path.join(CACHE, f"graphscores_{year}{tag}.npz"), score=score, cls=cls,
                            classes=np.array(classes))
        print(f"  scored {len(units):,} units of {year} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--leave-out", default=None)
    main(ap.parse_args().leave_out)
