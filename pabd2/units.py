"""Detection units: one per (source host, 10-second bin).

A SOC sees alerts per host, not per flow. Each unit is scored with the host's sequence that ends
at its last flow in the bin. Ground truth: the majority family among *successful* attack flows in
the bin; bins containing only "- Attempted" flows are marked attempted and excluded from metrics.

Split (2017): within each day, every attack episode (family) is split chronologically at its
60th-percentile unit (count, not time span: attacks run in bursts with long gaps), and
benign traffic at the day's 60% point. Units whose sequence straddles their split boundary are
purged, so no test sequence contains a training flow.
"""
import numpy as np
import pandas as pd
from .features import secs, sequence_index

BIN = 10  # seconds


def build_units(df: pd.DataFrame, split=True, train_frac=0.6) -> pd.DataFrame:
    """df: tokenized flow frame sorted by (Src IP, Timestamp)."""
    t = secs(df["Timestamp"]).astype(np.int64)
    b = t // BIN
    fam = df["family"].to_numpy(); att = df["attempted"].to_numpy()
    is_att = (fam != "Benign") & ~att
    g = pd.DataFrame({"host": df["Src IP"].to_numpy(), "day": df["day"].to_numpy(), "bin": b,
                      "i": np.arange(len(df)), "att": is_att, "fam": np.where(is_att, fam, ""),
                      "tried": (fam != "Benign") & att, "dst": df["Dst IP"].to_numpy()})
    agg = g.groupby(["day", "host", "bin"], sort=False).agg(
        end=("i", "max"), n=("i", "size"), n_att=("att", "sum"), tried=("tried", "any"), n_dst=("dst", "nunique"))
    fams = g[g.att].groupby(["day", "host", "bin"])["fam"].agg(lambda s: s.value_counts().index[0])
    u = agg.join(fams.rename("family")).reset_index()
    u["family"] = u["family"].fillna("Benign")
    u.loc[(u.family == "Benign") & u.tried, "family"] = "Attempted"
    u["t"] = pd.to_datetime(u["bin"] * BIN, unit="s")
    seq = sequence_index(df, u["end"].to_numpy())
    first = np.where(seq >= 0, seq, np.iinfo(np.int64).max).min(1)
    u["t_seq_start"] = pd.to_datetime(t[first], unit="s")
    if split:
        u["split"] = "train"
        for day, d in u.groupby("day"):
            day_cut = d.t.min() + train_frac * (d.t.max() - d.t.min())
            cut = pd.Series(day_cut, index=d.index)
            for f, e in d[~d.family.isin(["Benign", "Attempted"])].groupby("family"):
                ts = e.t.sort_values()
                cut[e.index] = ts.iloc[min(len(ts) - 1, int(train_frac * len(ts)))]
            test = d.t >= cut
            u.loc[d.index[test], "split"] = "test"
            u.loc[d.index[test & (d.t_seq_start < cut)], "split"] = "purged"
    return u, seq
