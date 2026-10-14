"""Tokenize a year and build its detection units; cache to data_v2/cache/.

    python -m pabd2.prepare 2017        (fits the scaler on 2017 benign training flows)
    python -m pabd2.prepare 2018        (reuses the 2017 scaler: nothing is refitted on 2018)
"""
import os, pickle, sys, time
import numpy as np
from .features import ROOT, Scaler, load, tokenize
from .schema import FEATURES
from .units import build_units

CACHE = os.path.join(ROOT, "cache")


def prepare(year):
    os.makedirs(CACHE, exist_ok=True); t0 = time.time()
    df = load(year)
    sp = os.path.join(CACHE, "scaler.pkl")
    if year == 2017:
        df0 = df.sort_values(["Src IP", "Timestamp"]).reset_index(drop=True)
        # provisional split to fit the scaler on benign training flows only
        u0, _ = build_units(df0)
        tr_bins = u0[u0.split == "train"]
        cut = {d: g.t.max() for d, g in tr_bins.groupby("day")}
        tmask = (df0.family == "Benign") & (df0.Timestamp <= df0.day.map(cut))
        scaler = Scaler().fit(df0.loc[tmask, FEATURES].to_numpy(np.float64)[::5])
        pickle.dump(scaler, open(sp, "wb"))
    else:
        scaler = pickle.load(open(sp, "rb"))
    df, num, cat = tokenize(df, scaler)
    units, seq = build_units(df, split=(year == 2017))
    if year != 2017: units["split"] = "test"
    np.savez_compressed(os.path.join(CACHE, f"tok_{year}.npz"), num=num, cat=cat)
    np.save(os.path.join(CACHE, f"seq_{year}.npy"), seq.astype(np.int32))
    units.to_parquet(os.path.join(CACHE, f"units_{year}.parquet"), index=False)
    df[["Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp", "family", "stage", "attempted", "day", "port_cat"]
       ].to_parquet(os.path.join(CACHE, f"flows_{year}.parquet"), index=False)
    print(f"{year}: {len(df):,} flows -> {len(units):,} units  ({time.time() - t0:.0f}s)")
    print(units.groupby(["split", "family"]).size().unstack(0).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    prepare(int(sys.argv[1]))
