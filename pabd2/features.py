"""Token features and per-host flow sequences.

Every flow becomes a token: its 57 behavioural features (signed log + robust scaling fitted on
benign training traffic only), plus context a defender actually has without identities -- port
category, protocol, time since the host's previous flow, whether this destination is new for the
host within the sequence, and whether the destination is internal. A host's sequence is its last
SEQ_LEN flows in time order, so every score is causal: it never looks at the future.
"""
import glob, os
import numpy as np
import pandas as pd
from .schema import FEATURES, PORT_CATS

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_v2")
SEQ_LEN = 32
N_NUM = len(FEATURES) + 3          # + log dt, new-destination flag, internal-destination flag
PROTO = {6: 0, 17: 1, 1: 2}         # tcp, udp, icmp, other=3


def load(year: int, days=None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ROOT, str(year), "*.parquet")))
    if days: files = [f for f in files if any(d in f for d in days)]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def secs(ts) -> np.ndarray:
    """Seconds since epoch, independent of the datetime unit pandas chose (us vs ns)."""
    return ts.to_numpy().astype("datetime64[ns]").astype(np.int64) / 1e9


def is_internal(ip: pd.Series) -> np.ndarray:
    s = ip.astype(str)
    return (s.str.startswith("192.168.") | s.str.startswith("10.") | s.str.startswith("172.31.")).to_numpy()


class Scaler:
    """Signed-log then median/IQR scaling, fitted on benign training flows only."""
    def fit(self, X):
        L = np.sign(X) * np.log1p(np.abs(X))
        self.med = np.median(L, 0); iqr = np.subtract(*np.percentile(L, [75, 25], 0))
        self.iqr = np.where(iqr < 1e-3, 1.0, iqr); return self
    def transform(self, X, chunk=500_000):
        out = np.empty(X.shape, np.float32)
        for s in range(0, len(X), chunk):
            x = np.asarray(X[s:s + chunk], dtype=np.float64)
            L = np.sign(x) * np.log1p(np.abs(x))
            out[s:s + chunk] = np.clip((L - self.med) / self.iqr, -8, 8)
        return out


def tokenize(df: pd.DataFrame, scaler: Scaler):
    """Returns per-flow numeric tokens, categorical ids, and per-flow host order (sorted)."""
    df = df.sort_values(["Src IP", "Timestamp"]).reset_index(drop=True)
    num = scaler.transform(df[FEATURES].to_numpy(np.float32))
    t = secs(df["Timestamp"])
    same = np.r_[False, df["Src IP"].to_numpy()[1:] == df["Src IP"].to_numpy()[:-1]]
    dt = np.where(same, np.r_[0, np.diff(t)], 3600.0)
    log_dt = (np.log1p(np.clip(dt, 0, 3600)) / np.log1p(3600)).astype(np.float32)
    # "new destination for this host in the recent past" (last 256 flows of the host)
    key = df["Src IP"].astype(str) + ">" + df["Dst IP"].astype(str)
    grp = df.groupby("Src IP", sort=False).cumcount().to_numpy()
    last_seen = {}; newdst = np.zeros(len(df), np.float32)
    for i, (k, g) in enumerate(zip(key.to_numpy(), grp)):
        p = last_seen.get(k)
        newdst[i] = 1.0 if p is None or g - p > 256 else 0.0
        last_seen[k] = g
    internal = is_internal(df["Dst IP"]).astype(np.float32)
    num = np.concatenate([num, log_dt[:, None], newdst[:, None], internal[:, None]], 1)
    port = df["port_cat"].map({c: i for i, c in enumerate(PORT_CATS)}).fillna(len(PORT_CATS) - 1).astype(np.int64).to_numpy()
    proto = df["Protocol"].map(PROTO).fillna(3).astype(np.int64).to_numpy()
    return df, num, np.stack([port, proto], 1)


def sequence_index(df: pd.DataFrame, ends: np.ndarray):
    """For each flow index in `ends`, the indices of the host's SEQ_LEN flows ending at it
    (left-padded with -1). df must be sorted by (Src IP, Timestamp)."""
    host_start = df.groupby("Src IP", sort=False).cumcount().to_numpy()
    offs = np.arange(-SEQ_LEN + 1, 1)
    idx = ends[:, None] + offs[None, :]
    valid = offs[None, :] >= -host_start[ends][:, None]
    return np.where(valid, idx, -1)
