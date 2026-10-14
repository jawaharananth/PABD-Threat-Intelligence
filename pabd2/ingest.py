"""Ingest the corrected CIC-IDS datasets into compact per-day parquet files.

2017: read locally from CICIDS2017_improved.zip (all flows kept).
2018: streamed member-by-member over HTTP range requests (remotezip), so the 10 GB archive is
never stored. To fit a laptop, we keep every attack flow plus the complete flow history of a
deterministic 10% of benign source hosts -- sampling hosts rather than rows keeps per-host
sequences intact for the sequence model and graph layer.

    python -m pabd2.ingest 2017
    python -m pabd2.ingest 2018 [Wednesday-14-02-2018 ...]
"""
import hashlib, io, os, sys, time, zipfile, zlib
import numpy as np
import pandas as pd
from .schema import FEATURES, ID_COLS, canon, port_cat

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_v2")
URL18 = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip"
DAYS18 = ["Wednesday-14-02-2018", "Thursday-15-02-2018", "Wednesday-21-02-2018",
          "Friday-23-02-2018", "Friday-02-03-2018"]
HOST_KEEP = 10  # percent of benign source hosts kept in 2018


def _host_bucket(ip: pd.Series) -> np.ndarray:
    u = pd.unique(ip.astype(str))
    m = {x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100 for x in u}
    return ip.astype(str).map(m).to_numpy()


def _clean(df: pd.DataFrame, year: int, day: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    lab = df["Label"].astype(str).map(canon)
    df["family"] = [x[0] for x in lab]; df["stage"] = [x[1] for x in lab]
    df["attempted"] = np.array([x[2] for x in lab], dtype=bool)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    f = df[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float32")
    f = f.replace([np.inf, -np.inf], np.nan)
    df = pd.concat([df[ID_COLS + ["Label", "family", "stage", "attempted"]], f], axis=1)
    df = df.dropna(subset=["Timestamp"] + FEATURES)
    df["port_cat"] = port_cat(df["Dst Port"].to_numpy())
    df["year"] = np.int16(year); df["day"] = day
    return df.sort_values("Timestamp").reset_index(drop=True)


def ingest_2017():
    z = zipfile.ZipFile(os.path.join(ROOT, "raw", "CICIDS2017_improved.zip"))
    os.makedirs(os.path.join(ROOT, "2017"), exist_ok=True)
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        t0 = time.time()
        df = _clean(pd.read_csv(z.open(f"{day}.csv"), low_memory=False), 2017, day)
        df.to_parquet(os.path.join(ROOT, "2017", f"{day}.parquet"), index=False)
        print(f"2017 {day}: {len(df):,} flows, {(df.family != 'Benign').sum():,} attack, "
              f"unknown={int((df.family == 'Unknown').sum())}  ({time.time() - t0:.0f}s)", flush=True)


def _fetch_member(day, dest, chunk=16 << 20, workers=6, retries=10):
    """Download one archive member's compressed bytes: parallel ranged requests, retried, resumable."""
    import json, requests
    from concurrent.futures import ThreadPoolExecutor
    from remotezip import RemoteZip
    with RemoteZip(URL18) as z:
        info = z.getinfo(f"{day}.csv")
    hdr = requests.get(URL18, headers={"Range": f"bytes={info.header_offset}-{info.header_offset + 29}"}, timeout=60).content
    start = info.header_offset + 30 + int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little")
    size = info.compress_size
    done_path = dest + ".done"
    done = set(json.load(open(done_path))) if os.path.exists(done_path) and os.path.exists(dest) else set()
    if not os.path.exists(dest) or os.path.getsize(dest) != size:
        with open(dest, "wb") as fh: fh.truncate(size)
        done = set()
    todo = [o for o in range(0, size, chunk) if o not in done]
    sess = requests.Session()

    def get(o):
        hi = min(size, o + chunk) - 1
        for k in range(retries):
            try:
                r = sess.get(URL18, headers={"Range": f"bytes={start + o}-{start + hi}"}, timeout=180)
                if len(r.content) != hi - o + 1: raise IOError(f"short read {len(r.content)}")
                with open(dest, "r+b") as fh: fh.seek(o); fh.write(r.content)
                return o
            except Exception as ex:
                print(f"    retry {k + 1} at {o:,}: {ex}", flush=True); time.sleep(3 * (k + 1))
        raise RuntimeError("download failed")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for o in ex.map(get, todo):
            done.add(o); json.dump(sorted(done), open(done_path, "w"))
            mb = len(done) * chunk / 1e6
            print(f"    {day}: {min(mb, size / 1e6):,.0f} / {size / 1e6:,.0f} MB  "
                  f"({len(done) * chunk / 1e6 / max(1, time.time() - t0):.2f} MB/s this session)", flush=True)
    os.remove(done_path)


class _Inflate(io.RawIOBase):
    """File-like raw-deflate decompressor over a local file of compressed member bytes."""
    def __init__(self, path):
        self.f = open(path, "rb"); self.d = zlib.decompressobj(-15); self.buf = b""
    def readable(self): return True
    def readinto(self, b):
        while len(self.buf) < len(b):
            raw = self.f.read(1 << 20)
            if not raw: self.buf += self.d.flush(); break
            self.buf += self.d.decompress(raw)
        n = min(len(b), len(self.buf)); b[:n] = self.buf[:n]; self.buf = self.buf[n:]; return n
    def close(self): self.f.close(); super().close()


def ingest_2018(days):
    os.makedirs(os.path.join(ROOT, "2018"), exist_ok=True)
    for day in days:
        out = os.path.join(ROOT, "2018", f"{day}.parquet")
        if os.path.exists(out):
            print("skip", day); continue
        tmp = os.path.join(ROOT, "raw", f"{day}.deflate")
        t0 = time.time(); _fetch_member(day, tmp)
        parts = []; n_in = 0
        stream = io.BufferedReader(_Inflate(tmp), 1 << 22)
        for chunk in pd.read_csv(stream, chunksize=400_000, low_memory=False):
            n_in += len(chunk)
            chunk.columns = [c.strip() for c in chunk.columns]
            benign = chunk["Label"].astype(str).str.strip().eq("BENIGN").to_numpy()
            keep = ~benign | (_host_bucket(chunk["Src IP"]) < HOST_KEEP)
            parts.append(chunk[keep])
        stream.close()
        df = _clean(pd.concat(parts, ignore_index=True), 2018, day)
        df.to_parquet(out, index=False); os.remove(tmp)
        print(f"2018 {day}: {len(df):,} flows kept of {n_in:,}, {(df.family != 'Benign').sum():,} attack, "
              f"unknown labels: {sorted(df.loc[df.family == 'Unknown', 'Label'].unique())[:8]}  "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "2017":
        ingest_2017()
    else:
        ingest_2018(sys.argv[2:] or DAYS18)
