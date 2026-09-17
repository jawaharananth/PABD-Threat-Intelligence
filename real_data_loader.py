import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

CSV_DIR = "C:/Users/jawahar/Downloads/GeneratedLabelledFlows (1)/TrafficLabelling"

DAY_FILES = {
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "friday": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "thursday_web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_infil": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
}

WINDOW_SECONDS = 120

FEATURES = [
    "connection_frequency",
    "port_entropy",
    "failed_auth_rate",
    "packet_size_dist",
    "protocol_ratio",
    "burst_activity",
    "syn_rate",
    "flow_duration_cv",
    "bytes_per_sec_mean",
    "down_up_ratio_mean",
]


def load_and_clean(day_key):
    path = os.path.join(CSV_DIR, DAY_FILES[day_key])
    df = pd.read_csv(path, low_memory=False, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.dropna(subset=numeric_cols)
    df = df.drop_duplicates()
    print(f"[{day_key}] cleaned {before} -> {len(df)} rows ({100*(before-len(df))/before:.2f}% dropped)")

    # CICIDS2017's own files are inconsistent across days: most use
    # "d/m/Y H:M" (no leading zeros, no seconds), but Monday's file uses
    # zero-padded "d/m/Y H:M:S". Try both instead of assuming one format.
    parsed = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")
    still_bad = parsed.isna()
    if still_bad.any():
        parsed.loc[still_bad] = pd.to_datetime(
            df.loc[still_bad, "Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
    df["parsed_time"] = parsed
    n_bad = df["parsed_time"].isna().sum()
    if n_bad > 0:
        print(f"[{day_key}] WARNING: {n_bad} rows failed timestamp parse.")
    df = df.dropna(subset=["parsed_time"]).reset_index(drop=True)

    print(f"[{day_key}] label distribution:")
    print(df["Label"].value_counts())
    return df


def _entropy(counts):
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def compute_window(window_df):
    n = len(window_df)
    if n == 0:
        return None

    connection_frequency = n
    port_counts = window_df["Destination Port"].value_counts()
    port_entropy = _entropy(port_counts)

    auth_flows = window_df[window_df["Destination Port"].isin([21, 22])]
    if len(auth_flows) > 0:
        likely_failed = auth_flows[
            (auth_flows["Flow Duration"] < auth_flows["Flow Duration"].median())
            & (auth_flows.get("Total Fwd Packets", pd.Series([0]*len(auth_flows))) <= 3)
        ]
        failed_auth_rate = len(likely_failed) / len(auth_flows)
    else:
        failed_auth_rate = 0.0

    size_col = "Average Packet Size" if "Average Packet Size" in window_df.columns else "Packet Length Mean"
    packet_size_dist = float(window_df[size_col].mean())

    times = window_df["parsed_time"].sort_values()
    gaps = times.diff().dt.total_seconds().dropna()
    burst_activity = float(gaps.std()) if len(gaps) > 1 else 0.0

    proto_counts = window_df["Protocol"].value_counts(normalize=True)
    protocol_ratio = float(proto_counts.iloc[0]) if len(proto_counts) > 0 else 0.0

    # SYN flag rate: scan/recon signature -- many half-open connection attempts
    syn_rate = float(window_df["SYN Flag Count"].mean()) if "SYN Flag Count" in window_df.columns else 0.0

    # Flow duration coefficient of variation: scans produce many short,
    # similar-length flows (low CV); normal mixed traffic has high CV.
    fd = window_df["Flow Duration"]
    fd_mean = fd.mean()
    # len(fd) < 2 -> sample std is undefined (NaN), not just zero. Single-flow
    # windows do exist (found via a real downstream bug: two such windows
    # silently NaN-poisoned a plain numpy .mean() over the whole benign class
    # in adversarial_robustness.py). Treat "can't measure variability from
    # one sample" as 0.0 variability, consistent with burst_activity's
    # existing len(gaps)>1 else 0.0 handling above.
    flow_duration_cv = float(fd.std() / fd_mean) if len(fd) > 1 and fd_mean and fd_mean > 0 else 0.0

    # Mean flow byte rate: DoS-preparation staging shows elevated/bursty rates.
    if "Flow Bytes/s" in window_df.columns:
        fbs = window_df["Flow Bytes/s"].replace([np.inf, -np.inf], np.nan).dropna()
        bytes_per_sec_mean = float(fbs.mean()) if len(fbs) > 0 else 0.0
    else:
        bytes_per_sec_mean = 0.0

    # Down/Up ratio: traffic asymmetry -- exfil/lateral-movement flows skew
    # heavily one-directional compared to normal request/response traffic.
    down_up_ratio_mean = float(window_df["Down/Up Ratio"].mean()) if "Down/Up Ratio" in window_df.columns else 0.0

    # NEW: presence-based labeling + continuous attack fraction
    label_counts = window_df["Label"].value_counts()
    non_benign = label_counts.drop("BENIGN", errors="ignore")
    attack_flow_count = int(non_benign.sum())
    attack_flow_fraction = attack_flow_count / n

    if attack_flow_count > 0:
        window_label = non_benign.idxmax()  # dominant attack type present
    else:
        window_label = "BENIGN"

    return {
        "connection_frequency": connection_frequency,
        "port_entropy": port_entropy,
        "failed_auth_rate": failed_auth_rate,
        "packet_size_dist": packet_size_dist,
        "protocol_ratio": protocol_ratio,
        "burst_activity": burst_activity,
        "syn_rate": syn_rate,
        "flow_duration_cv": flow_duration_cv,
        "bytes_per_sec_mean": bytes_per_sec_mean,
        "down_up_ratio_mean": down_up_ratio_mean,
        "window_label": window_label,
        "attack_flow_fraction": attack_flow_fraction,
        "window_start": window_df["parsed_time"].min(),
    }


def build_windows(df, window_seconds=WINDOW_SECONDS):
    df = df.sort_values("parsed_time").reset_index(drop=True)
    start, end = df["parsed_time"].min(), df["parsed_time"].max()
    delta = pd.Timedelta(seconds=window_seconds)

    rows = []
    cur = start
    while cur < end:
        wdf = df[(df["parsed_time"] >= cur) & (df["parsed_time"] < cur + delta)]
        m = compute_window(wdf)
        if m is not None:
            rows.append(m)
        cur += delta

    return pd.DataFrame(rows)


if __name__ == "__main__":
    for day in ["tuesday", "wednesday", "friday", "thursday_web", "thursday_infil",
                "friday_ddos", "friday_morning", "monday"]:
        print(f"\n=== Processing {day} ===")
        df = load_and_clean(day)
        windows = build_windows(df)
        print(f"[{day}] built {len(windows)} windows")
        print(windows["window_label"].value_counts())
        print(windows[["window_label", "attack_flow_fraction"]].groupby("window_label").mean())
        windows.to_csv(f"windows_{day}.csv", index=False)
        print(f"[{day}] saved windows_{day}.csv")
