import pandas as pd

LABEL_MAP = {
    "BENIGN": (0, "Normal"),
    "FTP-Patator": (1, "Brute Force"),
    "SSH-Patator": (1, "Brute Force"),
    "PortScan": (2, "Reconnaissance"),
    # Renamed from "DoS Preparation": DoS Hulk/GoldenEye/slowloris/Slowhttptest/
    # Heartbleed are live, ongoing DoS attacks in CICIDS2017, not a pre-attack
    # staging phase. The old name overclaimed what this class actually is.
    "DoS Hulk": (3, "DoS"),
    "DoS GoldenEye": (3, "DoS"),
    "DoS slowloris": (3, "DoS"),
    "DoS Slowhttptest": (3, "DoS"),
    "Heartbleed": (3, "DoS"),
    # New classes from previously-unused CICIDS2017 day files.
    "DDoS": (4, "DDoS"),
    "Bot": (5, "Bot"),
    "Web Attack \x96 Brute Force": (6, "Web Attack"),
    "Web Attack \x96 XSS": (6, "Web Attack"),
    "Web Attack \x96 Sql Injection": (6, "Web Attack"),
    # Infiltration deliberately excluded: only 36 raw flows / 26 windows in the
    # source file, with attack_flow_fraction ~0.0005 per window (near-invisible
    # at flow-count level -- consistent with real infiltration being low and
    # slow, but too sparse to train a class on). Evaluated separately as a
    # small-n case study, not folded into the main classifier.
}

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

frames = []
for day in ["tuesday", "wednesday", "friday", "thursday_web", "friday_ddos", "friday_morning"]:
    df = pd.read_csv(f"windows_{day}.csv")
    df = df[df["window_label"].isin(LABEL_MAP.keys())].copy()
    df["label"] = df["window_label"].map(lambda x: LABEL_MAP[x][0])
    df["attack_type"] = df["window_label"].map(lambda x: LABEL_MAP[x][1])
    df["source_day"] = day
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
full = full.sample(frac=1, random_state=42).reset_index(drop=True)

print("Combined dataset size:", len(full))
print(full["attack_type"].value_counts())
print(full["label"].value_counts())

out_cols = FEATURES + ["label", "attack_type", "attack_flow_fraction", "source_day"]
full[out_cols].to_csv("network_data_real.csv", index=False)
print("\nSaved network_data_real.csv")
