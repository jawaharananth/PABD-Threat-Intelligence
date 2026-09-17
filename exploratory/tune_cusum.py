import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import itertools

FEATURES = [
    "connection_frequency",
    "port_entropy",
    "failed_auth_rate",
    "packet_size_dist",
    "protocol_ratio",
    # burst_activity dropped -- dead metric (std=0) at this timestamp resolution
]

WINDOW_SECONDS = 60
days = ["tuesday", "wednesday", "friday"]
day_dfs = {}
for day in days:
    df = pd.read_csv(f"windows_{day}.csv")
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)
    day_dfs[day] = df

all_df = pd.concat(day_dfs.values(), ignore_index=True)
normal_df = all_df[all_df["window_label"] == "BENIGN"]
baseline_mean = normal_df[FEATURES].mean()
baseline_std = normal_df[FEATURES].std().replace(0, 1e-6)

def cusum(series, mean, std, k_mult, h_mult):
    k = k_mult * std
    h = h_mult * std
    s = 0.0
    out = []
    for x in series:
        s = max(0, s + (x - mean - k))
        out.append(s > h)
    return np.array(out)

k_grid = [0.25, 0.5, 1.0]
h_grid = [1.5, 2.0, 3.0, 4.0]

best = {"f1": -1}
print("Grid search over (k_mult, h_mult), N=3-of-5 conjunction rule:\n")
for k_mult, h_mult in itertools.product(k_grid, h_grid):
    all_pred, all_true, all_score = [], [], []
    for day, df in day_dfs.items():
        crossed = np.zeros((len(df), len(FEATURES)), dtype=bool)
        for i, f in enumerate(FEATURES):
            crossed[:, i] = cusum(df[f].values, baseline_mean[f], baseline_std[f], k_mult, h_mult)
        score = crossed.sum(axis=1)
        pred = (score >= 3).astype(int)
        gt = (df["window_label"] != "BENIGN").astype(int).values
        all_pred.extend(pred)
        all_true.extend(gt)
        all_score.extend(score)

    f1 = f1_score(all_true, all_pred, zero_division=0)
    auc = roc_auc_score(all_true, all_score)
    print(f"  k={k_mult}, h={h_mult}: F1={f1:.3f}, AUC={auc:.3f}")
    if f1 > best["f1"]:
        best = {"f1": f1, "auc": auc, "k": k_mult, "h": h_mult}

print(f"\nBest: k={best['k']}, h={best['h']} -> F1={best['f1']:.3f}, AUC={best['auc']:.3f}")
