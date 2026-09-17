"""
CUSUM with rolling baseline recompute -- addresses the concept-drift /
seasonality weakness documented in CUSUM literature (WAVE-CUSUM, Farkas 2016):
a single fixed baseline computed once at the start goes stale as normal
traffic characteristics drift over a monitoring period, inflating false
positives over time.

This does NOT claim to solve seasonality the way wavelet-filtered CUSUM does
(that requires longer traffic history with real daily/weekly cycles than a
single-day CICIDS capture has -- there is no multi-day seasonal pattern to
filter out of a ~2-3 hour capture). What it does fix: the baseline adapts as
new confirmed-BENIGN windows arrive, instead of assuming the first day's mean
holds for the entire monitoring window. Report both the fixed-baseline and
adaptive-baseline results side by side -- if they're close, that's evidence
the fixed baseline wasn't actually a problem for this dataset's timespan; if
they diverge, the adaptive version is the more defensible number to publish.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from cusum_core import CusumTracker, cusum_breach_matrix

# Full 10-feature set (original 6 + syn_rate/flow_duration_cv/bytes_per_sec_mean/
# down_up_ratio_mean engineered from raw CICIDS columns). None are dead metrics
# at 120s window resolution -- all have non-trivial BENIGN std, checked before
# use. K/H/N below are the grid-search best over this feature set
# (k=0.5, h=2.0, N=2-of-10), which beat the old 5-feature tuned result
# (F1 0.385 -> 0.471) -- use these, not hand-picked values.
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

WINDOW_SECONDS = 120
K, H, N_CONJUNCTION = 0.5, 2.0, 2
ROLLING_WINDOW = 50  # recompute baseline over the last N confirmed-BENIGN windows

days = ["tuesday", "wednesday", "friday", "thursday_web", "friday_ddos", "friday_morning"]
day_dfs = {}
for day in days:
    df = pd.read_csv(f"windows_{day}.csv")
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)
    day_dfs[day] = df


def cusum_fixed(df):
    normal = df[df["window_label"] == "BENIGN"]
    mean, std = normal[FEATURES].mean(), normal[FEATURES].std().replace(0, 1e-6)
    crossed = cusum_breach_matrix(df, FEATURES, mean, std, k_mult=K, h_mult=H)
    return (crossed.sum(axis=1) >= N_CONJUNCTION).astype(int)


def cusum_adaptive(df):
    """Baseline recomputed from the trailing ROLLING_WINDOW confirmed-BENIGN
    windows seen so far (expanding then rolling), instead of one fixed value
    for the whole timeline. Falls back to global stats until enough BENIGN
    history has accumulated."""
    global_normal = df[df["window_label"] == "BENIGN"]
    fallback_mean, fallback_std = global_normal[FEATURES].mean(), global_normal[FEATURES].std().replace(0, 1e-6)

    trackers = {f: CusumTracker() for f in FEATURES}
    pred = np.zeros(len(df), dtype=int)
    benign_history = []  # list of feature dicts, BENIGN windows only, chronological

    for i, row in df.iterrows():
        if len(benign_history) >= 5:
            hist_df = pd.DataFrame(benign_history[-ROLLING_WINDOW:])
            mean, std = hist_df.mean(), hist_df.std().replace(0, 1e-6)
        else:
            mean, std = fallback_mean, fallback_std

        crossed = 0
        for f in FEATURES:
            if trackers[f].update(row[f], mean[f], std[f], k_mult=K, h_mult=H):
                crossed += 1
        pred[i] = int(crossed >= N_CONJUNCTION)

        if row["window_label"] == "BENIGN":
            benign_history.append({f: row[f] for f in FEATURES})

    return pred


def score(pred, gt, label):
    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    fp = ((pred == 1) & (gt == 0)).sum()
    tn = ((pred == 0) & (gt == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = f1_score(gt, pred, zero_division=0)
    print(f"  [{label}] precision={precision:.3f} recall={recall:.3f} FPR={fpr:.3f} F1={f1:.3f}")
    return precision, recall, fpr, f1


print("=== Fixed baseline vs rolling-adaptive baseline CUSUM, per day ===\n")
all_fixed_pred, all_adapt_pred, all_gt = [], [], []
for day, df in day_dfs.items():
    gt = (df["window_label"] != "BENIGN").astype(int).values
    fixed_pred = cusum_fixed(df)
    adapt_pred = cusum_adaptive(df)

    print(f"{day} (n={len(df)}):")
    score(fixed_pred, gt, "fixed baseline  ")
    score(adapt_pred, gt, "rolling baseline")
    print()

    all_fixed_pred.extend(fixed_pred)
    all_adapt_pred.extend(adapt_pred)
    all_gt.extend(gt)

print("=== Pooled across all days ===")
score(np.array(all_fixed_pred), np.array(all_gt), "fixed baseline  ")
score(np.array(all_adapt_pred), np.array(all_gt), "rolling baseline")
