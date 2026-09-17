import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = [
    "connection_frequency",
    "port_entropy",
    "failed_auth_rate",
    "packet_size_dist",
    "protocol_ratio",
    "burst_activity",
]

WINDOW_SECONDS = 60
TRADITIONAL_THRESHOLD = 0.5  # "Traditional IDS" fires once attack traffic is the majority

days = ["tuesday", "wednesday", "friday"]
day_dfs = {}
for day in days:
    df = pd.read_csv(f"windows_{day}.csv")
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)
    day_dfs[day] = df

# --- Real baseline stats from BENIGN windows across all days ---
all_df = pd.concat(day_dfs.values(), ignore_index=True)
normal_df = all_df[all_df["window_label"] == "BENIGN"]
baseline_mean = normal_df[FEATURES].mean()
baseline_std = normal_df[FEATURES].std().replace(0, 1e-6)  # avoid div-by-zero

print("=== Real baseline stats (from BENIGN windows) ===")
for f in FEATURES:
    print(f"  {f}: mean={baseline_mean[f]:.4f}, std={baseline_std[f]:.4f}")

K = 0.5   # slack, per metric = 0.5 * std (standard CUSUM heuristic)
H = 5.0   # threshold, per metric = 5 * std

def cusum(series, mean, std):
    k = K * std
    h = H * std
    s = 0.0
    out = []
    for x in series:
        s = max(0, s + (x - mean - k))
        out.append(s > h)
    return np.array(out)

# --- Compute per-day CUSUM crossings and conjunction score ---
results = {}
for day, df in day_dfs.items():
    crossed = np.zeros((len(df), len(FEATURES)), dtype=bool)
    for i, f in enumerate(FEATURES):
        crossed[:, i] = cusum(df[f].values, baseline_mean[f], baseline_std[f])
    score = crossed.sum(axis=1)  # how many of 6 metrics crossed, per window
    ground_truth = (df["window_label"] != "BENIGN").astype(int).values
    results[day] = {"df": df, "score": score, "ground_truth": ground_truth}

# --- Ablation: N-of-6 conjunction rule, N = 2, 3, 4 ---
print("\n=== Conjunction rule ablation (N-of-6) ===")
for N in [2, 3, 4]:
    all_pred, all_true, lead_times = [], [], []
    for day, r in results.items():
        pred = (r["score"] >= N).astype(int)
        all_pred.extend(pred)
        all_true.extend(r["ground_truth"])

        # Lead time: find first real attack episode start vs first PABD alarm
        gt = r["ground_truth"]
        if gt.sum() == 0:
            continue
        attack_start = np.argmax(gt == 1)
        # Traditional baseline: first window where attack_flow_fraction > threshold
        trad_idx = None
        af = r["df"]["attack_flow_fraction"].values
        trad_hits = np.where(af > TRADITIONAL_THRESHOLD)[0]
        if len(trad_hits) > 0:
            trad_idx = trad_hits[0]

        pabd_hits = np.where(pred[:attack_start + 20] == 1)[0]  # search near episode
        pabd_idx = pabd_hits[0] if len(pabd_hits) > 0 else None

        if pabd_idx is not None and trad_idx is not None:
            lead_windows = trad_idx - pabd_idx
            lead_minutes = lead_windows * WINDOW_SECONDS / 60
            lead_times.append(lead_minutes)
            print(f"  [{day}] N={N}: PABD fires window {pabd_idx}, "
                  f"Traditional fires window {trad_idx}, "
                  f"lead = {lead_minutes:.1f} min")

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    tp = ((all_pred == 1) & (all_true == 1)).sum()
    fp = ((all_pred == 1) & (all_true == 0)).sum()
    fn = ((all_pred == 0) & (all_true == 1)).sum()
    tn = ((all_pred == 0) & (all_true == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"N={N}: TP={tp} FP={fp} FN={fn} TN={tn} | "
          f"precision={precision:.3f} recall={recall:.3f} FPR={fpr:.3f}")
    if lead_times:
        print(f"  Mean lead time: {np.mean(lead_times):.1f} min "
              f"(n={len(lead_times)}, std={np.std(lead_times):.1f})\n")
    else:
        print("  No valid lead-time comparisons found.\n")

# --- ROC / PR curves using conjunction score (0-6) as continuous risk score ---
all_score, all_true_full = [], []
for day, r in results.items():
    all_score.extend(r["score"])
    all_true_full.extend(r["ground_truth"])
all_score = np.array(all_score)
all_true_full = np.array(all_true_full)

auc = roc_auc_score(all_true_full, all_score)
fpr_curve, tpr_curve, _ = roc_curve(all_true_full, all_score)
prec_curve, rec_curve, _ = precision_recall_curve(all_true_full, all_score)

print(f"=== ROC-AUC using conjunction score (0-6) as risk score: {auc:.4f} ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr_curve, tpr_curve, label=f"AUC={auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve - PABD Conjunction Score")
axes[0].legend()

axes[1].plot(rec_curve, prec_curve)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")

plt.tight_layout()
plt.savefig("roc_pr_curves.png", dpi=150)
print("Saved roc_pr_curves.png")
