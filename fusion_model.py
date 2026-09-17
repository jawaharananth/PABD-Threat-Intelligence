"""
Fusion layer: right now IsolationForest, RandomForest and CUSUM run as three
separate, uncombined signals -- app.py's dashboard shows all three but never
merges them into one number. This trains a small meta-learner (logistic
regression, calibrated) that combines:
  - IsolationForest anomaly score
  - CUSUM breach count (how many of 10 features are drifting)
  - RandomForest max class probability (classification confidence)
  - RandomForest predicted-attack flag (pred != Normal)
into a single calibrated "PABD Risk Score" in [0,1].

This is tested against each signal alone, not assumed to help. If fusion
doesn't beat the best individual signal, that's the honest result to report
-- a fusion layer earns its place in the paper only if it actually earns it
on this data.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from cusum_core import cusum_breach_matrix

FEATURES = [
    "connection_frequency", "port_entropy", "failed_auth_rate", "packet_size_dist",
    "protocol_ratio", "burst_activity", "syn_rate", "flow_duration_cv",
    "bytes_per_sec_mean", "down_up_ratio_mean",
]
K, H, N_CUSUM = 0.5, 2.0, 2  # tuned params from cusum_adaptive.py

attack_days = ["tuesday", "wednesday", "friday", "thursday_web", "friday_ddos", "friday_morning"]

scaler = joblib.load("scaler.pkl")
iso_forest = joblib.load("isolation_forest.pkl")
rf_classifier = joblib.load("random_forest.pkl")


def cusum_breach_count(df, mean, std):
    crossed = cusum_breach_matrix(df, FEATURES, mean, std, k_mult=K, h_mult=H)
    return crossed.sum(axis=1)


def build_meta_features(day):
    df = pd.read_csv(f"windows_{day}.csv")
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)

    normal = df[df["window_label"] == "BENIGN"]
    mean, std = normal[FEATURES].mean(), normal[FEATURES].std().replace(0, 1e-6)

    Xs = scaler.transform(df[FEATURES])
    iso_score = -iso_forest.score_samples(df[FEATURES])  # higher = more anomalous
    cusum_breaches = cusum_breach_count(df, mean, std)
    rf_proba = rf_classifier.predict_proba(Xs)
    rf_max_proba = rf_proba.max(axis=1)
    rf_pred = rf_classifier.predict(Xs)
    rf_is_attack = (rf_pred != 0).astype(float)

    meta = pd.DataFrame({
        "iso_score": iso_score,
        "cusum_breaches": cusum_breaches,
        "rf_max_proba": rf_max_proba,
        "rf_is_attack": rf_is_attack,
    })
    gt = (df["window_label"] != "BENIGN").astype(int).values
    return meta, gt, day


print("Building per-window meta-features (IsoForest + CUSUM + RF) across all attack days...")
all_meta, all_gt, all_day = [], [], []
for day in attack_days:
    meta, gt, d = build_meta_features(day)
    all_meta.append(meta)
    all_gt.extend(gt)
    all_day.extend([d] * len(meta))

meta_df = pd.concat(all_meta, ignore_index=True)
gt = np.array(all_gt)
day_arr = np.array(all_day)

# Chronological-style split: last 30% of EACH day held out (consistent with
# cv_eval_pooled.py's per-day split elsewhere in this repo), not a random
# shuffle across days which would leak future windows into training.
train_idx, test_idx = [], []
offset = 0
for day in attack_days:
    n = (day_arr == day).sum()
    idxs = np.arange(offset, offset + n)
    split = int(n * 0.7)
    train_idx.extend(idxs[:split])
    test_idx.extend(idxs[split:])
    offset += n
train_idx, test_idx = np.array(train_idx), np.array(test_idx)

X_train, X_test = meta_df.iloc[train_idx], meta_df.iloc[test_idx]
y_train, y_test = gt[train_idx], gt[test_idx]

meta_model = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight="balanced"), cv=3)
meta_model.fit(X_train, y_train)
joblib.dump(meta_model, "fusion_model.pkl")

fused_score = meta_model.predict_proba(X_test)[:, 1]
fused_pred = (fused_score >= 0.5).astype(int)


def report(name, pred_or_score, is_score=False):
    pred = (pred_or_score >= 0.5).astype(int) if is_score else pred_or_score
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, pred_or_score) if is_score else float("nan")
    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "n/a"
    print(f"  {name:<28} precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  AUC={auc_str}")


print(f"\nHeld-out test set: n={len(y_test)} windows, {y_test.sum()} attack windows\n")
print("Individual signals alone vs fused meta-model:")
report("IsoForest score only", X_test["iso_score"].values, is_score=True)
report("CUSUM breach count only", X_test["cusum_breaches"].values, is_score=True)
report("RF max proba only", X_test["rf_max_proba"].values, is_score=True)
report("RF is-attack flag only", X_test["rf_is_attack"].values)
report("FUSED (calibrated logistic)", fused_score, is_score=True)
