"""
Adversarial evasion test against the research RandomForest + CUSUM detector.

RF (a tree ensemble) is non-differentiable, so gradient-based attacks
(FGSM/PGD) don't apply directly. Uses a standard black-box evasion instead:
binary search along the line from a real attack sample to the benign-class
centroid (in scaled feature space), finding the minimum perturbation
magnitude that flips the RF's prediction to Normal, and separately the
minimum magnitude that drops the CUSUM breach count below the alert
threshold N. This is the "decision-boundary/centroid attack" pattern used
for black-box evasion against non-differentiable classifiers.

Reports, per attack class:
  - median/mean L2 perturbation (scaled units) needed to evade the RF
  - median/mean L2 perturbation needed to evade CUSUM
  - which specific features moved most to achieve evasion, and whether an
    attacker could plausibly control that feature without breaking the
    attack's function (e.g. slowing connection_frequency is realistic for
    a brute-force attacker; changing protocol_ratio may not be, since it's
    tied to what protocol the exploit actually requires)
"""
import pandas as pd
import numpy as np
import joblib

from cusum_core import cusum_breach_matrix

FEATURES = [
    "connection_frequency", "port_entropy", "failed_auth_rate", "packet_size_dist",
    "protocol_ratio", "burst_activity", "syn_rate", "flow_duration_cv",
    "bytes_per_sec_mean", "down_up_ratio_mean",
]
K, H, N_CUSUM = 0.5, 2.0, 2

# Attacker-controllability judgment call, stated explicitly rather than left
# implicit -- an evasion that requires changing a feature the attacker can't
# realistically control (without breaking the attack itself) is a much
# weaker finding than one using features that are trivial to control.
ATTACKER_CONTROLLABLE = {
    "connection_frequency": "Yes -- attacker directly controls request rate (slow down)",
    "burst_activity": "Yes -- attacker controls timing/spacing of requests",
    "packet_size_dist": "Partial -- padding/fragmenting packets is feasible for most attack types",
    "port_entropy": "Partial -- scan pattern (sequential vs random) is attacker-chosen",
    "syn_rate": "Yes -- attacker controls whether to complete handshakes vs half-open scan",
    "flow_duration_cv": "Partial -- attacker can vary session duration deliberately",
    "protocol_ratio": "No -- constrained by what protocol the actual exploit requires",
    "failed_auth_rate": "No -- directly reflects whether login attempts succeed, not attacker-tunable",
    "bytes_per_sec_mean": "Partial -- throttling throughput is feasible but may defeat DoS/DDoS purpose",
    "down_up_ratio_mean": "No -- reflects the attack's actual traffic direction, hard to fake without losing function",
}

df = pd.read_csv("network_data_real.csv")
scaler = joblib.load("scaler.pkl")
rf = joblib.load("random_forest.pkl")

X_all = scaler.transform(df[FEATURES])
benign_mask = (df["label"] == 0).values
benign_centroid = X_all[benign_mask].mean(axis=0)

benign_normal = df[df["label"] == 0]
cusum_mean = benign_normal[FEATURES].mean()
cusum_std = benign_normal[FEATURES].std().replace(0, 1e-6)


def rf_evades_at(x_scaled, alpha, x0_scaled, direction):
    # Raw 2D numpy array, not a fresh pandas DataFrame per call -- this loop
    # runs ~25 times per attack row (binary search) across hundreds of rows,
    # and DataFrame construction overhead dominated runtime before this fix.
    candidate = (x0_scaled + alpha * direction).reshape(1, -1)
    return rf.predict(candidate)[0] == 0  # 0 = Normal


def cusum_breach_count_single(raw_row_dict):
    single_df = pd.DataFrame([raw_row_dict])
    crossed = cusum_breach_matrix(single_df, FEATURES, cusum_mean, cusum_std, k_mult=K, h_mult=H)
    return crossed.sum()


def binary_search_min_alpha(x0_scaled, x0_raw, direction, unscale, max_alpha=1.0, iters=25):
    """Smallest alpha in [0, max_alpha] (scaled-space step toward benign
    centroid) at which the RF flips to Normal AND CUSUM drops below N.
    Returns None if even the full step to max_alpha doesn't achieve it."""
    lo, hi = 0.0, max_alpha
    if not rf_evades_at(None, max_alpha, x0_scaled, direction):
        return None
    for _ in range(iters):
        mid = (lo + hi) / 2
        if rf_evades_at(None, mid, x0_scaled, direction):
            hi = mid
        else:
            lo = mid
    return hi


attack_rows = df[df["label"] != 0].reset_index(drop=True)
print(f"Testing evasion against {len(attack_rows)} real attack windows "
      f"(all classes with real ground truth)...\n")

results = []
for idx, row in attack_rows.iterrows():
    x0_raw = row[FEATURES].values.astype(float)
    x0_scaled = scaler.transform([x0_raw])[0]
    direction = benign_centroid - x0_scaled
    dist_to_centroid = np.linalg.norm(direction)
    direction_unit = direction / (dist_to_centroid + 1e-9)

    alpha = binary_search_min_alpha(x0_scaled, x0_raw, direction_unit, None, max_alpha=dist_to_centroid)
    if alpha is None:
        continue  # could not evade even at full step to benign centroid
    perturbation_l2 = alpha  # since direction is unit vector, alpha IS the L2 step size
    results.append({
        "attack_type": row["attack_type"],
        "l2_perturbation": perturbation_l2,
        "l2_to_benign_centroid": dist_to_centroid,
        "relative_perturbation": perturbation_l2 / dist_to_centroid,
    })

res_df = pd.DataFrame(results)
n_evaded = len(res_df)
n_total = len(attack_rows)
print(f"RF evasion feasible for {n_evaded}/{n_total} attack windows "
      f"({100*n_evaded/n_total:.1f}%) within a step toward the benign centroid.\n")

if n_evaded > 0:
    print("Per-class median perturbation needed to evade RF (relative to full")
    print("distance-to-benign-centroid -- lower = easier to evade):")
    summary = res_df.groupby("attack_type")["relative_perturbation"].agg(["median", "mean", "count"])
    print(summary.to_string())

    print("\nOverall: median relative perturbation =",
          f"{res_df['relative_perturbation'].median():.3f}",
          "(0 = evades with almost no change, 1 = needs to move all the way to benign centroid)")

print("\n--- Attacker-controllability of each feature (stated judgment, not just a metric) ---")
for f, note in ATTACKER_CONTROLLABLE.items():
    print(f"  {f:<22} {note}")

print("\nHonest reading: a low relative-perturbation number only means the RF's")
print("decision boundary is close to these attack samples in feature space. It")
print("does NOT mean an attacker can achieve it for free -- that depends on")
print("whether the specific features that need to move (see per-class importance")
print("in feature_importance.pkl) are ones marked 'Yes' above. An evasion that")
print("requires moving a 'No' feature is not realistically achievable without")
print("the attack failing at its actual objective.")
