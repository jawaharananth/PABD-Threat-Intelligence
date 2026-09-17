"""
Offline SHAP analysis for the paper's static figures. Explains the actual
saved research model (random_forest.pkl, 10-feature) instead of training a
throwaway RF from scratch with different features/hyperparameters -- the old
version trained its own separate classifier here, which meant the SHAP
figures didn't necessarily reflect the model the paper's accuracy numbers
came from.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mitre_map import LABEL_IDS

FEATURES = [
    "connection_frequency", "port_entropy", "failed_auth_rate",
    "packet_size_dist", "protocol_ratio", "burst_activity",
    "syn_rate", "flow_duration_cv", "bytes_per_sec_mean", "down_up_ratio_mean",
]
LABEL = 'label'
DAY_COL = 'source_day'
CLASS_NAMES = LABEL_IDS  # single source of truth, see mitre_map.py

df = pd.read_csv('network_data_real.csv')

train_parts, test_parts = [], []
for day, group in df.groupby(DAY_COL):
    group = group.reset_index(drop=True)
    split_idx = int(len(group) * 0.7)
    train_parts.append(group.iloc[:split_idx])
    test_parts.append(group.iloc[split_idx:])
train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

X_test, y_test = test[FEATURES], test[LABEL]

scaler = joblib.load('scaler.pkl')
clf = joblib.load('random_forest.pkl')
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_scaled)
sv = np.array(shap_values)
print(f"Raw shap_values type/shape info: {'list of ' + str(len(shap_values)) if isinstance(shap_values, list) else sv.shape}")

if isinstance(shap_values, list):
    sv3 = np.stack(shap_values, axis=-1)
elif sv.ndim == 3:
    sv3 = sv
else:
    sv3 = sv[:, :, np.newaxis]

n_classes = sv3.shape[2]
classes_present = sorted(y_test.unique())

for class_idx in range(n_classes):
    class_name = CLASS_NAMES.get(class_idx, f"class_{class_idx}")
    n_in_class = (y_test == class_idx).sum() if class_idx in classes_present else 0
    if n_in_class == 0:
        print(f"Skipping {class_name}: 0 test-set samples of this class "
              f"(too small to plot meaningfully).")
        continue
    plt.figure()
    shap.summary_plot(sv3[:, :, class_idx], X_test_scaled, feature_names=FEATURES, show=False, plot_type="bar")
    plt.title(f'SHAP feature importance -- {class_name} (n={n_in_class} test samples)')
    plt.tight_layout()
    fname = f'shap_summary_{class_name.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

mean_abs = np.abs(sv3).mean(axis=(0, 2))
importance_df = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False)
print("\nOverall SHAP feature importance (mean |SHAP| across classes):")
print(importance_df.to_string(index=False))

attack_test_idx = test[test[LABEL] != 0].index
sample_idx = attack_test_idx[0] if len(attack_test_idx) else test.index[0]
pos = X_test_scaled.index.get_loc(sample_idx)
sample_row = X_test_scaled.loc[[sample_idx]]
sample_class = int(y_test.loc[sample_idx])
pred_class = clf.predict(sample_row)[0]
print(f"\nExample explained prediction: true={CLASS_NAMES[sample_class]}, predicted={CLASS_NAMES[pred_class]}")
sv_row = sv3[pos, :, pred_class]
for f, v in sorted(zip(FEATURES, sv_row), key=lambda x: -abs(x[1])):
    print(f"  {f}: SHAP={v:+.4f}, scaled_value={sample_row[f].values[0]:.4f}")
