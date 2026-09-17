import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load data
print("Loading data...")
df = pd.read_csv('network_data_real.csv')

FEATURES = [
    'connection_frequency',
    'port_entropy',
    'failed_auth_rate',
    'packet_size_dist',
    'protocol_ratio',
    'burst_activity',
    'syn_rate',
    'flow_duration_cv',
    'bytes_per_sec_mean',
    'down_up_ratio_mean',
]

# ─── SCALER ───────────────────────────────────────────
print("Training scaler...")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved.")

# ─── ISOLATION FOREST ─────────────────────────────────
print("Training Isolation Forest...")
normal_data = df[df['label'] == 0][FEATURES]
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
iso_forest.fit(normal_data)
joblib.dump(iso_forest, 'isolation_forest.pkl')
print("Isolation Forest saved.")

# ─── RANDOM FOREST ────────────────────────────────────
print("Training Random Forest classifier...")
X = df[FEATURES]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight='balanced'
)
rf_classifier.fit(X_train_scaled, y_train)
joblib.dump(rf_classifier, 'random_forest.pkl')

# Evaluation
# Label ids follow consolidate_data.py's LABEL_MAP: 0 Normal, 1 Brute Force,
# 2 Reconnaissance, 3 DoS, 4 DDoS, 5 Bot, 6 Web Attack. labels= passed
# explicitly since the smallest classes (DDoS n=11) can vanish from a random
# test split by chance, which would otherwise misalign target_names.
LABEL_NAMES = {0: 'Normal', 1: 'Brute Force', 2: 'Reconnaissance', 3: 'DoS',
               4: 'DDoS', 5: 'Bot', 6: 'Web Attack'}
y_pred = rf_classifier.predict(X_test_scaled)
present_labels = sorted(set(y_test) | set(y_pred))
print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test, y_pred,
    labels=present_labels,
    target_names=[LABEL_NAMES[l] for l in present_labels],
    zero_division=0,
))

# Feature importance
importances = rf_classifier.feature_importances_
feature_importance_dict = dict(zip(FEATURES, importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for feat, imp in sorted_features:
    print(f"  {feat}: {imp:.4f}")

joblib.dump(feature_importance_dict, 'feature_importance.pkl')
print("\nRandom Forest saved.")

# ─── MITRE ATT&CK COVERAGE TABLE ───────────────────────
from mitre_map import tag as mitre_tag
print("\nMITRE ATT&CK coverage (real-data classes only -- Infiltration/Lateral")
print("Movement excluded, see consolidate_data.py / README for why):")
counts = df['attack_type'].value_counts()
print(f"{'Class':<16}{'n windows':<11}{'Tactic':<22}{'Technique':<45}")
for cls in LABEL_NAMES.values():
    if cls == 'Normal' or cls not in counts.index:
        continue
    m = mitre_tag(cls)
    technique = f"{m['technique_id']} {m['technique_name']}" if m['technique_id'] else '-'
    print(f"{cls:<16}{counts[cls]:<11}{m['tactic']:<22}{technique:<45}")

# LSTM training moved to train_lstm_countdown_real.py -- it trains on real
# CICIDS attack-onset timelines (windows_*.csv) instead of data_generator.py's
# synthetic generate_pre_attack_sequence(), which only produces the original
# 6-feature schema and would break against this file's now-expanded FEATURES.
# Run train_lstm_countdown_real.py separately after this script.

# ─── DEMO-SCALE MODELS (original 6 features) ──────────
# app.py's live demo only simulates the original 6-feature schema, so the
# 10-feature iso_forest/rf_classifier above can't score its live traffic
# (dimension mismatch). Train matching 6-feature versions here instead of
# silently pointing the demo at models it can't actually call. The 10-feature
# models above are the ones to report in the paper/ablation -- these _demo6
# files exist only to keep the live dashboard functional.
DEMO_FEATURES = [
    'connection_frequency', 'port_entropy', 'failed_auth_rate',
    'packet_size_dist', 'protocol_ratio', 'burst_activity',
]
print("\nTraining demo-scale (6-feature) models for app.py + countdown LSTM...")
scaler_demo6 = StandardScaler()
scaler_demo6.fit(df[DEMO_FEATURES])
joblib.dump(scaler_demo6, 'scaler_demo6.pkl')

normal_data_demo6 = df[df['label'] == 0][DEMO_FEATURES]
iso_forest_demo6 = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest_demo6.fit(normal_data_demo6)
joblib.dump(iso_forest_demo6, 'isolation_forest_demo6.pkl')

X6, y6 = df[DEMO_FEATURES], df['label']
X6_train, X6_test, y6_train, y6_test = train_test_split(
    X6, y6, test_size=0.2, random_state=42, stratify=y6
)
X6_train_scaled = scaler_demo6.transform(X6_train)
X6_test_scaled = scaler_demo6.transform(X6_test)
rf_demo6 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
rf_demo6.fit(X6_train_scaled, y6_train)
joblib.dump(rf_demo6, 'random_forest_demo6.pkl')

feature_importance_demo6 = dict(zip(DEMO_FEATURES, rf_demo6.feature_importances_))
joblib.dump(feature_importance_demo6, 'feature_importance_demo6.pkl')
print("Demo-scale (6-feature) models saved: scaler_demo6.pkl, isolation_forest_demo6.pkl, "
      "random_forest_demo6.pkl, feature_importance_demo6.pkl")

# ─── BASELINE STATS ───────────────────────────────────
print("\nSaving baseline statistics...")
normal_df = df[df['label'] == 0][FEATURES]
baseline_stats = {
    'mean': normal_df.mean().to_dict(),
    'std': normal_df.std().to_dict()
}
joblib.dump(baseline_stats, 'baseline_stats.pkl')
print("Baseline stats saved.")

print("\nALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print("Files created:")
for f in ['scaler.pkl', 'isolation_forest.pkl', 'random_forest.pkl', 'baseline_stats.pkl',
          'feature_importance.pkl', 'scaler_demo6.pkl', 'isolation_forest_demo6.pkl',
          'random_forest_demo6.pkl', 'feature_importance_demo6.pkl']:
    if os.path.exists(f):
        print(f"  OK {f}")
    else:
        print(f"  MISSING {f}")
        
