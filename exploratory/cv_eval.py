import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate', 'burst_activity']
LABEL = 'label'
DAY_COL = 'source_day'

df = pd.read_csv('network_data_real.csv')

if DAY_COL not in df.columns:
    raise ValueError(f"No '{DAY_COL}' column found. Columns present: {list(df.columns)}. Fix DAY_COL above.")

results = []
for day, group in df.groupby(DAY_COL):
    group = group.reset_index(drop=True)
    split_idx = int(len(group) * 0.7)
    train = group.iloc[:split_idx]
    test = group.iloc[split_idx:]

    if test[LABEL].nunique() < 2:
        print(f"[WARN] {day}: test split has <2 classes, skipping fold")
        continue

    X_train, y_train = train[FEATURES], train[LABEL]
    X_test, y_test = test[FEATURES], test[LABEL]

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append(acc)

    print(f"\n=== {day} (train={len(train)}, test={len(test)}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

print("\n=== CHRONOLOGICAL CV SUMMARY ===")
print(f"Per-day accuracies: {[round(r,4) for r in results]}")
print(f"Mean: {np.mean(results):.4f} ± {np.std(results):.4f}")
