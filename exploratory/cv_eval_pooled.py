import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate', 'burst_activity']
LABEL = 'label'
DAY_COL = 'source_day'

df = pd.read_csv('network_data_real.csv')

train_parts, test_parts = [], []
for day, group in df.groupby(DAY_COL):
    group = group.reset_index(drop=True)
    split_idx = int(len(group) * 0.7)
    train_parts.append(group.iloc[:split_idx])
    test_parts.append(group.iloc[split_idx:])

train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)

X_train, y_train = train[FEATURES], train[LABEL]
X_test, y_test = test[FEATURES], test[LABEL]

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Pooled chronological split: train={len(train)}, test={len(test)}")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, preds, zero_division=0))
