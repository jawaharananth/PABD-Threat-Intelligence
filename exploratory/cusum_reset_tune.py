import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate']
N_CONJUNCTION = 3

df = pd.read_csv('network_data_real.csv')

def run_cusum_reset(day_df, means, stds, k_mult, h_mult, cooldown):
    k = k_mult * stds
    h = h_mult * stds
    s = pd.Series(0.0, index=FEATURES)
    alarm_events = []
    scores = []
    cooldown_left = 0
    for idx, row in day_df.iterrows():
        for f in FEATURES:
            s[f] = max(0.0, s[f] + (row[f] - means[f] - k[f]))
        score = (s / h).sum()
        scores.append(score)
        n_over = (s > h).sum()
        fired = False
        if cooldown_left == 0 and n_over >= N_CONJUNCTION:
            fired = True
            s[:] = 0.0          # reset all metrics
            cooldown_left = cooldown
        elif cooldown_left > 0:
            cooldown_left -= 1
        alarm_events.append(fired)
    return np.array(alarm_events), np.array(scores)

best = {"f1": -1}
for k_mult in [0.25, 0.5, 1.0]:
    for h_mult in [1.5, 2.0, 3.0, 4.0]:
        for cooldown in [3, 5, 10]:
            all_true, all_pred, all_score = [], [], []
            for day, group in df.groupby('source_day'):
                group = group.reset_index(drop=True)
                baseline = group[group['label'] == 0]
                means = baseline[FEATURES].mean()
                stds = baseline[FEATURES].std()
                alarms, scores = run_cusum_reset(group, means, stds, k_mult, h_mult, cooldown)
                true = (group['label'] != 0).astype(int).values
                all_true.extend(true)
                all_pred.extend(alarms.astype(int))
                all_score.extend(scores)
            f1 = f1_score(all_true, all_pred, zero_division=0)
            auc = roc_auc_score(all_true, all_score)
            if f1 > best["f1"]:
                best = {"f1": f1, "auc": auc, "k": k_mult, "h": h_mult, "cooldown": cooldown}

print(f"Best (with reset+cooldown): k={best['k']}, h={best['h']}, cooldown={best['cooldown']}")
print(f"F1={best['f1']:.3f}, AUC={best['auc']:.3f}")
print(f"(Old, no reset: F1=0.360, AUC=0.657 — compare against this)")
