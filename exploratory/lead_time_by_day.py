import pandas as pd
import numpy as np

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate']
K_MULT, H_MULT, N_CONJUNCTION = 0.25, 3.0, 3

df = pd.read_csv('network_data_real.csv')
by_day = {}

for day, group in df.groupby('source_day'):
    group = group.reset_index(drop=True)
    baseline = group[group['label'] == 0]
    means = baseline[FEATURES].mean()
    stds = baseline[FEATURES].std()
    k = K_MULT * stds
    h = H_MULT * stds

    s = pd.Series(0.0, index=FEATURES)
    pabd_alarm, trad_alarm = [], []
    for _, row in group.iterrows():
        for f in FEATURES:
            s[f] = max(0.0, s[f] + (row[f] - means[f] - k[f]))
        pabd_alarm.append((s > h).sum() >= N_CONJUNCTION)
        trad_alarm.append(row['attack_flow_fraction'] > 0.5)
    group['pabd_alarm'] = pabd_alarm
    group['trad_alarm'] = trad_alarm

    is_attack = (group['label'] != 0).astype(int)
    run_id = (is_attack != is_attack.shift()).cumsum()
    leads = []
    for rid, run in group[is_attack == 1].groupby(run_id):
        start, end = run.index.min(), run.index.max()
        search_end = min(len(group) - 1, end + 5)
        pabd_hits = group.index[(group.index >= start) & (group.index <= search_end) & group['pabd_alarm']]
        trad_hits = group.index[(group.index >= start) & (group.index <= search_end) & group['trad_alarm']]
        if len(pabd_hits) and len(trad_hits):
            leads.append(trad_hits.min() - pabd_hits.min())
    by_day[day] = np.array(leads)

for day, leads in by_day.items():
    if len(leads):
        print(f"{day}: n={len(leads)}, mean lead={leads.mean():.2f} windows ({leads.mean()*60:.0f} sec), "
              f"range=[{leads.min()},{leads.max()}]")
    else:
        print(f"{day}: no matched events")
