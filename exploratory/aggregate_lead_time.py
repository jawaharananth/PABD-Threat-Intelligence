import pandas as pd
import numpy as np

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate']
K_MULT, H_MULT, N_CONJUNCTION = 0.25, 3.0, 3

df = pd.read_csv('network_data_real.csv')
all_lead_times = []
event_log = []

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
    for rid, run in group[is_attack == 1].groupby(run_id):
        start = run.index.min()
        end = run.index.max()
        search_end = min(len(group) - 1, end + 5)
        pabd_hits = group.index[(group.index >= start) & (group.index <= search_end) & group['pabd_alarm']]
        trad_hits = group.index[(group.index >= start) & (group.index <= search_end) & group['trad_alarm']]
        pabd_t = pabd_hits.min() if len(pabd_hits) else None
        trad_t = trad_hits.min() if len(trad_hits) else None
        if pabd_t is not None and trad_t is not None:
            lead = trad_t - pabd_t
            all_lead_times.append(lead)
            event_log.append((day, start, end, pabd_t, trad_t, lead))

all_lead_times = np.array(all_lead_times)
print(f"Total attack-onset events with both detectors firing: {len(all_lead_times)}")
print(f"Raw lead times (windows): {list(all_lead_times)}")
print(f"Mean lead time: {all_lead_times.mean():.2f} windows ({all_lead_times.mean()*60:.0f} sec)")

# bootstrap 95% CI on mean lead time
rng = np.random.default_rng(42)
boot_means = [rng.choice(all_lead_times, size=len(all_lead_times), replace=True).mean()
              for _ in range(2000)]
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}] windows ([{ci_low*60:.0f}, {ci_high*60:.0f}] sec)")

print("\nPer-event detail:")
for day, start, end, pt, tt, lead in event_log:
    print(f"  {day}: attack windows {start}-{end}, PABD@{pt}, Traditional@{tt}, lead={lead}")
