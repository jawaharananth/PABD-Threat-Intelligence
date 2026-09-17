import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate']
K_MULT = 0.25
H_MULT = 3.0
N_CONJUNCTION = 3
RUN_START, RUN_END = 376, 380
PAD = 40  # windows of context before/after

df = pd.read_csv('network_data_real.csv')
wed = df[df['source_day'] == 'wednesday'].reset_index(drop=True)

baseline = wed[wed['label'] == 0]
means = baseline[FEATURES].mean()
stds = baseline[FEATURES].std()
k = K_MULT * stds
h = H_MULT * stds

# run CUSUM over the FULL day so accumulation carries in correctly, then slice window for plotting
cusum = pd.DataFrame(0.0, index=wed.index, columns=FEATURES)
for f in FEATURES:
    s = 0.0
    vals = []
    for x in wed[f]:
        s = max(0.0, s + (x - means[f] - k[f]))
        vals.append(s)
    cusum[f] = vals

alarm_mask = (cusum > h).sum(axis=1) >= N_CONJUNCTION
wed['pabd_alarm'] = alarm_mask
wed['trad_ids_alarm'] = wed['attack_flow_fraction'] > 0.5

lo, hi = max(0, RUN_START - PAD), min(len(wed) - 1, RUN_END + PAD)
window = wed.loc[lo:hi]
cusum_window = cusum.loc[lo:hi]

pabd_alarms_in_window = window.index[window['pabd_alarm']]
trad_alarms_in_window = window.index[window['trad_ids_alarm']]
pabd_first = pabd_alarms_in_window.min() if len(pabd_alarms_in_window) else None
trad_first = trad_alarms_in_window.min() if len(trad_alarms_in_window) else None

print(f"Zoomed window: index {lo} to {hi} (run at {RUN_START}-{RUN_END})")
print(f"PABD first alarm in this window: {pabd_first}")
print(f"Traditional IDS first alarm in this window: {trad_first}")
if pabd_first is not None and trad_first is not None:
    lead = trad_first - pabd_first
    print(f"Lead time: {lead} windows ({lead * 60} seconds)")

fig, ax1 = plt.subplots(figsize=(12, 6))
cusum_sum = (cusum_window / h).sum(axis=1)
ax1.plot(window.index, cusum_sum, color='steelblue', label='CUSUM conjunction signal (sum of S/h)')
ax1.axhline(N_CONJUNCTION, color='steelblue', linestyle='--', alpha=0.5, label=f'Alarm threshold (N={N_CONJUNCTION})')
ax1.axvspan(RUN_START, RUN_END, color='gray', alpha=0.15, label='Actual attack windows (ground truth)')
ax1.set_xlabel('Window index (60s each)')
ax1.set_ylabel('CUSUM conjunction signal', color='steelblue')

ax2 = ax1.twinx()
ax2.plot(window.index, window['attack_flow_fraction'], color='darkorange', alpha=0.6, label='attack_flow_fraction')
ax2.axhline(0.5, color='darkorange', linestyle='--', alpha=0.5, label='Traditional IDS threshold (0.5)')
ax2.set_ylabel('attack_flow_fraction', color='darkorange')

if pabd_first is not None:
    ax1.axvline(pabd_first, color='green', linestyle='-', linewidth=2, label='PABD alarm')
if trad_first is not None:
    ax1.axvline(trad_first, color='red', linestyle='-', linewidth=2, label='Traditional IDS alarm')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

plt.title(f'PABD vs Traditional IDS — Wednesday DoS episode, windows {lo}-{hi}')
plt.tight_layout()
plt.savefig('wednesday_dos_case_study_zoomed.png', dpi=150)
print("Saved: wednesday_dos_case_study_zoomed.png")
