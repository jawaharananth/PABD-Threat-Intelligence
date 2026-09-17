import pandas as pd

FEATURES = ['port_entropy', 'connection_frequency', 'packet_size_dist',
            'protocol_ratio', 'failed_auth_rate']
K_MULT = 0.25
H_MULT = 3.0
N_CONJUNCTION = 3

df = pd.read_csv('network_data_real.csv')
wed = df[df['source_day'] == 'wednesday'].reset_index(drop=True)

baseline = wed[wed['label'] == 0]
means = baseline[FEATURES].mean()
stds = baseline[FEATURES].std()
k = K_MULT * stds
h = H_MULT * stds

cusum = pd.DataFrame(0.0, index=wed.index, columns=FEATURES)
for f in FEATURES:
    s = 0.0
    vals = []
    for x in wed[f]:
        s = max(0.0, s + (x - means[f] - k[f]))
        vals.append(s)
    cusum[f] = vals

n_over_h = (cusum > h).sum(axis=1)
wed['n_metrics_over_h'] = n_over_h

print(wed.loc[300:390, ['label', 'attack_type', 'attack_flow_fraction', 'n_metrics_over_h']].to_string())
