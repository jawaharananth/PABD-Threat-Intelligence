import pandas as pd

df = pd.read_csv('network_data_real.csv')
tue = df[df['source_day'] == 'tuesday'].reset_index(drop=True)
print(tue['label'].value_counts())
print(f"\nattack_flow_fraction stats for attack windows (label!=0):")
print(tue[tue['label'] != 0]['attack_flow_fraction'].describe())
print(f"\nHow many Tuesday attack windows have attack_flow_fraction > 0.5?")
print((tue[tue['label'] != 0]['attack_flow_fraction'] > 0.5).sum(), "out of", (tue['label'] != 0).sum())
