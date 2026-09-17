import pandas as pd

df = pd.read_csv('network_data_real.csv')
wed = df[df['source_day'] == 'wednesday'].reset_index(drop=True)

cols = ['label', 'attack_type', 'attack_flow_fraction']
print(wed.loc[0:20, cols].to_string())
print(f"\nTotal wednesday windows: {len(wed)}")
print(f"Label counts:\n{wed['label'].value_counts()}")
