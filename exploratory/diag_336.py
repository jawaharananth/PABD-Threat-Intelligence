import pandas as pd

df = pd.read_csv('network_data_real.csv')
wed = df[df['source_day'] == 'wednesday'].reset_index(drop=True)

cols = ['label', 'attack_type', 'attack_flow_fraction']
print(wed.loc[330:390, cols].to_string())
