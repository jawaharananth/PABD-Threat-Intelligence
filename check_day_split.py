import pandas as pd

df = pd.read_csv('network_data_real.csv')
print(df['source_day'].value_counts())
print(f"\nTotal windows: {len(df)}")
