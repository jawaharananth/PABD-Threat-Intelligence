import pandas as pd

df = pd.read_csv('network_data_real.csv')
wed = df[df['source_day'] == 'wednesday'].reset_index(drop=True)

is_attack = (wed['label'] != 0).astype(int)
# find runs of consecutive attack windows
wed['run_id'] = (is_attack != is_attack.shift()).cumsum()
runs = wed[is_attack == 1].groupby('run_id').agg(
    start_idx=('label', lambda x: x.index.min()),
    end_idx=('label', lambda x: x.index.max()),
    length=('label', 'count')
)
runs = runs.sort_values('length', ascending=False)
print(runs.head(10))
print(f"\nLongest consecutive attack run: {runs['length'].max()} windows")
