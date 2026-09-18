# Exploratory scripts (superseded)

Early tuning/diagnostic scripts from before the CUSUM reset fix and the
6-day data expansion. Kept as process history, but **do not treat their
output as current results** — they use:

- The old 5-6 feature set (missing `syn_rate`, `flow_duration_cv`,
  `bytes_per_sec_mean`, `down_up_ratio_mean`)
- Hand-picked CUSUM params (K=0.25, H=3.0) instead of the grid-searched ones
- The pre-fix CUSUM implementation (no reset-after-spike logic — see the
  "stuck alarm" bug documented in `cusum_core.py` and `tests/test_cusum_logic.py`)
- Only 3 of the 8 available CICIDS2017 days

The current, authoritative pipeline is `cusum_adaptive.py` + `cusum_core.py`
+ `fusion_model.py`, evaluated across all 6 attack-containing days plus the
Monday/Infiltration checks — see the main `README.md`'s Results section.

Also here (moved in the cleanup pass): `data_generator.py` (synthetic
traffic generator, no longer used by any pipeline step), `train_lstm_real.py`
(predicted next-window attack fraction; superseded by
`train_lstm_countdown_real.py`), `check_day_split.py`, `find_runs.py`.
Stale artifacts they produced (`lstm_model.keras`, `lstm_model_real.keras`,
`network_data.csv`, the `*_30s/_60s/_120s` window variants with the old
6-column schema) were removed from the working tree; they remain in git
history if ever needed.
