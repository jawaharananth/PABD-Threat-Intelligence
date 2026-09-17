"""
Real-data countdown LSTM.

Fixes the methodology gap where the deployed countdown model (lstm_model.keras)
was trained purely on data_generator.py's synthetic generate_pre_attack_sequence()
sequences, while every other model (Isolation Forest, Random Forest) trains on
real CICIDS2017 flow data.

Target semantics matched to app.py's usage (S['countdown_max'] = pr * 60 * 3):
  pr -> 1.0  =  attack onset far away / not imminent
  pr -> 0.0  =  attack onset imminent (next window or two)

Built from real chronological windows_{day}.csv timelines: for each day, find
real attack-run onsets (first window of each contiguous attack run, same
definition used in find_runs.py / aggregate_lead_time.py), then label every
pre-attack window within a lookback horizon by its true distance-to-onset,
normalized and capped. Windows further than HORIZON steps from any attack
(or with no upcoming attack in range) are excluded from training rather than
force-labeled, to avoid inventing a signal that isn't there.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

FEATURES = [
    "connection_frequency",
    "port_entropy",
    "failed_auth_rate",
    "packet_size_dist",
    "protocol_ratio",
    "burst_activity",
]

SEQUENCE_LENGTH = 10
HORIZON = 20  # windows; only label windows within this lookback of a real attack onset

# scaler.pkl is now the 10-feature research scaler (see train_models.py) -- the
# countdown LSTM stays on the original 6-feature schema to match app.py's live
# demo simulator, so it needs scaler_demo6.pkl specifically.
scaler = joblib.load("scaler_demo6.pkl")


def attack_run_starts(df):
    is_attack = (df["window_label"] != "BENIGN").astype(int)
    run_id = (is_attack != is_attack.shift()).cumsum()
    starts = []
    for rid, run in df[is_attack == 1].groupby(run_id):
        starts.append(run.index.min())
    return sorted(starts)


def build_countdown_sequences(day):
    df = pd.read_csv(f"windows_{day}.csv")
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)

    starts = attack_run_starts(df)
    X_scaled = scaler.transform(df[FEATURES].values)

    X_seq, y_seq = [], []
    for i in range(len(df) - SEQUENCE_LENGTH):
        anchor = i + SEQUENCE_LENGTH  # the "current" window this sequence predicts from
        # distance to the next attack onset at/after the anchor
        future_starts = [s for s in starts if s >= anchor]
        if not future_starts:
            continue
        dist = future_starts[0] - anchor
        if dist > HORIZON:
            continue
        target = dist / HORIZON  # 0 = imminent, 1 = at horizon edge
        X_seq.append(X_scaled[i:anchor])
        y_seq.append(target)

    return np.array(X_seq), np.array(y_seq)


print("Building real countdown sequences from chronological windows (real attack onsets only)...")
X_list, y_list, day_list = [], [], []
for day in ["tuesday", "wednesday", "friday"]:
    Xd, yd = build_countdown_sequences(day)
    print(f"  {day}: {Xd.shape[0]} labeled pre-attack sequences (within {HORIZON}-window horizon)")
    if len(Xd) == 0:
        continue
    X_list.append(Xd)
    y_list.append(yd)
    day_list.extend([day] * len(Xd))

if not X_list:
    raise SystemExit("No pre-attack sequences found within horizon on any day -- "
                      "check window files / HORIZON setting before training.")

X_lstm = np.concatenate(X_list, axis=0)
y_lstm = np.concatenate(y_list, axis=0)
print(f"\nTotal real countdown dataset: {X_lstm.shape}, targets: {y_lstm.shape}")
print(f"Target stats: mean={y_lstm.mean():.4f}, std={y_lstm.std():.4f}")

if len(X_lstm) < 40:
    print(f"\nWARNING: only {len(X_lstm)} labeled sequences available. This is a small-sample "
          f"caveat that must be stated explicitly in any paper/report -- do not claim tight "
          f"countdown precision without reporting this n.")

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

lstm_model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
    Dropout(0.2),
    LSTM(16, return_sequences=False),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid"),
])

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

print("\nTraining real countdown LSTM...")
history = lstm_model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1,
)

WINDOW_SECONDS = 120  # confirmed from windows_*.csv timestamp deltas -- do not assume 60s

y_pred = lstm_model.predict(X_test, verbose=0).flatten()
mae_norm = mean_absolute_error(y_test, y_pred)
mae_seconds = mae_norm * HORIZON * WINDOW_SECONDS

print(f"\nHeld-out MAE (normalized 0-1): {mae_norm:.4f}")
print(f"Held-out MAE in real seconds (HORIZON={HORIZON} windows @ {WINDOW_SECONDS}s each): ~{mae_seconds:.0f} sec")
print(f"n_test = {len(y_test)} -- report this sample size alongside the MAE, do not state "
      f"MAE alone as if it were computed on a large held-out set.")

lstm_model.save("lstm_model_real_countdown.keras")
print("\nSaved lstm_model_real_countdown.keras")
