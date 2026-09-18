import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
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

scaler = joblib.load("scaler.pkl")

def build_sequences(day):
    df = pd.read_csv(f"windows_{day}.csv")
    df["window_start"] = pd.to_datetime(df["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)

    X_scaled = scaler.transform(df[FEATURES].values)
    targets = df["attack_flow_fraction"].values

    X_seq, y_seq = [], []
    for i in range(len(df) - SEQUENCE_LENGTH):
        X_seq.append(X_scaled[i:i+SEQUENCE_LENGTH])
        y_seq.append(targets[i + SEQUENCE_LENGTH])  # next window, real forecast target

    return np.array(X_seq), np.array(y_seq)

print("Building real sequences from chronological windows...")
X_list, y_list = [], []
for day in ["tuesday", "wednesday", "friday"]:
    Xd, yd = build_sequences(day)
    print(f"  {day}: {Xd.shape[0]} sequences")
    X_list.append(Xd)
    y_list.append(yd)

X_lstm = np.concatenate(X_list, axis=0)
y_lstm = np.concatenate(y_list, axis=0)
print(f"Total LSTM dataset shape: {X_lstm.shape}, targets: {y_lstm.shape}")
print(f"Target (attack_flow_fraction) stats: mean={y_lstm.mean():.4f}, "
      f"std={y_lstm.std():.4f}, max={y_lstm.max():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_model.summary()

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining LSTM on real data...")
history = lstm_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1,
)

val_mae = min(history.history["val_mae"])
print(f"\nBest validation MAE: {val_mae:.4f} "
      f"(target range is 0-1, this is fraction-of-window-that-is-attack-traffic)")

lstm_model.save("lstm_model_real.keras")
print("Saved lstm_model_real.keras")
