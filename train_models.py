import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load data
print("Loading data...")
df = pd.read_csv('network_data.csv')

FEATURES = [
    'connection_frequency',
    'port_entropy', 
    'failed_auth_rate',
    'packet_size_dist',
    'protocol_ratio',
    'burst_activity'
]

# ─── SCALER ───────────────────────────────────────────
print("Training scaler...")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved.")

# ─── ISOLATION FOREST ─────────────────────────────────
print("Training Isolation Forest...")
normal_data = df[df['label'] == 0][FEATURES]
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
iso_forest.fit(normal_data)
joblib.dump(iso_forest, 'isolation_forest.pkl')
print("Isolation Forest saved.")

# ─── RANDOM FOREST ────────────────────────────────────
print("Training Random Forest classifier...")
X = df[FEATURES]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    class_weight='balanced'
)
rf_classifier.fit(X_train_scaled, y_train)
joblib.dump(rf_classifier, 'random_forest.pkl')

# Evaluation
y_pred = rf_classifier.predict(X_test_scaled)
print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Normal', 'Reconnaissance', 'Brute Force', 'Lateral Movement', 'DoS Preparation']
))

# Feature importance
importances = rf_classifier.feature_importances_
feature_importance_dict = dict(zip(FEATURES, importances))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importances:")
for feat, imp in sorted_features:
    print(f"  {feat}: {imp:.4f}")

joblib.dump(feature_importance_dict, 'feature_importance.pkl')
print("\nRandom Forest saved.")

# ─── LSTM ─────────────────────────────────────────────
print("\nTraining LSTM...")
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_generator import generate_pre_attack_sequence

# Generate sequences for LSTM training
def create_lstm_dataset(sequence_length=10):
    X_seq = []
    y_seq = []
    
    attack_types = ['Reconnaissance', 'Brute Force', 'Lateral Movement']
    
    for attack_type in attack_types:
        for _ in range(100):
            seq = generate_pre_attack_sequence(attack_type, n_steps=50)
            seq_features = seq[FEATURES].values
            seq_features_scaled = scaler.transform(seq_features)
            
            for i in range(len(seq_features_scaled) - sequence_length):
                X_seq.append(seq_features_scaled[i:i+sequence_length])
                remaining = (len(seq_features_scaled) - i - sequence_length) / len(seq_features_scaled)
                y_seq.append(remaining)
    
    return np.array(X_seq), np.array(y_seq)

print("Generating LSTM training sequences...")
X_lstm, y_lstm = create_lstm_dataset(sequence_length=10)
print(f"LSTM dataset shape: {X_lstm.shape}")

X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=42
)

# Build LSTM model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(10, 6)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nLSTM Model Summary:")
lstm_model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\nTraining LSTM model...")
history = lstm_model.fit(
    X_lstm_train, y_lstm_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_lstm_test, y_lstm_test),
    callbacks=[early_stopping],
    verbose=1
)

lstm_model.save('lstm_model.keras')
print("LSTM model saved.")

# ─── BASELINE STATS ───────────────────────────────────
print("\nSaving baseline statistics...")
normal_df = df[df['label'] == 0][FEATURES]
baseline_stats = {
    'mean': normal_df.mean().to_dict(),
    'std': normal_df.std().to_dict()
}
joblib.dump(baseline_stats, 'baseline_stats.pkl')
print("Baseline stats saved.")

print("\n✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print("Files created:")
for f in ['scaler.pkl', 'isolation_forest.pkl', 'random_forest.pkl', 'lstm_model.keras', 'baseline_stats.pkl', 'feature_importance.pkl']:
    if os.path.exists(f):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} MISSING")
        