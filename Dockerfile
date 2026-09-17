FROM python:3.11-slim

WORKDIR /app

# System deps needed by tensorflow/scipy wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model artifacts (scaler_demo6.pkl, isolation_forest_demo6.pkl,
# random_forest_demo6.pkl, lstm_model_real_countdown.keras, baseline_stats.pkl)
# must already exist in the build context -- this image serves the live
# dashboard, it does not retrain models on startup. Run train_models.py and
# train_lstm_countdown_real.py locally first if these are missing.

EXPOSE 8050

CMD ["python", "app.py"]
