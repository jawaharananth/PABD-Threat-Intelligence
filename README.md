# PABD Threat Intelligence System

A real-time network threat detection dashboard built using machine learning. This was my first major ML project — I wanted to build something in cybersecurity that goes beyond just detecting attacks after they happen.

## The Problem

Most intrusion detection systems only alert you once an attack is already running. By that point there's not much you can do. I wanted to see if I could detect the early warning signs — the weird behaviour that happens before the attack actually starts.

## What I Built

A live dashboard that watches network traffic, spots unusual patterns early, figures out what kind of attack it might be, and estimates how much time you have before it fully executes.

## How It Works
 
I used three models working together:

Isolation Forest — detects anomalous packets in real time
Random Forest — classifies the attack type once drift is detected
LSTM — predicts how much time is left before the attack hits

On top of that I used a CUSUM algorithm to track 6 network metrics at once and trigger an alert when 3 or more of them start drifting from normal.
The attack types it can detect:

Reconnaissance
Brute Force
Lateral Movement
DoS Preparation


## Tech Used

Python
Dash + Plotly (dashboard)
Scikit-learn (Isolation Forest, Random Forest)
TensorFlow / Keras (LSTM)
Pandas, NumPy


## Project Files

PABD_System/
├── app.py                 
├── train_models.py        
├── data_generator.py      
├── network_data.csv       
├── scaler.pkl
├── isolation_forest.pkl
├── random_forest.pkl
├── lstm_model.keras
└── baseline_stats.pkl

## How to Run

bashpip install dash plotly scikit-learn tensorflow joblib pandas numpy
bashpython train_models.py
bashpython app.py
Then open localhost:8050 in your browser. Hit SIMULATE ATTACK to see it in action.

## Results

Fires an alert 20+ steps before a traditional IDS would
Random Forest got 95%+ accuracy across all 5 classes
LSTM countdown accurate within ±90 seconds


## Future improvements

Connect to actual live network traffic
Push alerts to Slack or email
Deploy it online so it runs 24/7