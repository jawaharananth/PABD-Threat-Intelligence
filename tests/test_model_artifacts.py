"""
Sanity checks that the saved model artifacts actually load and produce
sane, correctly-shaped output. These don't re-validate accuracy (that's
what cv_eval_pooled.py / fusion_model.py's held-out numbers are for) -- they
catch the class of bug where a retrain silently breaks a downstream
consumer's expected input/output shape (e.g. the scaler/feature-count
mismatch bug fixed earlier between app.py and the 10-feature research
models).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import pytest

PABD_DIR = os.path.join(os.path.dirname(__file__), "..")


def _p(fname):
    return os.path.join(PABD_DIR, fname)


DEMO_FEATURES = [
    "connection_frequency", "port_entropy", "failed_auth_rate",
    "packet_size_dist", "protocol_ratio", "burst_activity",
]
RESEARCH_FEATURES = DEMO_FEATURES + [
    "syn_rate", "flow_duration_cv", "bytes_per_sec_mean", "down_up_ratio_mean",
]


@pytest.mark.skipif(not os.path.exists(_p("scaler_demo6.pkl")), reason="models not trained in this environment")
def test_demo6_scaler_matches_demo6_model_dimensions():
    scaler = joblib.load(_p("scaler_demo6.pkl"))
    rf = joblib.load(_p("random_forest_demo6.pkl"))
    row = pd.DataFrame([{f: 1.0 for f in DEMO_FEATURES}])
    Xs = scaler.transform(row)
    pred = rf.predict(Xs)
    assert pred.shape == (1,)


@pytest.mark.skipif(not os.path.exists(_p("scaler.pkl")), reason="models not trained in this environment")
def test_research_scaler_matches_research_model_dimensions():
    scaler = joblib.load(_p("scaler.pkl"))
    rf = joblib.load(_p("random_forest.pkl"))
    row = pd.DataFrame([{f: 1.0 for f in RESEARCH_FEATURES}])
    Xs = scaler.transform(row)
    pred = rf.predict(Xs)
    assert pred.shape == (1,)


@pytest.mark.skipif(not os.path.exists(_p("scaler.pkl")), reason="models not trained in this environment")
def test_demo6_and_research_scalers_are_not_accidentally_the_same_object():
    # Regression guard for the exact bug fixed earlier: app.py's live demo
    # was pointed at the 10-feature research scaler, which crashes on a
    # 6-feature input. If these two files ever get collapsed back into one
    # by an unwary edit, this test catches the dimension mismatch.
    scaler6 = joblib.load(_p("scaler_demo6.pkl"))
    scaler10 = joblib.load(_p("scaler.pkl"))
    assert scaler6.n_features_in_ == 6
    assert scaler10.n_features_in_ == 10


@pytest.mark.skipif(not os.path.exists(_p("isolation_forest_demo6.pkl")), reason="models not trained in this environment")
def test_demo6_isolation_forest_returns_finite_scores():
    scaler = joblib.load(_p("scaler_demo6.pkl"))
    iso = joblib.load(_p("isolation_forest_demo6.pkl"))
    row = pd.DataFrame([{f: 1.0 for f in DEMO_FEATURES}])
    score = iso.score_samples(row)
    assert np.isfinite(score).all()


@pytest.mark.skipif(not os.path.exists(_p("lstm_model_real_countdown.keras")), reason="LSTM not trained in this environment")
def test_countdown_lstm_output_is_bounded_zero_to_one():
    from tensorflow import keras
    model = keras.models.load_model(_p("lstm_model_real_countdown.keras"))
    scaler = joblib.load(_p("scaler_demo6.pkl"))
    dummy = np.tile(np.array([[10, 0.5, 0.1, 500, 0.9, 0.2]]), (10, 1))
    Xsc = scaler.transform(dummy).reshape(1, 10, 6)
    pr = model.predict(Xsc, verbose=0)[0][0]
    assert 0.0 <= pr <= 1.0
