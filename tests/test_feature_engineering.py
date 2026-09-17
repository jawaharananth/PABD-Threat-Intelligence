"""
Tests real_data_loader.py's compute_window() -- the feature-engineering
function that turns raw CICIDS flow rows into the window-level features
everything downstream (CUSUM, RF, fusion) depends on. Uses small constructed
DataFrames with known expected values, not the real multi-hundred-MB CSVs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest

from real_data_loader import compute_window, _entropy


def test_entropy_is_zero_for_a_single_uniform_value():
    counts = pd.Series([10])  # everything is the same value -> no uncertainty
    assert _entropy(counts) == pytest.approx(0.0)


def test_entropy_is_positive_for_mixed_values():
    counts = pd.Series([5, 5])  # two equally-likely values -> max entropy for n=2
    assert _entropy(counts) == pytest.approx(1.0)  # log2(2) = 1


def _make_window_df(n=10):
    return pd.DataFrame({
        "Destination Port": [80, 80, 443, 443, 22, 22, 21, 21, 8080, 8080][:n],
        "Flow Duration": [1000, 1200, 900, 1100, 500, 600, 400, 450, 2000, 2100][:n],
        "Total Fwd Packets": [3, 3, 3, 3, 2, 2, 2, 2, 5, 5][:n],
        "Average Packet Size": [500.0] * n,
        "Protocol": [6, 6, 6, 6, 17, 17, 6, 6, 6, 6][:n],
        "SYN Flag Count": [1, 0, 1, 0, 0, 0, 1, 1, 0, 0][:n],
        "Flow Bytes/s": [1000.0, 2000.0, np.inf, 1500.0, 1200.0, 900.0, 800.0, 1100.0, 1300.0, 1400.0][:n],
        "Down/Up Ratio": [1.0, 0.5, 2.0, 1.5, 0.8, 1.1, 0.9, 1.0, 1.2, 0.7][:n],
        "Label": ["BENIGN"] * n,
        "parsed_time": pd.date_range("2017-07-04 08:00", periods=n, freq="10s"),
    })


def test_compute_window_returns_none_for_empty_input():
    empty = _make_window_df(0)
    assert compute_window(empty) is None


def test_compute_window_produces_all_expected_feature_keys():
    result = compute_window(_make_window_df())
    expected_keys = {
        "connection_frequency", "port_entropy", "failed_auth_rate",
        "packet_size_dist", "protocol_ratio", "burst_activity",
        "syn_rate", "flow_duration_cv", "bytes_per_sec_mean",
        "down_up_ratio_mean", "window_label", "attack_flow_fraction",
        "window_start",
    }
    assert expected_keys.issubset(result.keys())


def test_compute_window_connection_frequency_is_row_count():
    df = _make_window_df(n=10)
    result = compute_window(df)
    assert result["connection_frequency"] == 10


def test_compute_window_syn_rate_is_mean_of_syn_flags():
    df = _make_window_df(n=10)
    result = compute_window(df)
    expected = df["SYN Flag Count"].mean()
    assert result["syn_rate"] == pytest.approx(expected)


def test_compute_window_drops_inf_from_bytes_per_sec():
    # One row has Flow Bytes/s = inf -- must be excluded from the mean, not
    # poison the whole window's average into inf.
    df = _make_window_df(n=10)
    result = compute_window(df)
    assert np.isfinite(result["bytes_per_sec_mean"])


def test_compute_window_all_benign_gives_zero_attack_fraction():
    df = _make_window_df(n=10)
    result = compute_window(df)
    assert result["window_label"] == "BENIGN"
    assert result["attack_flow_fraction"] == 0.0


def test_compute_window_majority_attack_label_wins():
    df = _make_window_df(n=10)
    df.loc[0:6, "Label"] = "PortScan"  # 7 of 10 rows are PortScan
    df.loc[7:, "Label"] = "BENIGN"
    result = compute_window(df)
    assert result["window_label"] == "PortScan"
    assert result["attack_flow_fraction"] == pytest.approx(0.7)
