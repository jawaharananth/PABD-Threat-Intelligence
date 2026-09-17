"""
Tests the canonical CUSUM implementation in cusum_core.py, shared by
cusum_adaptive.py and fusion_model.py. Importing the real function (not a
local reimplementation) means these tests actually catch a regression in
the shared module, not just in a copy that could drift from it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from cusum_core import cusum


def test_cusum_stays_quiet_on_pure_baseline_noise():
    rng = np.random.default_rng(0)
    series = rng.normal(loc=10.0, scale=1.0, size=200)
    flags = cusum(series, mean=10.0, std=1.0)
    # Some noise crossings are expected (this is exactly the false-positive
    # problem documented in the paper), but it must not fire on almost every
    # window -- that would mean the k-slack term isn't doing anything.
    assert flags.mean() < 0.5


def test_cusum_fires_on_a_sustained_mean_shift():
    # 100 baseline windows, then a sustained +5-std shift (a real attack
    # onset should look like this: persistent drift, not a single spike).
    baseline = np.full(100, 10.0)
    shifted = np.full(100, 15.0)
    series = np.concatenate([baseline, shifted])
    flags = cusum(series, mean=10.0, std=1.0)
    # Must fire at some point during the shifted segment.
    assert flags[100:].any()


def test_cusum_does_not_fire_on_a_single_isolated_spike():
    # One-off spike should not trip a change-point detector meant to catch
    # *sustained* drift -- CUSUM's whole design point vs a simple threshold.
    series = np.full(50, 10.0)
    series[25] = 100.0  # one huge one-off spike
    flags = cusum(series, mean=10.0, std=1.0, k_mult=0.5, h_mult=5.0)
    # It's fine if the single spike window itself trips (the instantaneous
    # value is large), but it must NOT stay tripped for many windows after --
    # the cumulative sum should decay back down once traffic returns to
    # baseline, since s = max(0, s + ...) pulls it back below the mean+k line.
    assert not flags[30:].any(), "CUSUM stayed tripped long after an isolated spike decayed"


def test_higher_h_threshold_never_fires_more_than_lower_h():
    rng = np.random.default_rng(1)
    series = rng.normal(loc=10.0, scale=2.0, size=300)
    low_h = cusum(series, mean=10.0, std=2.0, h_mult=1.5)
    high_h = cusum(series, mean=10.0, std=2.0, h_mult=4.0)
    assert high_h.sum() <= low_h.sum()
