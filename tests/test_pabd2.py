"""Unit tests for PABD 2.0's statistical and data machinery."""
import numpy as np
import pandas as pd
import pytest

from pabd2.conformal import Conformal, drift_alarm
from pabd2.features import SEQ_LEN, Scaler, sequence_index
from pabd2.killchain import T, filter_host, stage_evidence
from pabd2.schema import STAGES, canon, port_cat


def test_conformal_false_alarm_rate_is_guaranteed():
    rng = np.random.default_rng(1)
    cal, test = rng.normal(size=5000), rng.normal(size=200_000)
    p = Conformal().fit(cal).pvalue(test)
    for a in (0.001, 0.01, 0.05):
        assert (p <= a).mean() <= a + 3 * np.sqrt(a * (1 - a) / len(test)) + 1 / 5001


def test_conformal_pvalues_uniform_and_attack_small():
    rng = np.random.default_rng(2)
    c = Conformal().fit(rng.normal(size=4000))
    p = c.pvalue(rng.normal(size=50_000))
    assert abs(p.mean() - 0.5) < 0.01
    assert (c.pvalue(rng.normal(6, 1, size=1000)) <= 0.01).mean() > 0.99


def test_martingale_quiet_on_exchangeable_and_fires_on_shift():
    rng = np.random.default_rng(3)
    c = Conformal().fit(rng.normal(size=4000))
    false_alarms = sum(drift_alarm(c.pvalue(rng.normal(size=3000)), delta=0.01)[0] >= 0 for _ in range(40))
    assert false_alarms <= 3          # Ville: <= 1% per stream, generous slack for 40 trials
    k, _ = drift_alarm(c.pvalue(np.r_[rng.normal(size=1000), rng.normal(0.8, 1, size=2000)]), delta=0.01)
    assert 1000 <= k < 1600           # fires soon after the change, never before it


def test_killchain_rows_are_distributions_and_filter_escalates():
    assert np.allclose(T.sum(1), 1)
    fams = ["Benign", "Port Scan", "Brute Force", "Botnet", "DoS"]
    P = np.zeros((30, len(fams))); P[:10, 0] = 1; P[10:20, 1] = 1; P[20:, 2] = 1
    b = filter_host(stage_evidence(P, fams))
    assert STAGES[b[9].argmax()] == "Benign"
    assert STAGES[b[19].argmax()] == "Reconnaissance"
    assert STAGES[b[29].argmax()] == "Initial Access"


def test_labels_map_across_years_and_attempted_is_flagged():
    assert canon("DoS Hulk")[:2] == ("DoS", "Impact")
    assert canon("DoS-Hulk")[:2] == ("DoS", "Impact")
    assert canon("Infiltration - Portscan")[0] == "Internal Scan (post-compromise)"
    fam, stage, attempted = canon("Web Attack - XSS - Attempted")
    assert (fam, attempted) == ("Web Attack", True)


def test_port_buckets_hide_raw_ports():
    assert list(port_cat(np.array([443, 22, 445, 53, 50000, 1000]))) == ["web", "ssh", "smb", "dns", "high", "low_other"]


def test_sequences_are_causal_and_per_host():
    df = pd.DataFrame({"Src IP": ["a"] * 5 + ["b"] * 3})
    s = sequence_index(df, np.array([4, 6]))
    assert s.shape == (2, SEQ_LEN)
    assert list(s[0][-5:]) == [0, 1, 2, 3, 4] and (s[0][:-5] == -1).all()
    assert list(s[1][-2:]) == [5, 6] and (s[1][:-2] == -1).all()   # never crosses into host "a"


def test_scaler_roundtrip_used_by_adversarial():
    from pabd2.adversarial import raw
    X = np.abs(np.random.default_rng(4).lognormal(3, 2, size=(2000, 5)))
    sc = Scaler().fit(X)
    Z = sc.transform(X)
    ok = (np.abs(Z) < 7.9).all(1)
    assert np.allclose(raw(sc, Z)[ok], X[ok], rtol=1e-3)
