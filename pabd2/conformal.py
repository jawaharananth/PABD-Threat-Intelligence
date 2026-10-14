"""Calibrated alerting.

1. Split-conformal p-values (Vovk et al.; FIRCE 2026 for IDS). Scores of benign calibration
   units define what "normal" looks like. A new unit's p-value is the fraction of benign
   calibration scores at least as extreme. Alarm if p <= alpha. If test benign traffic is
   exchangeable with calibration benign traffic, the false-alarm rate is at most alpha -- a
   guarantee, not a tuned threshold. CUSUM's recall/false-alarm trade-off becomes a dial with a
   meaning: "1 false alarm per 100 benign host-windows".

2. Conformal test martingale (Vovk, Volkhonskiy et al.) for drift: under exchangeability,
   p-values are uniform; a power martingale M_t = prod eps * p_t^(eps-1) stays small. By Ville's
   inequality P(sup M_t >= 1/delta) <= delta, so crossing 1/delta is a drift alarm whose false-
   alarm probability is bounded by delta over the *whole* stream. This replaces CUSUM's
   hand-tuned k/h with a bound you can state.
"""
import numpy as np


class Conformal:
    def fit(self, benign_cal_scores):
        self.cal = np.sort(np.asarray(benign_cal_scores, dtype=np.float64)); return self

    def pvalue(self, scores, rng=None):
        """Smoothed conformal p-value (higher score = more anomalous)."""
        s = np.asarray(scores, dtype=np.float64); n = len(self.cal)
        gt = n - np.searchsorted(self.cal, s, side="right")
        eq = np.searchsorted(self.cal, s, side="right") - np.searchsorted(self.cal, s, side="left")
        u = (rng or np.random.default_rng(0)).random(len(s))
        return (gt + u * (eq + 1)) / (n + 1)


def power_martingale(p, eps=0.92, cap=1e12):
    """Log of a mixture-free power martingale; resets are left to the caller."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return np.cumsum(np.log(eps) + (eps - 1) * np.log(p))


def drift_alarm(p, delta=0.01, eps=0.92):
    """Index of first time the martingale crosses 1/delta, or -1. Uses the Shiryaev-Roberts-style
    restart: the running log-martingale minus its running minimum, so old benign history does not
    bury a new change (still a valid alarm rule; the bound applies per restart window)."""
    lm = power_martingale(p, eps)
    stat = lm - np.minimum.accumulate(np.minimum(lm, 0))
    hit = np.flatnonzero(stat >= np.log(1 / delta))
    return (int(hit[0]) if len(hit) else -1), stat
