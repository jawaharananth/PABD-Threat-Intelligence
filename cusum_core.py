"""
Single canonical CUSUM implementation, shared by cusum_adaptive.py,
fusion_model.py, and the test suite. Previously each script had its own
copy-pasted cusum() -- exactly the kind of duplication that let the ANAMES
label-id bug happen elsewhere in this project. One copy here instead.

Includes a reset mechanism found necessary by tests/test_cusum_logic.py:
without it, a single large one-off spike leaves the cumulative sum stuck
above threshold for ~(spike_magnitude / k) windows afterward, since s only
decays by the fixed slack term k per step, not proportional to how far it
overshot. A spike of +90 std with k=0.5*std takes ~180 windows (6+ hours at
120s/window) to decay back down on its own. That is very likely part of why
the false-positive rate measured on Monday's fully clean day (73% at the
"best F1" operating point) was so high -- one early fluctuation could stay
tripped for most of the day.

The reset: once a feature has stayed within `reset_tolerance` standard
deviations of baseline for `reset_window` consecutive windows, force the
cumulative sum back to 0. This is a standard bounded/reset-CUSUM pattern
(a real drift keeps the series away from baseline and never satisfies the
reset condition, so genuine attacks are still caught) -- it specifically
targets the "stuck after one spike" failure mode, not sensitivity to
sustained drift.
"""
import numpy as np


class CusumTracker:
    """Stateful per-metric CUSUM tracker. Exists so callers whose baseline
    (mean/std) changes every step -- e.g. cusum_adaptive.py's rolling
    baseline -- can reuse the exact same update rule as the fixed-baseline
    batch case, instead of a second hand-copied formula."""

    def __init__(self):
        self.s = 0.0
        self.consecutive_normal = 0

    def update(self, x, mean, std, k_mult=0.5, h_mult=2.0, reset_tolerance=0.5, reset_window=5):
        k, h = k_mult * std, h_mult * std
        if abs(x - mean) <= reset_tolerance * std:
            self.consecutive_normal += 1
        else:
            self.consecutive_normal = 0
        if self.consecutive_normal >= reset_window:
            self.s = 0.0
        self.s = max(0.0, self.s + (x - mean - k))
        return self.s > h


def cusum(series, mean, std, k_mult=0.5, h_mult=2.0, reset_tolerance=0.5, reset_window=5):
    """Batch version for a fixed (mean, std) baseline over the whole series."""
    tracker = CusumTracker()
    return np.array([
        tracker.update(x, mean, std, k_mult, h_mult, reset_tolerance, reset_window)
        for x in series
    ])


def cusum_breach_matrix(df, features, mean, std, **cusum_kwargs):
    """Per-window breach matrix (n_windows, n_features) of bool crossings,
    one column per feature, using the shared cusum() above."""
    crossed = np.zeros((len(df), len(features)), dtype=bool)
    for j, f in enumerate(features):
        crossed[:, j] = cusum(df[f].values, mean[f], std[f], **cusum_kwargs)
    return crossed
