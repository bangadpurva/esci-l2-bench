"""Paired bootstrap over users, with Holm correction across model comparisons.

Decision rule (pre-registered): a model beats the baseline on NDCG@10 only if
its Holm-adjusted paired-bootstrap test rejects at alpha = 0.05 AND the mean
difference is positive. We report the percentile CI at the Holm-adjusted level
for each comparison, so "CI excludes zero" and "adjusted p < alpha" agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedResult:
    name: str
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float = float("nan")
    ci_level: float = 0.95
    significant: bool = False


def paired_bootstrap(model: np.ndarray, baseline: np.ndarray, n_resamples: int = 10_000,
                     seed: int = 0) -> tuple[float, np.ndarray]:
    """Return observed mean difference and the bootstrap distribution of it.

    Both arrays are per-user metric values aligned on the same user order.
    """
    model = np.asarray(model, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if model.shape != baseline.shape or model.ndim != 1:
        raise ValueError("expected two aligned 1-D per-user arrays")
    diff = model - baseline
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    return float(diff.mean()), diff[idx].mean(axis=1)


def bootstrap_p_value(boot: np.ndarray) -> float:
    """Two-sided p from the bootstrap distribution's mass on each side of zero."""
    left = (np.sum(boot <= 0) + 1) / (len(boot) + 1)
    right = (np.sum(boot >= 0) + 1) / (len(boot) + 1)
    return float(min(1.0, 2 * min(left, right)))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone, in the input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted.tolist()


def compare_to_baseline(per_model: dict[str, np.ndarray], baseline: np.ndarray,
                        alpha: float = 0.05, n_resamples: int = 10_000,
                        seed: int = 0) -> list[PairedResult]:
    names = list(per_model)
    raw = []
    for j, name in enumerate(names):
        mean, boot = paired_bootstrap(per_model[name], baseline, n_resamples, seed + j)
        raw.append((name, mean, boot, bootstrap_p_value(boot)))
    adjusted = holm([r[3] for r in raw])
    m = len(raw)
    order = np.argsort([r[3] for r in raw])
    # Holm's per-step alpha: alpha / (m - rank) for the rank-th smallest p.
    step_alpha = {int(i): alpha / (m - rank) for rank, i in enumerate(order)}

    results = []
    for i, (name, mean, boot, p) in enumerate(raw):
        a = step_alpha[i]
        lo, hi = np.quantile(boot, [a / 2, 1 - a / 2])
        results.append(PairedResult(
            name=name, mean_diff=mean, ci_low=float(lo), ci_high=float(hi),
            p_value=p, p_holm=adjusted[i], ci_level=1 - a,
            significant=bool(adjusted[i] < alpha and mean > 0),
        ))
    return results
