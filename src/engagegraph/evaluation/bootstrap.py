"""Paired bootstrap and multiple-comparison utilities.

Recommendation metrics are macro-averaged over users, so the user is the
natural resampling unit.  Paired bootstrap holds the evaluation users fixed and
resamples them with replacement, which respects the correlation between two
models scored on the same users.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n_boot: int
    confidence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_boot": float(self.n_boot),
            "confidence": self.confidence,
        }


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for i in range(n_boot):
        means[i] = values[rng.integers(0, n, n)].mean()
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        mean=float(values.mean()),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
        n_boot=n_boot,
        confidence=confidence,
    )


@dataclass(frozen=True)
class PairedResult:
    baseline: str
    candidate: str
    metric: str
    mean_baseline: float
    mean_candidate: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_users: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "metric": self.metric,
            "mean_baseline": self.mean_baseline,
            "mean_candidate": self.mean_candidate,
            "difference": self.difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_users": self.n_users,
        }


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    metric: str = "ndcg@10",
    n_boot: int = 1000,
    seed: int = 42,
) -> PairedResult:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError("paired arrays must have the same length")
    if baseline.size == 0:
        raise ValueError("empty metric arrays")
    diff = candidate - baseline
    rng = np.random.default_rng(seed)
    n = diff.size
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()
    observed = float(diff.mean())
    # Two-sided bootstrap p-value with the standard +1 correction.
    above = float(np.sum(boot >= 0.0))
    below = float(np.sum(boot <= 0.0))
    p_value = 2.0 * min((above + 1.0) / (n_boot + 1.0), (below + 1.0) / (n_boot + 1.0))
    p_value = min(1.0, p_value)
    alpha = 0.05
    return PairedResult(
        baseline=baseline_name,
        candidate=candidate_name,
        metric=metric,
        mean_baseline=float(baseline.mean()),
        mean_candidate=float(candidate.mean()),
        difference=observed,
        ci_low=float(np.quantile(boot, alpha / 2)),
        ci_high=float(np.quantile(boot, 1 - alpha / 2)),
        p_value=p_value,
        n_users=int(n),
    )


def wilcoxon_p(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Signed-rank test as a distribution-free sanity check."""

    diff = np.asarray(candidate) - np.asarray(baseline)
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:  # pragma: no cover - degenerate input
        return 1.0


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Family-wise error control across model comparisons."""

    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        value = min(1.0, (m - rank + 1) * p)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def compare_results(
    results: Mapping[str, object],
    *,
    baseline: str,
    metric: str = "ndcg@10",
    n_boot: int = 1000,
    seed: int = 42,
) -> list[PairedResult]:
    """Compare every result against ``baseline`` on paired users.

    All results must have been evaluated on the same dataset fingerprint and
    contain the same per-user array for ``metric``.
    """

    if baseline not in results:
        raise KeyError(f"baseline {baseline!r} not in results")
    reference = results[baseline]
    ref_values = reference.per_user[metric]  # type: ignore[attr-defined]
    comparisons: list[PairedResult] = []
    for name, result in results.items():
        if name == baseline:
            continue
        if result.dataset_fingerprint != reference.dataset_fingerprint:  # type: ignore[attr-defined]
            raise ValueError("cannot compare results from different datasets")
        values = result.per_user[metric]  # type: ignore[attr-defined]
        comparisons.append(
            paired_bootstrap(
                ref_values,
                values,
                baseline_name=baseline,
                candidate_name=name,
                metric=metric,
                n_boot=n_boot,
                seed=seed,
            )
        )
    return comparisons
