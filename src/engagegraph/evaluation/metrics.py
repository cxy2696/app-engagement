"""Ranking metrics for full-catalogue next-app recommendation.

All metrics are computed per user and then macro-averaged, so a user with 60
future apps does not dominate a user with one.  Relevance is binary (the user
adopted the app in February); graded engagement is deliberately not used for
the primary benchmark because it makes cross-model comparisons sensitive to the
weight transform.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import BaseRecommender

EPS = 1e-12


def _ideal_dcg(n_relevant: int, k: int) -> float:
    n = min(n_relevant, k)
    if n <= 0:
        return 0.0
    return float(np.sum(1.0 / np.log2(np.arange(2, n + 2))))


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    if k == scores.shape[1]:
        return np.argsort(-scores, axis=1)
    candidate = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidate, axis=1)
    order = np.argsort(-candidate_scores, axis=1, kind="stable")
    return np.take_along_axis(candidate, order, axis=1)


def _mask_seen(scores: np.ndarray, train: sparse.csr_matrix, users: np.ndarray) -> None:
    """In-place ``-inf`` for items the user already used in January."""

    for row, user in enumerate(users):
        start, end = train.indptr[user], train.indptr[user + 1]
        scores[row, train.indices[start:end]] = -np.inf


def _gini(counts: np.ndarray) -> float:
    values = np.sort(counts.astype(np.float64))
    n = values.size
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * values)) / (n * values.sum()) - (n + 1.0) / n)


@dataclass
class EvalResult:
    model: str
    dataset_fingerprint: str
    metrics: dict[str, float]
    per_user: dict[str, np.ndarray]
    n_eval_users: int
    n_items: int
    coverage: dict[str, float]
    gini: float
    runtime_sec: float
    k_values: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "dataset_fingerprint": self.dataset_fingerprint,
            "metrics": self.metrics,
            "n_eval_users": self.n_eval_users,
            "n_items": self.n_items,
            "coverage": self.coverage,
            "gini": self.gini,
            "runtime_sec": self.runtime_sec,
            "k_values": list(self.k_values),
            "metadata": self.metadata,
        }
        (directory / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        np.savez_compressed(directory / "per_user.npz", **self.per_user)  # type: ignore[arg-type]
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> EvalResult:
        directory = Path(directory)
        payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "per_user.npz", allow_pickle=False)
        return cls(
            model=payload["model"],
            dataset_fingerprint=payload["dataset_fingerprint"],
            metrics=payload["metrics"],
            per_user={key: arrays[key] for key in arrays.files},
            n_eval_users=payload["n_eval_users"],
            n_items=payload["n_items"],
            coverage=payload["coverage"],
            gini=payload["gini"],
            runtime_sec=payload["runtime_sec"],
            k_values=tuple(payload["k_values"]),
            metadata=payload.get("metadata", {}),
        )


def evaluate(
    model: BaseRecommender,
    dataset: Dataset,
    *,
    k_values: tuple[int, ...] = (10, 20, 50),
    primary_k: int = 10,
    batch_size: int = 512,
    coverage_k: int = 10,
    min_test_items: int = 1,
) -> EvalResult:
    """Full-catalogue evaluation with seen-item masking.

    The evaluator never looks at February data except through the relevance
    matrix already stored in ``dataset.test``; this makes the protocol easy to
    audit and prevents accidental target leakage in model code.
    """

    if primary_k not in k_values:
        raise ValueError("primary_k must be in k_values")
    max_k = max(k_values)
    if coverage_k <= 0 or coverage_k > max_k:
        raise ValueError("coverage_k must be in [1, max(k_values)]")
    result_arrays: dict[str, list[np.ndarray]] = {
        f"{metric}@{k}": []
        for k in k_values
        for metric in ("recall", "ndcg", "map", "hit_rate", "precision", "mrr")
    }
    recommendation_counts = np.zeros(dataset.n_items, dtype=np.int64)
    eval_users_all = dataset.eval_users
    counts = np.diff(dataset.test.indptr)[eval_users_all]
    eval_users = eval_users_all[counts >= min_test_items]
    if eval_users.size == 0:
        raise ValueError("no evaluation users with the requested minimum number of targets")

    novelty_sum = 0.0
    start_time = time.perf_counter()
    for start in range(0, len(eval_users), batch_size):
        users = eval_users[start : start + batch_size]
        scores = model.score_users(users, train=dataset.train)
        if scores.shape != (len(users), dataset.n_items):
            raise AssertionError(
                f"model returned {scores.shape}, expected {(len(users), dataset.n_items)}"
            )
        _mask_seen(scores, dataset.train, users)
        top = _top_k(scores, max_k)
        relevant = dataset.test[users].toarray() > 0
        relevant_counts = relevant.sum(axis=1)
        if np.any(relevant_counts == 0):
            raise AssertionError("evaluation user without relevant items")

        for k in k_values:
            hits = np.take_along_axis(relevant, top[:, :k], axis=1)
            hit_counts = hits.sum(axis=1)
            ideal = np.array([_ideal_dcg(int(c), k) for c in relevant_counts])
            dcg = np.sum(hits / np.log2(np.arange(2, k + 2))[None, :], axis=1)
            ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
            precision_at = np.cumsum(hits, axis=1) / np.arange(1, k + 1)[None, :]
            ap_denominator = np.minimum(relevant_counts, k)
            average_precision = np.divide(
                np.sum(hits * precision_at, axis=1),
                ap_denominator,
                out=np.zeros(len(users)),
                where=ap_denominator > 0,
            )
            first_hit = np.argmax(hits, axis=1) + 1
            mrr = np.where(hit_counts > 0, 1.0 / first_hit, 0.0)

            result_arrays[f"recall@{k}"].append(
                (hit_counts / np.maximum(relevant_counts, 1)).astype(np.float32)
            )
            result_arrays[f"ndcg@{k}"].append(ndcg.astype(np.float32))
            result_arrays[f"map@{k}"].append(average_precision.astype(np.float32))
            result_arrays[f"hit_rate@{k}"].append((hit_counts > 0).astype(np.float32))
            result_arrays[f"precision@{k}"].append((hit_counts / float(k)).astype(np.float32))
            result_arrays[f"mrr@{k}"].append(mrr.astype(np.float32))

        top_coverage = top[:, :coverage_k]
        np.add.at(recommendation_counts, top_coverage.ravel(), 1)
        # Novelty uses January popularity only, never February labels.
        popularity_share = np.maximum(dataset.train.getnnz(axis=0).ravel(), 1) / dataset.n_users
        novelty_sum += float(np.sum(-np.log2(np.maximum(popularity_share[top_coverage], EPS))))

    metrics: dict[str, float] = {}
    per_user: dict[str, np.ndarray] = {}
    for key, values in result_arrays.items():
        array = np.concatenate(values).astype(np.float32)
        per_user[key] = array
        metrics[key] = float(np.mean(array))
    metrics[f"novelty@{coverage_k}"] = float(
        novelty_sum / (len(eval_users) * min(coverage_k, max_k))
    )
    coverage = {
        f"coverage@{coverage_k}": float(np.count_nonzero(recommendation_counts) / dataset.n_items),
        f"gini@{coverage_k}": _gini(recommendation_counts),
    }
    metrics.update(coverage)
    metrics["n_eval_users"] = float(len(eval_users))
    runtime = time.perf_counter() - start_time
    return EvalResult(
        model=model.name,
        dataset_fingerprint=dataset.fingerprint,
        metrics=metrics,
        per_user=per_user,
        n_eval_users=len(eval_users),
        n_items=dataset.n_items,
        coverage=coverage,
        gini=coverage[f"gini@{coverage_k}"],
        runtime_sec=runtime,
        k_values=k_values,
        metadata={"primary_k": primary_k, "batch_size": batch_size},
    )


def format_metrics(result: EvalResult, k_values: tuple[int, ...] | None = None) -> str:
    """Compact table row for CLI/report output."""

    ks = k_values or result.k_values
    cells = [f"{result.model:<14}"]
    for k in ks:
        cells.append(f"R@{k}={result.metrics[f'recall@{k}']:.4f}")
        cells.append(f"N@{k}={result.metrics[f'ndcg@{k}']:.4f}")
    cov = result.metrics.get("coverage@10")
    if cov is None:
        cov = next(
            (value for key, value in result.metrics.items() if key.startswith("coverage@")), 0.0
        )
    cells.append(f"cov={cov:.3f}")
    return "  ".join(cells)
