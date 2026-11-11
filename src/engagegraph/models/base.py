"""Model contracts shared by every recommender in EngageGraph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse

from engagegraph.data.preprocess import Dataset


@dataclass(frozen=True)
class Explanation:
    """A human-readable reason attached to one recommendation."""

    item_id: str
    reason: str
    contribution: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "reason": self.reason,
            "contribution": round(float(self.contribution), 6),
        }


@dataclass
class Recommendation:
    item_id: str
    score: float
    rank: int
    explanations: list[Explanation] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
            "explanations": [e.as_dict() for e in self.explanations],
        }


class BaseRecommender(ABC):
    """Minimal interface required by the evaluator, API, and CLI.

    Implementations must be deterministic given a fitted dataset and must not
    read February labels.  ``score_users`` returns one score per catalogue item
    for each requested user; masking seen items is the caller's responsibility
    so that the evaluator and serving layer use exactly the same candidate set.
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.n_users: int = 0
        self.n_items: int = 0
        self.item_popularity: np.ndarray = np.zeros(0, dtype=np.float32)

    @abstractmethod
    def fit(self, dataset: Dataset) -> BaseRecommender:
        """Fit on the January training matrix only."""

    @abstractmethod
    def score_users(
        self,
        user_indices: np.ndarray,
        candidate_items: np.ndarray | None = None,
        train: sparse.csr_matrix | None = None,
    ) -> np.ndarray:
        """Return a ``(len(user_indices), n_items)`` score matrix.

        ``candidate_items`` is an optional subset used by tests and by the
        serving layer; when supplied the returned matrix has one column per
        candidate, in that order.  ``train`` is required by memory-based models
        such as item-kNN and ignored by parametric models.
        """

    @abstractmethod
    def get_state(self) -> dict[str, np.ndarray]:
        """NumPy-only state, loadable without pickle."""

    @abstractmethod
    def set_state(self, state: dict[str, np.ndarray]) -> None:
        """Restore state previously produced by :meth:`get_state`."""

    def meta(self) -> dict[str, Any]:
        return {"model_type": self.name, "params": self.params}

    def similar_items(self, item_index: int, k: int = 10) -> list[tuple[int, float]]:
        """Nearest catalogue items; models without item geometry may raise."""

        raise NotImplementedError(f"{self.name} does not expose item similarity")

    def explain(
        self,
        user_index: int,
        item_index: int,
        dataset: Dataset,
        k: int = 3,
    ) -> list[Explanation]:
        """Default explanation: strongest co-usage signal in January.

        This is model-agnostic and intentionally conservative.  It answers
        "which apps in this user's history co-occur most with the recommended
        app?", not "which latent factor fired?".
        """

        if item_index < 0 or item_index >= dataset.n_items:
            return []
        domain = dataset.stats.get("domain", {}) if dataset.stats else {}
        train_label = domain.get("train_period_label", "the training period")
        column = dataset.train[:, item_index]
        row = column.toarray().ravel()
        seen = np.flatnonzero(dataset.train[user_index].toarray().ravel() > 0)
        if seen.size == 0:
            return []
        support = max(float(row.sum()), 1.0)
        contributions = row[seen] / support
        order = np.argsort(-contributions)[:k]
        reasons: list[Explanation] = []
        for position in order:
            if contributions[position] <= 0:
                continue
            source = int(seen[position])
            reasons.append(
                Explanation(
                    item_id=dataset.item_ids[source],
                    reason=f"co-used with {dataset.item_ids[source]} in {train_label}",
                    contribution=float(contributions[position]),
                )
            )
        return reasons

    def explain_popularity(self, item_index: int, dataset: Dataset) -> list[Explanation]:
        domain = dataset.stats.get("domain", {}) if dataset.stats else {}
        user_plural = domain.get("user_label_plural", "users")
        train_label = domain.get("train_period_label", "the training period")
        return [
            Explanation(
                item_id=dataset.item_ids[item_index],
                reason=(
                    f"used by {int(self.item_popularity[item_index]):,} "
                    f"{user_plural} in {train_label}"
                ),
                contribution=float(self.item_popularity[item_index]),
            )
        ]

    @staticmethod
    def _as_candidate_scores(
        full_scores: np.ndarray, candidate_items: np.ndarray | None
    ) -> np.ndarray:
        if candidate_items is None:
            return full_scores
        return full_scores[candidate_items]

    @staticmethod
    def _seen_mask(train: sparse.csr_matrix, user_indices: np.ndarray) -> np.ndarray:
        return np.asarray(train[user_indices].toarray() > 0)


def build_model(name: str, params: dict[str, Any] | None = None) -> BaseRecommender:
    """Factory used by configs and artefact loading."""

    params = dict(params or {})
    if name == "popularity":
        from engagegraph.models.popularity import PopularityRecommender

        return PopularityRecommender(**params)
    if name == "itemknn":
        from engagegraph.models.itemknn import ItemKNNRecommender

        return ItemKNNRecommender(**params)
    if name == "als":
        from engagegraph.models.als import ALSRecommender

        return ALSRecommender(**params)
    if name == "metapath2vec":
        from engagegraph.models.metapath2vec import MetaPath2VecRecommender

        return MetaPath2VecRecommender(**params)
    raise ValueError(f"unknown model: {name}")
