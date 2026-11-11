"""Global popularity baseline.

Popularity is a surprisingly strong next-app baseline: catalogue-level demand
is stable across adjacent months, and it is the correct fallback for users with
no usable history.  The implementation deliberately exposes no personalisation;
any gain over it must come from user-specific signal.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import BaseRecommender, Explanation


class PopularityRecommender(BaseRecommender):
    name = "popularity"

    def __init__(self, score: str = "users", log_scale: bool = True) -> None:
        if score not in {"users", "weight"}:
            raise ValueError("score must be 'users' or 'weight'")
        super().__init__(score=score, log_scale=log_scale)
        self.score = score
        self.log_scale = log_scale
        self._scores = np.zeros(0, dtype=np.float32)

    def fit(self, dataset: Dataset) -> PopularityRecommender:
        self.n_users, self.n_items = dataset.train.shape
        if self.score == "users":
            counts = np.asarray(dataset.train.getnnz(axis=0)).ravel().astype(np.float64)
        else:
            counts = np.asarray(dataset.train.sum(axis=0)).ravel().astype(np.float64)
        if self.log_scale:
            counts = np.log1p(counts)
        self._scores = counts.astype(np.float32)
        self.item_popularity = np.asarray(dataset.train.getnnz(axis=0)).ravel().astype(np.float32)
        return self

    def score_users(
        self,
        user_indices: np.ndarray,
        candidate_items: np.ndarray | None = None,
        train: sparse.csr_matrix | None = None,
    ) -> np.ndarray:
        scores = np.broadcast_to(self._scores, (len(user_indices), self.n_items)).copy()
        return self._as_candidate_scores(scores, candidate_items)

    def similar_items(self, item_index: int, k: int = 10) -> list[tuple[int, float]]:
        order = np.argsort(-self._scores)
        return [(int(i), float(self._scores[i])) for i in order if i != item_index][:k]

    def explain(
        self, user_index: int, item_index: int, dataset: Dataset, k: int = 3
    ) -> list[Explanation]:
        return self.explain_popularity(item_index, dataset)

    def get_state(self) -> dict[str, np.ndarray]:
        return {
            "scores": self._scores,
            "item_popularity": self.item_popularity,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self._scores = state["scores"].astype(np.float32)
        self.item_popularity = state.get("item_popularity", np.zeros_like(self._scores)).astype(
            np.float32
        )
        self.n_items = int(self._scores.shape[0])
