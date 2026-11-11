"""Implicit-feedback matrix factorisation (Hu, Koren & Volinsky, 2008).

The implementation delegates the alternating-least-squares solver to the
well-tested :mod:`implicit` package and keeps only the learned factors in the
artefact.  Confidence is ``1 + alpha * r`` where ``r`` is the configured
engagement transform; this is the standard implicit ALS formulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import BaseRecommender, Explanation


class ALSRecommender(BaseRecommender):
    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 40.0,
        random_state: int = 42,
        use_gpu: bool = False,
    ) -> None:
        if factors <= 0:
            raise ValueError("factors must be positive")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        super().__init__(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=random_state,
            use_gpu=use_gpu,
        )
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.user_factors = np.zeros((0, factors), dtype=np.float32)
        self.item_factors = np.zeros((0, factors), dtype=np.float32)

    def fit(self, dataset: Dataset) -> ALSRecommender:
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "implicit is required for the ALS model: pip install 'engagegraph[als]'"
            ) from exc

        self.n_users, self.n_items = dataset.train.shape
        self.item_popularity = np.asarray(dataset.train.getnnz(axis=0)).ravel().astype(np.float32)
        confidence = dataset.train.astype(np.float32)
        # Store item factors in the same orientation used by implicit (items x factors).
        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=self.alpha,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            dtype=np.float32,
        )
        model.fit(confidence, show_progress=False)
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        return self

    def score_users(
        self,
        user_indices: np.ndarray,
        candidate_items: np.ndarray | None = None,
        train: sparse.csr_matrix | None = None,
    ) -> np.ndarray:
        scores = self.user_factors[user_indices] @ self.item_factors.T
        return self._as_candidate_scores(scores.astype(np.float32), candidate_items)

    def similar_items(self, item_index: int, k: int = 10) -> list[tuple[int, float]]:
        if self.item_factors.size == 0:
            return []
        vectors = self.item_factors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        scores = normalized @ normalized[item_index]
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order if i != item_index][:k]

    def explain(
        self, user_index: int, item_index: int, dataset: Dataset, k: int = 3
    ) -> list[Explanation]:
        # Latent dimensions are not semantically labelled; co-usage is the
        # honest, auditable explanation for this model.
        return super().explain(user_index, item_index, dataset, k)

    def get_state(self) -> dict[str, np.ndarray]:
        return {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "item_popularity": self.item_popularity,
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        self.user_factors = state["user_factors"].astype(np.float32)
        self.item_factors = state["item_factors"].astype(np.float32)
        self.n_users, self.factors = self.user_factors.shape
        self.n_items = int(self.item_factors.shape[0])
        self.item_popularity = state.get(
            "item_popularity", np.zeros(self.n_items, dtype=np.float32)
        ).astype(np.float32)

    def meta(self) -> dict[str, Any]:
        return {
            "model_type": self.name,
            "params": self.params,
            "n_users": int(self.n_users),
            "n_items": int(self.n_items),
            "factors": int(self.factors),
        }
