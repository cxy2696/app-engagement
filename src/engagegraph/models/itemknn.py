"""Item-based collaborative filtering with truncated cosine similarity.

The model is a strong, transparent baseline for implicit engagement data.  It
stores only the top-``k`` neighbours of each item, which keeps the artefact
small and makes explanations exact (the neighbour graph itself is the reason).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import BaseRecommender, Explanation


class ItemKNNRecommender(BaseRecommender):
    name = "itemknn"

    def __init__(
        self,
        k: int = 100,
        binary: bool = True,
        tfidf: bool = False,
        block_size: int = 512,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        super().__init__(k=k, binary=binary, tfidf=tfidf, block_size=block_size)
        self.k = k
        self.binary = binary
        self.tfidf = tfidf
        self.block_size = block_size
        self._sim: sparse.csr_matrix | None = None
        self._sim_t: sparse.csr_matrix | None = None

    def _prepare_matrix(self, train: sparse.csr_matrix) -> sparse.csr_matrix:
        matrix = train.copy().astype(np.float32)
        if self.binary:
            matrix.data = np.ones_like(matrix.data)
        if self.tfidf:
            df = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float64)
            idf = np.log((1.0 + matrix.shape[0]) / (1.0 + df)) + 1.0
            matrix = matrix.multiply(idf[None, :]).tocsr().astype(np.float32)
        return matrix

    def fit(self, dataset: Dataset) -> ItemKNNRecommender:
        self.n_users, self.n_items = dataset.train.shape
        matrix = self._prepare_matrix(dataset.train)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        self.item_popularity = np.asarray(dataset.train.getnnz(axis=0)).ravel().astype(np.float32)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        k = min(self.k, max(self.n_items - 1, 1))
        for start in range(0, self.n_items, self.block_size):
            end = min(start + self.block_size, self.n_items)
            # (block, n_items) co-occurrence counts, then cosine normalisation.
            block = (matrix[:, start:end].T @ matrix).toarray().astype(np.float32)
            block /= norms[start:end, None] * norms[None, :]
            block[np.arange(end - start), np.arange(start, end)] = 0.0
            if k < block.shape[1]:
                top = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
            else:
                top = np.tile(np.arange(block.shape[1]), (block.shape[0], 1))
            block_rows = np.arange(end - start)[:, None]
            values = block[block_rows, top]
            # Drop zero-similarity neighbours to keep the artefact sparse.
            keep = values > 0
            rows.append(np.repeat(np.arange(start, end), keep.sum(axis=1)))
            cols.append(top[keep])
            vals.append(values[keep])
            del block

        sim = sparse.coo_matrix(
            (
                np.concatenate(vals).astype(np.float32) if vals else np.array([], np.float32),
                (
                    np.concatenate(rows) if rows else np.array([], np.int32),
                    np.concatenate(cols) if cols else np.array([], np.int32),
                ),
            ),
            shape=(self.n_items, self.n_items),
            dtype=np.float32,
        ).tocsr()
        sim.sum_duplicates()
        sim.sort_indices()
        self._sim = sim
        self._sim_t = sim.T.tocsr()
        return self

    def _require_fit(self) -> sparse.csr_matrix:
        if self._sim is None:
            raise RuntimeError("model is not fitted")
        return self._sim

    def score_users(
        self,
        user_indices: np.ndarray,
        candidate_items: np.ndarray | None = None,
        train: sparse.csr_matrix | None = None,
    ) -> np.ndarray:
        if train is None:
            raise ValueError("item-kNN scoring requires the training matrix")
        sim = self._require_fit()
        batch = train[user_indices]
        scores = (batch @ sim).toarray().astype(np.float32)
        return self._as_candidate_scores(scores, candidate_items)

    def score_from_matrix(
        self,
        train: sparse.csr_matrix,
        user_indices: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        """Kept for explicit, readable call sites in notebooks and tests."""

        return self.score_users(user_indices, candidate_items, train)

    def similar_items(self, item_index: int, k: int = 10) -> list[tuple[int, float]]:
        sim = self._require_fit()
        row = sim.getrow(item_index)
        if row.nnz == 0:
            return []
        order = np.argsort(-row.data)[:k]
        return [(int(row.indices[i]), float(row.data[i])) for i in order]

    def explain(
        self, user_index: int, item_index: int, dataset: Dataset, k: int = 3
    ) -> list[Explanation]:
        if self._sim_t is None:
            return super().explain(user_index, item_index, dataset, k)
        domain = dataset.stats.get("domain", {}) if dataset.stats else {}
        user_plural = domain.get("user_label_plural", "users")
        item_label = domain.get("item_label", "item")
        column = self._sim_t[:, item_index].toarray().ravel()
        seen = np.flatnonzero(dataset.train[user_index].toarray().ravel() > 0)
        contributions = column[seen]
        order = np.argsort(-contributions)[:k]
        explanations: list[Explanation] = []
        for position in order:
            if contributions[position] <= 0:
                continue
            source = int(seen[position])
            explanations.append(
                Explanation(
                    item_id=dataset.item_ids[source],
                    reason=(
                        f"{user_plural} of {dataset.item_ids[source]} also used this {item_label}"
                    ),
                    contribution=float(contributions[position]),
                )
            )
        return explanations

    def get_state(self) -> dict[str, np.ndarray]:
        sim = self._require_fit()
        return {
            "sim_indptr": sim.indptr.astype(np.int64),
            "sim_indices": sim.indices.astype(np.int32),
            "sim_data": sim.data.astype(np.float32),
            "item_popularity": self.item_popularity,
            "shape": np.asarray(sim.shape, dtype=np.int64),
        }

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        shape = tuple(int(x) for x in state["shape"])
        self._sim = sparse.csr_matrix(
            (state["sim_data"], state["sim_indices"], state["sim_indptr"]), shape=shape
        )
        self._sim_t = self._sim.T.tocsr()
        self.n_items = shape[1]
        self.item_popularity = state.get(
            "item_popularity", np.zeros(shape[1], dtype=np.float32)
        ).astype(np.float32)
