from __future__ import annotations

import numpy as np
from scipy import sparse

from engagegraph.config import DomainConfig
from engagegraph.data.preprocess import Dataset, build_dataset
from engagegraph.evaluation.metrics import evaluate
from engagegraph.models.base import BaseRecommender
from engagegraph.models.itemknn import ItemKNNRecommender
from engagegraph.models.popularity import PopularityRecommender
from engagegraph.recommend import within_user_percentile


def _dataset_from_matrix(matrix: np.ndarray, test: np.ndarray | None = None) -> Dataset:
    n_users, n_items = matrix.shape
    test = np.zeros_like(matrix) if test is None else test
    return Dataset(
        train=sparse.csr_matrix(matrix.astype(np.float32)),
        test=sparse.csr_matrix(test.astype(np.float32)),
        user_ids=[f"u{i}" for i in range(n_users)],
        item_ids=[f"i{i}" for i in range(n_items)],
        stats={},
        config={},
        fingerprint="math-check",
    )


def test_percentile_definition_and_ties():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    candidates = np.array([0, 1, 2, 3])
    selected = np.array([0, 1, 3])
    result = within_user_percentile(scores, candidates, selected)
    assert np.allclose(result, [0.25, 0.5, 1.0])

    tied = np.array([2.0, 2.0, 2.0, 2.0])
    assert np.allclose(within_user_percentile(tied, candidates, selected), [1.0, 1.0, 1.0])
    # A strictly increasing transform cannot change a rank statistic.
    assert np.allclose(
        within_user_percentile(scores**3, candidates, selected),
        within_user_percentile(scores, candidates, selected),
    )


def test_itemknn_matches_bruteforce_cosine():
    matrix = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    dataset = _dataset_from_matrix(matrix)
    model = ItemKNNRecommender(k=4, binary=True).fit(dataset)

    norms = np.linalg.norm(matrix, axis=0)
    cosine = (matrix.T @ matrix) / np.outer(norms, norms)
    np.fill_diagonal(cosine, 0.0)
    expected = matrix @ cosine
    actual = model.score_users(np.array([0, 1, 2]), train=dataset.train)
    assert np.allclose(actual, expected[[0, 1, 2]], atol=1e-5)


def test_metrics_are_invariant_to_positive_monotone_scores(tiny_dataset):
    class MonotoneModel(BaseRecommender):
        name = "monotone"

        def __init__(self, base: PopularityRecommender) -> None:
            super().__init__()
            self.base = base

        def fit(self, dataset):  # pragma: no cover
            return self

        def score_users(self, user_indices, candidate_items=None, train=None):
            scores = self.base.score_users(user_indices)
            return np.power(scores, 3).astype(np.float32)

        def get_state(self):  # pragma: no cover
            return {}

        def set_state(self, state):  # pragma: no cover
            return None

    base = PopularityRecommender().fit(tiny_dataset)
    reference = evaluate(base, tiny_dataset, k_values=(5, 10), primary_k=10, coverage_k=10)
    transformed = evaluate(
        MonotoneModel(base), tiny_dataset, k_values=(5, 10), primary_k=10, coverage_k=10
    )
    for key in ("recall@10", "ndcg@10", "map@10", "hit_rate@10"):
        assert transformed.metrics[key] == reference.metrics[key]


def test_domain_labels_do_not_change_matrices_or_scores(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train_rows = []
    for user in range(8):
        for item in range(4):
            train_rows.append((f"u{user}", f"i{item}", 1))
    train_rows.extend([("u0", "i4", 1), ("u1", "i5", 1)])
    train.write_text("user,item,weight\n" + "\n".join(f"{u},{i},{w}" for u, i, w in train_rows))
    test.write_text("user,item,weight\nu2,i4,1\nu3,i5,1\nu4,i4,1\nu5,i5,1\n")

    from engagegraph.config import DataConfig

    app_domain = DomainConfig(name="apps", item_label="app", item_label_plural="apps")
    other_domain = DomainConfig(name="products", item_label="product", item_label_plural="products")
    base = DataConfig(
        train_path=str(train),
        test_path=str(test),
        format="csv",
        user_col="user",
        item_col="item",
        weight_col="weight",
        min_item_users=1,
        min_user_items=2,
        max_items=None,
        weight_transform="binary",
        domain=app_domain,
    )
    other = base.model_copy(update={"domain": other_domain})
    first = build_dataset(base)
    second = build_dataset(other)
    assert (first.train != second.train).nnz == 0
    assert (first.test != second.test).nnz == 0
    assert first.stats["domain"]["item_label_plural"] == "apps"
    assert second.stats["domain"]["item_label_plural"] == "products"

    model_a = ItemKNNRecommender(k=5).fit(first)
    model_b = ItemKNNRecommender(k=5).fit(second)
    users = np.array([0, 1, 2])
    assert np.allclose(
        model_a.score_users(users, train=first.train),
        model_b.score_users(users, train=second.train),
    )


def test_test_matrix_does_not_affect_fitted_scores(tiny_dataset):
    base = PopularityRecommender().fit(tiny_dataset)
    altered = Dataset(
        train=tiny_dataset.train,
        test=sparse.csr_matrix(tiny_dataset.test.shape, dtype=np.float32),
        user_ids=tiny_dataset.user_ids,
        item_ids=tiny_dataset.item_ids,
        stats=tiny_dataset.stats,
        config=tiny_dataset.config,
        fingerprint="altered",
    )
    assert np.allclose(
        base.score_users(np.array([0, 5])),
        PopularityRecommender().fit(altered).score_users(np.array([0, 5])),
    )
