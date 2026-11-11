import numpy as np
import pytest
from scipy import sparse

from engagegraph.data.preprocess import Dataset
from engagegraph.evaluation.metrics import evaluate
from engagegraph.models.base import BaseRecommender


class FixedModel(BaseRecommender):
    name = "fixed"

    def __init__(self, matrix):
        super().__init__()
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def fit(self, dataset):  # pragma: no cover - not used
        return self

    def score_users(self, user_indices, candidate_items=None, train=None):
        return self.matrix[user_indices]

    def get_state(self):
        return {"matrix": self.matrix}

    def set_state(self, state):
        self.matrix = state["matrix"]


def make_dataset(train, test):
    return Dataset(
        train=sparse.csr_matrix(np.asarray(train, dtype=np.float32)),
        test=sparse.csr_matrix(np.asarray(test, dtype=np.float32)),
        user_ids=["u0"],
        item_ids=[f"i{i}" for i in range(train.shape[1])],
        stats={},
        config={},
        fingerprint="x",
    )


def test_recall_and_ndcg_hand_calculated():
    train = np.zeros((1, 5))
    train[0, 4] = 1  # seen, must be masked
    test = np.array([[1, 0, 1, 0, 0]])
    scores = np.array([[5, 4, 3, 2, 100]])  # seen item gets highest raw score
    dataset = make_dataset(train, test)
    result = evaluate(FixedModel(scores), dataset, k_values=(2,), primary_k=2, coverage_k=2)
    assert result.metrics["recall@2"] == pytest.approx(0.5)
    expected_ndcg = 1.0 / (1.0 + 1.0 / np.log2(3))
    assert result.metrics["ndcg@2"] == pytest.approx(expected_ndcg)
    assert result.metrics["hit_rate@2"] == pytest.approx(1.0)
    assert result.metrics["precision@2"] == pytest.approx(0.5)


def test_perfect_ranking_scores_one():
    train = np.zeros((2, 4))
    test = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    scores = np.array([[10, 0, 0, 0], [0, 0, 10, 0]])
    dataset = make_dataset(train, test)
    result = evaluate(FixedModel(scores), dataset, k_values=(1, 2), primary_k=1, coverage_k=1)
    assert result.metrics["recall@1"] == 1.0
    assert result.metrics["ndcg@1"] == 1.0
    assert result.metrics["map@1"] == 1.0


def test_evaluate_rejects_wrong_shape(tiny_dataset):
    model = FixedModel(np.zeros((tiny_dataset.n_users, tiny_dataset.n_items - 1)))
    with pytest.raises(AssertionError):
        evaluate(model, tiny_dataset, k_values=(10,), primary_k=10)


def test_metrics_are_bounded(tiny_dataset):
    from engagegraph.models.popularity import PopularityRecommender

    model = PopularityRecommender().fit(tiny_dataset)
    result = evaluate(model, tiny_dataset, k_values=(5, 10, 20), primary_k=10, coverage_k=10)
    for key, value in result.metrics.items():
        if key.startswith(("recall", "ndcg", "map", "hit_rate", "precision", "mrr", "coverage")):
            assert 0.0 <= value <= 1.0
    assert result.per_user["ndcg@10"].shape == (result.n_eval_users,)
