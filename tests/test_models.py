import numpy as np
import pytest

from engagegraph.models.itemknn import ItemKNNRecommender
from engagegraph.models.popularity import PopularityRecommender


def test_popularity_is_deterministic(tiny_dataset):
    first = PopularityRecommender().fit(tiny_dataset)
    second = PopularityRecommender().fit(tiny_dataset)
    users = np.array([0, 3, 7])
    assert np.allclose(first.score_users(users), second.score_users(users))
    assert first.similar_items(0, 3)


def test_itemknn_scores_and_state_roundtrip(tiny_dataset):
    model = ItemKNNRecommender(k=10).fit(tiny_dataset)
    users = np.array([0, 1, 2])
    scores = model.score_users(users, train=tiny_dataset.train)
    assert scores.shape == (3, tiny_dataset.n_items)
    assert np.isfinite(scores).all()
    neighbours = model.similar_items(0, 5)
    assert len(neighbours) <= 5
    assert all(0 <= idx < tiny_dataset.n_items for idx, _ in neighbours)

    clone = ItemKNNRecommender(k=10)
    clone.set_state(model.get_state())
    assert np.allclose(scores, clone.score_users(users, train=tiny_dataset.train))
    assert model.explain(0, neighbours[0][0], tiny_dataset)


def test_itemknn_requires_training_matrix(tiny_dataset):
    model = ItemKNNRecommender().fit(tiny_dataset)
    with pytest.raises(ValueError):
        model.score_users(np.array([0]))


def test_unknown_model_rejected():
    from engagegraph.models.base import build_model

    with pytest.raises(ValueError):
        build_model("not-a-model")


def test_als_if_available(tiny_dataset):
    pytest.importorskip("implicit")
    from engagegraph.models.als import ALSRecommender

    model = ALSRecommender(factors=8, iterations=2, random_state=1).fit(tiny_dataset)
    scores = model.score_users(np.array([0, 1]), train=tiny_dataset.train)
    assert scores.shape == (2, tiny_dataset.n_items)
    assert np.isfinite(scores).all()
    clone = ALSRecommender(factors=8, iterations=2)
    clone.set_state(model.get_state())
    assert np.allclose(scores, clone.score_users(np.array([0, 1])))


@pytest.mark.graph
def test_metapath2vec_if_available(tiny_dataset):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from engagegraph.models.metapath2vec import MetaPath2VecRecommender

    model = MetaPath2VecRecommender(
        embedding_dim=8,
        walk_length=4,
        context_size=2,
        walks_per_node=1,
        num_negative_samples=1,
        epochs=1,
        batch_size=8,
    ).fit(tiny_dataset)
    scores = model.score_users(np.array([0, 1]), train=tiny_dataset.train)
    assert scores.shape == (2, tiny_dataset.n_items)
    assert np.isfinite(scores).all()
