import pytest

from engagegraph.artifacts import save_artifact
from engagegraph.models.popularity import PopularityRecommender
from engagegraph.recommend import RecommendationService


@pytest.fixture()
def service(tiny_dataset, tmp_path):
    model = PopularityRecommender().fit(tiny_dataset)
    artifact = save_artifact(model, tiny_dataset, tmp_path / "artifact")
    return RecommendationService(artifact)


def test_personalized_recommendations_exclude_seen(service, tiny_dataset):
    user_id = tiny_dataset.user_ids[0]
    response = service.recommend(user_id, k=5)
    assert response.strategy == "personalized"
    assert len(response.items) == 5
    seen = {tiny_dataset.item_ids[i] for i in tiny_dataset.train[0].indices}
    assert not ({item.item_id for item in response.items} & seen)
    assert response.items[0].explanations
    assert all(item.percentile is not None for item in response.items)
    assert all(0.0 < item.percentile <= 1.0 for item in response.items)


def test_cold_start_falls_back_to_popularity(service):
    response = service.recommend("not-a-panelist", k=4)
    assert response.strategy == "cold_start_popularity"
    assert len(response.items) == 4
    assert response.items[0].rank == 1


def test_similar_and_profile(service, tiny_dataset):
    item = tiny_dataset.item_ids[0]
    neighbours = service.similar(item, k=3)
    assert neighbours
    assert all(n.item_id != item for n in neighbours)
    profile = service.profile(tiny_dataset.user_ids[0])
    assert profile["n_apps"] >= 1
    assert profile["top_apps"]


def test_unknown_ids_raise(service):
    with pytest.raises(KeyError):
        service.similar("not-an-app")
    with pytest.raises(KeyError):
        service.profile("not-a-user")


def test_k_bounds(service, tiny_dataset):
    with pytest.raises(ValueError):
        service.recommend(tiny_dataset.user_ids[0], k=0)
    with pytest.raises(ValueError):
        service.popular(k=101)


def test_metadata_contains_provenance(service):
    meta = service.metadata()
    assert meta["dataset"]["fingerprint"] == "test-fingerprint"
    assert meta["model"]["model_type"] == "popularity"
    assert meta["sample_users"]
