import json

import pytest
from fastapi.testclient import TestClient

from engagegraph.api.app import create_app
from engagegraph.artifacts import save_artifact
from engagegraph.models.popularity import PopularityRecommender


@pytest.fixture()
def client(tiny_dataset, tmp_path, monkeypatch):
    model = PopularityRecommender().fit(tiny_dataset)
    artifact = save_artifact(model, tiny_dataset, tmp_path / "artifact")
    monkeypatch.setenv("ENGAGEGRAPH_FEEDBACK_DIR", str(tmp_path / "feedback"))
    with TestClient(create_app(artifact.path)) as test_client:
        yield test_client


def test_health_and_meta(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    meta = client.get("/v1/meta").json()
    assert meta["dataset"]["n_users"] == 40
    assert meta["model"]["model_type"] == "popularity"
    assert client.get("/version").json()["version"]


def test_recommendations_and_similar(client, tiny_dataset):
    user = tiny_dataset.user_ids[0]
    response = client.get(f"/v1/users/{user}/recommendations", params={"k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 3
    assert payload["strategy"] == "personalized"
    assert 0.0 < payload["items"][0]["percentile"] <= 1.0

    filtered = client.get(
        f"/v1/users/{user}/recommendations", params={"k": 10, "min_percentile": 0.95}
    )
    assert filtered.status_code == 200
    assert filtered.json()["min_percentile"] == 0.95
    assert all(item["percentile"] >= 0.95 for item in filtered.json()["items"])

    similar = client.get("/v1/apps/similar", params={"item_id": tiny_dataset.item_ids[0]})
    assert similar.status_code == 200
    assert similar.json()["items"]

    missing = client.get("/v1/apps/similar", params={"item_id": "nope"})
    assert missing.status_code == 404


def test_batch_and_popular(client, tiny_dataset):
    response = client.post(
        "/v1/recommendations",
        json={"user_ids": tiny_dataset.user_ids[:3], "k": 2},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 3
    popular = client.get("/v1/popular", params={"k": 5}).json()
    assert len(popular["items"]) == 5


def test_feedback_is_persisted(client, tiny_dataset, tmp_path):
    response = client.post(
        "/v1/feedback",
        json={
            "impression_id": "imp-1",
            "user_id": tiny_dataset.user_ids[0],
            "item_id": tiny_dataset.item_ids[0],
            "event": "click",
        },
    )
    assert response.status_code == 202
    files = list((tmp_path / "feedback").glob("*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text().strip())
    assert record["event"] == "click"
    assert record["server_user_hash"]


def test_missing_artifact_is_degraded(tiny_dataset, tmp_path):
    with TestClient(create_app(tmp_path / "missing")) as test_client:
        health = test_client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        assert test_client.get("/readyz").status_code == 503
        assert test_client.get("/v1/meta").status_code == 503


def test_metrics_endpoint(client):
    client.get("/healthz")
    text = client.get("/metrics").text
    assert "engagegraph_requests_total" in text
    assert "engagegraph_uptime_seconds" in text
