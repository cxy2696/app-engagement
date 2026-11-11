import json

import numpy as np
import pytest

from engagegraph.artifacts import ModelRegistry, load_artifact, save_artifact
from engagegraph.models.popularity import PopularityRecommender


def test_artifact_roundtrip(tiny_dataset, tmp_path):
    model = PopularityRecommender().fit(tiny_dataset)
    save_artifact(model, tiny_dataset, tmp_path / "artifact", metrics={"ndcg@10": 0.1})
    assert (tmp_path / "artifact" / "manifest.json").exists()
    assert (tmp_path / "artifact" / "history.npz").exists()
    loaded = load_artifact(tmp_path / "artifact")
    assert loaded.user_ids == tiny_dataset.user_ids
    assert loaded.history.shape == tiny_dataset.train.shape
    assert loaded.metrics["ndcg@10"] == 0.1
    assert np.allclose(
        loaded.model.score_users(np.array([0, 1])),
        model.score_users(np.array([0, 1])),
    )
    manifest = json.loads((tmp_path / "artifact" / "manifest.json").read_text())
    assert manifest["artifact_fingerprint"]


def test_registry_activates_artifact(tiny_dataset, tmp_path):
    model = PopularityRecommender().fit(tiny_dataset)
    artifact = save_artifact(model, tiny_dataset, tmp_path / "artifacts" / "demo")
    registry = ModelRegistry(tmp_path / "artifacts")
    registry.register(artifact, make_active=True)
    resolved = registry.resolve()
    assert resolved.path == artifact.path
    payload = json.loads((tmp_path / "artifacts" / "registry.json").read_text())
    assert payload["active"] == str(artifact.path)


def test_loading_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope")
