from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from engagegraph.api.app import create_app
from engagegraph.artifacts import save_artifact
from engagegraph.config import DataConfig, DomainConfig
from engagegraph.data.preprocess import build_dataset
from engagegraph.models.popularity import PopularityRecommender
from engagegraph.recommend import RecommendationService


def _write_generic_csvs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(5)
    users = [f"shopper-{i:02d}" for i in range(12)]
    items = [f"product-{i:02d}" for i in range(10)]
    train_rows = []
    test_rows = []
    for user in users:
        chosen = rng.choice(items, size=4, replace=False)
        for item in chosen:
            train_rows.append({"user_id": user, "item_id": item, "count": int(rng.integers(1, 8))})
        remaining = [item for item in items if item not in set(chosen)]
        for item in rng.choice(remaining, size=2, replace=False):
            test_rows.append({"user_id": user, "item_id": item, "count": int(rng.integers(1, 4))})
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(train_rows).to_csv(train_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    return train_path, test_path


def _generic_config(tmp_path: Path, **overrides) -> DataConfig:
    train_path, test_path = _write_generic_csvs(tmp_path)
    domain = DomainConfig(
        name="retail",
        dataset_name="Synthetic retail",
        user_label="shopper",
        user_label_plural="shoppers",
        item_label="product",
        item_label_plural="products",
        interaction_label="event",
        catalog_label="product catalogue",
        task_name="next-period product adoption ranking",
        train_period_label="January 2026",
        test_period_label="February 2026",
        blocklist=("product-00",),
    )
    values = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "format": "csv",
        "user_col": "user_id",
        "item_col": "item_id",
        "weight_col": "count",
        "weight_unit": "count",
        "min_item_users": 1,
        "min_user_items": 2,
        "max_items": None,
        "domain": domain,
    }
    values.update(overrides)
    return DataConfig(**values)


def test_generic_csv_dataset_and_domain_metadata(tmp_path):
    config = _generic_config(tmp_path)
    dataset = build_dataset(config)
    assert dataset.stats["domain"]["item_label_plural"] == "products"
    assert dataset.stats["task"] == "next-period product adoption ranking"
    assert dataset.stats["test_period_users"] > 0
    assert dataset.n_users == 12
    assert dataset.n_items == 10


def test_weightless_csv_uses_binary_interactions(tmp_path):
    train_path, test_path = _write_generic_csvs(tmp_path)
    train = pd.read_csv(train_path)[["user_id", "item_id"]]
    test = pd.read_csv(test_path)[["user_id", "item_id"]]
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    config = DataConfig(
        train_path=str(train_path),
        test_path=str(test_path),
        format="csv",
        user_col="user_id",
        item_col="item_id",
        weight_col=None,
        weight_unit="count",
        min_item_users=1,
        min_user_items=2,
        max_items=None,
        weight_transform="binary",
    )
    dataset = build_dataset(config)
    assert set(np.unique(dataset.train.data)) == {1.0}


def test_domain_labels_validation():
    with pytest.raises(ValueError):
        DomainConfig(item_label="   ")


def test_service_blocklist_comes_from_domain(tmp_path):
    config = _generic_config(tmp_path)
    dataset = build_dataset(config)
    model = PopularityRecommender().fit(dataset)
    artifact = save_artifact(model, dataset, tmp_path / "artifact")
    service = RecommendationService(artifact)
    assert service.item_label_plural == "products"
    response = service.recommend(dataset.user_ids[0], k=5)
    assert "product-00" not in {item.item_id for item in response.items}
    assert service.metadata()["domain"]["name"] == "retail"


def test_generic_api_aliases_and_profile(tmp_path):
    config = _generic_config(tmp_path)
    dataset = build_dataset(config)
    model = PopularityRecommender().fit(dataset)
    artifact = save_artifact(model, dataset, tmp_path / "artifact")
    with TestClient(create_app(artifact.path)) as client:
        meta = client.get("/v1/meta").json()
        assert meta["domain"]["item_label_plural"] == "products"
        assert meta["labels"]["user"] == "shopper"
        user = dataset.user_ids[0]
        profile = client.get(f"/v1/users/{user}").json()
        assert "n_items" in profile and "top_items" in profile
        similar = client.get("/v1/items/similar", params={"item_id": dataset.item_ids[1]})
        assert similar.status_code == 200
        legacy = client.get("/v1/apps/similar", params={"item_id": dataset.item_ids[1]})
        assert legacy.status_code == 200
