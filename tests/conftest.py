from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from engagegraph.config import DataConfig
from engagegraph.data.preprocess import Dataset, build_dataset


@pytest.fixture()
def tiny_dataset() -> Dataset:
    """40 users, 30 items, 5 January apps per user, 2 February adoptions each."""
    rng = np.random.default_rng(11)
    n_users, n_items = 40, 30
    rows, cols = [], []
    for user in range(n_users):
        # Popular items are more likely to be adopted, producing a realistic skew.
        popularity = 1.0 / (np.arange(n_items) + 1.0) ** 0.6
        popularity /= popularity.sum()
        items = rng.choice(n_items, size=6, replace=False, p=popularity)
        for item in items[:5]:
            rows.append(user)
            cols.append(int(item))
    train = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_users, n_items)
    )
    test_rows, test_cols = [], []
    for user in range(n_users):
        seen = set(train[user].indices)
        candidates = [item for item in range(n_items) if item not in seen]
        # Bias toward popular unseen items but keep some variety.
        weights = np.array([1.0 / (item + 1.0) for item in candidates])
        weights /= weights.sum()
        chosen = rng.choice(candidates, size=2, replace=False, p=weights)
        for item in chosen:
            test_rows.append(user)
            test_cols.append(int(item))
    test = sparse.csr_matrix(
        (np.ones(len(test_rows), dtype=np.float32), (test_rows, test_cols)),
        shape=(n_users, n_items),
    )
    return Dataset(
        train=train,
        test=test,
        user_ids=[f"u{i:03d}" for i in range(n_users)],
        item_ids=[f"app-{i:02d}" for i in range(n_items)],
        stats={"n_train_interactions": int(train.nnz), "n_test_interactions": int(test.nnz)},
        config={},
        quality={},
        fingerprint="test-fingerprint",
    )


@pytest.fixture()
def raw_month_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    users = [f"u{i}" for i in range(25)]
    apps = [f"app-{i}" for i in range(12)]
    jan_rows = []
    for user in users:
        for app in rng.choice(apps, size=5, replace=False):
            jan_rows.append(
                {
                    "PANELISTID": user,
                    "APPDESCRIPTION": app,
                    "weight": float(rng.integers(100, 5000)),
                }
            )
    jan = pd.DataFrame(jan_rows)
    feb_rows = []
    for user in users:
        seen = set(jan.loc[user == jan.PANELISTID, "APPDESCRIPTION"])
        candidates = [app for app in apps if app not in seen]
        for app in rng.choice(candidates, size=2, replace=False):
            feb_rows.append(
                {
                    "PANELISTID": user,
                    "APPDESCRIPTION": app,
                    "weight": float(rng.integers(100, 5000)),
                }
            )
    feb = pd.DataFrame(feb_rows)
    unseen_user = pd.DataFrame([{"PANELISTID": "cold", "APPDESCRIPTION": "app-0", "weight": 100.0}])
    return jan, feb, unseen_user


@pytest.fixture()
def parquet_dataset(tmp_path: Path, raw_month_frames) -> tuple[Dataset, DataConfig]:
    jan, feb, cold = raw_month_frames
    jan_path = tmp_path / "jan.parquet"
    feb_path = tmp_path / "feb.parquet"
    pd.concat([jan, cold]).to_parquet(jan_path, index=False)
    pd.concat([feb, cold]).to_parquet(feb_path, index=False)
    config = DataConfig(
        jan_path=str(jan_path),
        feb_path=str(feb_path),
        min_item_users=2,
        min_user_items=3,
        max_items=None,
        weight_transform="log1p",
    )
    return build_dataset(config), config
