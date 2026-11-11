"""Build a leakage-free, temporal app-engagement benchmark from monthly exports.

Protocol
--------
* January is the only month used to fit mappings, item statistics, and models.
* February is used exclusively to construct *future adoption* labels.
* The candidate catalogue is fixed from January.  February items that were not
  in the January catalogue are out of scope and counted, not silently added.
* A test label is a February interaction with an item the user did **not** use
  in January.  This measures new-app discovery rather than repeat usage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from engagegraph.config import DataConfig
from engagegraph.data.schema import (
    clean_interactions,
    validate_no_leakage,
)

SCHEMA_VERSION = "2.0"


@dataclass
class Dataset:
    """In-memory benchmark matrices plus the provenance needed to audit them."""

    train: sparse.csr_matrix
    test: sparse.csr_matrix
    user_ids: list[str]
    item_ids: list[str]
    stats: dict[str, Any]
    config: dict[str, Any]
    quality: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def user_index(self) -> dict[str, int]:
        return {user: idx for idx, user in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[str, int]:
        return {item: idx for idx, item in enumerate(self.item_ids)}

    @property
    def eval_users(self) -> np.ndarray:
        counts = np.diff(self.test.indptr)
        return np.flatnonzero(counts > 0).astype(np.int32)

    def save(self, directory: str | Path) -> Path:
        """Persist to a directory of portable, non-pickle artefacts."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "dataset.npz",
            train_data=self.train.data.astype(np.float32, copy=False),
            train_indices=self.train.indices.astype(np.int32, copy=False),
            train_indptr=self.train.indptr.astype(np.int64, copy=False),
            test_data=self.test.data.astype(np.float32, copy=False),
            test_indices=self.test.indices.astype(np.int32, copy=False),
            test_indptr=self.test.indptr.astype(np.int64, copy=False),
            shape=np.asarray(self.train.shape, dtype=np.int64),
        )
        (directory / "mappings.json").write_text(
            json.dumps({"user_ids": self.user_ids, "item_ids": self.item_ids}, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "stats": self.stats,
            "config": self.config,
            "quality": self.quality,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> Dataset:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        mappings = json.loads((directory / "mappings.json").read_text(encoding="utf-8"))
        arrays = np.load(directory / "dataset.npz", allow_pickle=False)
        shape = tuple(int(x) for x in arrays["shape"])
        train = sparse.csr_matrix(
            (arrays["train_data"], arrays["train_indices"], arrays["train_indptr"]),
            shape=shape,
        )
        test = sparse.csr_matrix(
            (arrays["test_data"], arrays["test_indices"], arrays["test_indptr"]),
            shape=shape,
        )
        return cls(
            train=train.astype(np.float32),
            test=test.astype(np.float32),
            user_ids=list(mappings["user_ids"]),
            item_ids=list(mappings["item_ids"]),
            stats=manifest["stats"],
            config=manifest["config"],
            quality=manifest.get("quality", {}),
            fingerprint=manifest.get("fingerprint", ""),
        )


def _sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: str, columns: list[str], file_format: str = "auto") -> pd.DataFrame:
    """Read a Parquet, CSV, or JSONL interaction table.

    The reader is intentionally strict: missing requested columns raise a
    ``ValueError`` naming the file and the missing columns, rather than letting a
    downstream groupby fail with an opaque KeyError.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"interaction file not found: {source}")
    suffix = source.suffix.lower()
    kind = file_format
    if kind == "auto":
        if suffix in {".parquet", ".pq"}:
            kind = "parquet"
        elif suffix in {".csv", ".tsv", ".txt"}:
            kind = "csv"
        elif suffix in {".jsonl", ".ndjson"}:
            kind = "jsonl"
        else:
            raise ValueError(
                f"cannot infer format for {source}; set data.format to parquet/csv/jsonl"
            )
    if kind == "parquet":
        frame = pd.read_parquet(source)
    elif kind == "csv":
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(source, sep=sep, low_memory=False)
    elif kind == "jsonl":
        frame = pd.read_json(source, lines=True)
    else:  # pragma: no cover - schema guard
        raise ValueError(f"unsupported format: {kind}")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing column(s): {', '.join(missing)}; "
            f"found: {', '.join(map(str, frame.columns))}"
        )
    return frame.loc[:, columns].copy()


def _select_catalog(
    jan: pd.DataFrame, max_items: int | None, min_item_users: int
) -> tuple[list[str], pd.DataFrame]:
    support = jan.groupby("item").agg(
        users=("user", "nunique"),
        weight=("weight", "sum"),
    )
    support = support[support["users"] >= min_item_users]
    support = support.sort_values(["users", "weight", "item"], ascending=[False, False, True])
    if max_items is not None:
        support = support.head(max_items)
    return list(support.index), support.reset_index()


def _transform_weights(
    values: np.ndarray,
    user_codes: np.ndarray,
    *,
    method: str,
    clip_value: float,
) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64, copy=False), 0.0, clip_value)
    if method == "binary":
        return np.ones_like(clipped, dtype=np.float32)
    if method == "raw":
        return clipped.astype(np.float32)
    if method == "log1p":
        return np.log1p(clipped).astype(np.float32)
    if method == "share":
        totals = np.bincount(user_codes, weights=clipped, minlength=int(user_codes.max()) + 1)
        totals = np.maximum(totals, 1e-12)
        return (clipped / totals[user_codes]).astype(np.float32)
    raise ValueError(f"unsupported weight_transform: {method}")


def _build_csr(
    frame: pd.DataFrame,
    user_index: dict[str, int],
    item_index: dict[str, int],
    *,
    weight_transform: str,
    clip_value: float,
    binary: bool,
) -> sparse.csr_matrix:
    rows = frame["user"].map(user_index).to_numpy()
    cols = frame["item"].map(item_index).to_numpy()
    if np.isnan(rows).any() or np.isnan(cols).any():
        raise AssertionError("mapping produced NaN indices")
    rows = rows.astype(np.int32, copy=False)
    cols = cols.astype(np.int32, copy=False)
    values = frame["weight"].to_numpy()
    if binary:
        data = np.ones(len(frame), dtype=np.float32)
    else:
        data = _transform_weights(
            values,
            rows,
            method=weight_transform,
            clip_value=clip_value,
        )
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(user_index), len(item_index)), dtype=np.float32
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def build_dataset(config: DataConfig, *, verbose: bool = False) -> Dataset:
    """Execute the full train-period -> test-period benchmark construction."""

    weight_column = config.weight_col or "__weight__"
    read_columns = [config.user_col, config.item_col]
    if config.weight_col is not None:
        read_columns.append(config.weight_col)
    if config.time_col is not None:
        read_columns.append(config.time_col)

    train_raw = _read_table(config.train_file, read_columns, config.format)
    test_raw = _read_table(config.test_file, read_columns, config.format)
    if config.weight_col is None:
        train_raw["__weight__"] = 1.0
        test_raw["__weight__"] = 1.0

    if verbose:
        domain = config.domain
        print(
            f"domain: {domain.name} ({domain.item_label_plural}, weights={config.weight_transform})"
        )
        print(f"raw train rows: {len(train_raw):,}")
        print(f"raw test rows : {len(test_raw):,}")

    jan, jan_quality = clean_interactions(
        train_raw,
        user_col=config.user_col,
        item_col=config.item_col,
        weight_col=weight_column,
        normalize_names=config.normalize_names,
        drop_empty=config.drop_empty,
    )
    feb, feb_quality = clean_interactions(
        test_raw,
        user_col=config.user_col,
        item_col=config.item_col,
        weight_col=weight_column,
        normalize_names=config.normalize_names,
        drop_empty=config.drop_empty,
    )
    del train_raw, test_raw

    # The catalogue and clipping thresholds are estimated on January only.
    catalog, catalog_table = _select_catalog(jan, config.max_items, config.min_item_users)
    catalog_set = set(catalog)
    jan_catalog = jan[jan["item"].isin(catalog_set)].copy()
    clip_value = float(jan_catalog["weight"].quantile(config.clip_quantile))
    clip_value = max(clip_value, 1.0)

    # Users need at least ``min_user_items`` distinct catalogue items in January.
    user_support = jan_catalog.groupby("user")["item"].nunique()
    eligible_users = set(user_support[user_support >= config.min_user_items].index)
    jan_catalog = jan_catalog[jan_catalog["user"].isin(eligible_users)].copy()

    user_ids = sorted(eligible_users)
    item_ids = sorted(catalog_set)
    user_index = {user: idx for idx, user in enumerate(user_ids)}
    item_index = {item: idx for idx, item in enumerate(item_ids)}

    train = _build_csr(
        jan_catalog,
        user_index,
        item_index,
        weight_transform=config.weight_transform,
        clip_value=clip_value,
        binary=False,
    )
    if train.nnz == 0:
        raise ValueError("no training interactions remain after filtering")

    # February labels: catalogue items not used by the same user in January.
    seen = set(zip(jan_catalog["user"], jan_catalog["item"], strict=False))
    feb_common = feb[feb["user"].isin(user_index) & feb["item"].isin(item_index)].copy()
    feb_common["seen_in_jan"] = [
        (user, item) in seen
        for user, item in zip(feb_common["user"], feb_common["item"], strict=False)
    ]
    targets = feb_common[~feb_common["seen_in_jan"]].copy()
    if targets.empty:
        raise ValueError("no future adoption labels remain after filtering")
    validate_no_leakage(jan_catalog, targets)

    test = _build_csr(
        targets,
        user_index,
        item_index,
        weight_transform="raw",
        clip_value=clip_value,
        binary=True,
    )

    # Counters used by the report and API metadata.  A cold-start user is one
    # with no January history at all; an ineligible user has January history but
    # was excluded by the minimum-history rule.  Keeping these separate avoids
    # overstating the cold-start population.
    feb_users = set(feb["user"])
    january_users = set(jan["user"])
    cold_users = len(feb_users - january_users)
    ineligible_users = len((feb_users & january_users) - set(user_index))
    warm_users = len(feb_users & set(user_index))
    common_feb = feb[feb["user"].isin(user_index)]
    catalog_coverage = float(common_feb["item"].isin(item_index).mean()) if len(common_feb) else 0.0
    target_counts = np.diff(test.indptr)
    eval_counts = target_counts[target_counts >= 1]
    popularity = np.asarray(train.getnnz(axis=0)).ravel().astype(np.int64)
    weight_sum = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    history_counts = np.diff(train.indptr)
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
    target_hist, _ = np.histogram(target_counts[target_counts > 0], bins=bins)
    history_hist, _ = np.histogram(history_counts, bins=bins)
    popularity_quantiles = np.quantile(popularity, [0.5, 0.75, 0.9, 0.95, 0.99]).tolist()

    domain_payload = config.domain.model_dump(mode="json")
    stats: dict[str, Any] = {
        "domain": domain_payload,
        "dataset_name": config.domain.dataset_name,
        "task": config.domain.task_name,
        "months": {
            "train": config.domain.train_period_label,
            "test": config.domain.test_period_label,
        },
        "periods": {
            "train": config.domain.train_period_label,
            "test": config.domain.test_period_label,
        },
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "n_train_interactions": int(train.nnz),
        "n_test_interactions": int(test.nnz),
        "density": float(train.nnz / (train.shape[0] * train.shape[1])),
        "eval_users": len(eval_counts),
        "eval_users_min_test_items": 1,
        "mean_targets_per_eval_user": float(eval_counts.mean()) if len(eval_counts) else 0.0,
        "median_targets_per_eval_user": float(np.median(eval_counts)) if len(eval_counts) else 0.0,
        "february_users": len(feb_users),
        "test_period_users": len(feb_users),
        "cold_start_february_users": int(cold_users),
        "cold_start_test_users": int(cold_users),
        "ineligible_test_users": int(ineligible_users),
        "warm_test_users": int(warm_users),
        "catalog_coverage_of_february_rows": catalog_coverage,
        "catalog_coverage_of_test_rows": catalog_coverage,
        "catalog_items_total_january": int(jan["item"].nunique()),
        "catalog_items_total_train": int(jan["item"].nunique()),
        "catalog_item_support_min": int(config.min_item_users),
        "catalog_size_cap": config.max_items,
        "weight_transform": config.weight_transform,
        "weight_clip_value": clip_value,
        "weight_unit_declared": config.weight_unit,
        "train_weight_sum": float(weight_sum.sum()),
        "item_popularity_top10": [
            {"item": item_ids[i], "users": int(popularity[i])} for i in np.argsort(-popularity)[:10]
        ],
        "target_count_histogram": {
            "bins": bins,
            "counts": [int(x) for x in target_hist],
        },
        "history_count_histogram": {
            "bins": bins,
            "counts": [int(x) for x in history_hist],
        },
        "item_popularity_quantiles": {
            "p50": float(popularity_quantiles[0]),
            "p75": float(popularity_quantiles[1]),
            "p90": float(popularity_quantiles[2]),
            "p95": float(popularity_quantiles[3]),
            "p99": float(popularity_quantiles[4]),
        },
    }

    quality = {
        "train": jan_quality.as_dict(),
        "test": feb_quality.as_dict(),
        "january": jan_quality.as_dict(),
        "february": feb_quality.as_dict(),
        "catalog": {
            "selected": len(catalog_table),
            "top_by_users": catalog_table.head(5).to_dict(orient="records"),
        },
    }

    # The fingerprint identifies the data and the processing decisions, not the
    # filesystem location or display vocabulary. Moving a file or renaming a
    # domain label must not invalidate a reproducible dataset.
    fingerprint_config = config.model_dump(mode="json")
    for metadata_key in ("jan_path", "feb_path", "train_path", "test_path", "domain", "weight_unit"):
        fingerprint_config.pop(metadata_key, None)
    fingerprint_payload = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "train_nnz": int(train.nnz),
        "test_nnz": int(test.nnz),
        "config": fingerprint_config,
        "train_sha256": _sha256(config.train_file),
        "test_sha256": _sha256(config.test_file),
        "jan_sha256": _sha256(config.train_file),
        "feb_sha256": _sha256(config.test_file),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]

    return Dataset(
        train=train,
        test=test,
        user_ids=user_ids,
        item_ids=item_ids,
        stats=stats,
        config=config.model_dump(mode="json"),
        quality=quality,
        fingerprint=fingerprint,
    )


def dataset_summary(dataset: Dataset) -> str:
    """Human-readable one-screen summary used by ``engagegraph data stats``."""

    stats = dataset.stats
    domain = stats.get("domain", {})
    user_label = domain.get("user_label_plural", "users")
    item_label = domain.get("item_label_plural", "items")
    lines = [
        f"Dataset fingerprint : {dataset.fingerprint}",
        f"Domain              : {domain.get('name', 'generic')} ({item_label})",
        f"Users x items       : {dataset.n_users:,} x {dataset.n_items:,}",
        f"Train interactions  : {stats['n_train_interactions']:,} (density {stats['density']:.2%})",
        f"Test targets        : {stats['n_test_interactions']:,} across "
        f"{stats['eval_users']:,} users",
        f"Targets per user    : mean {stats['mean_targets_per_eval_user']:.1f}, "
        f"median {stats['median_targets_per_eval_user']:.0f}",
        f"Warm test users     : {stats.get('warm_test_users', 0):,}",
        f"Cold-start users    : {stats.get('cold_start_test_users', stats.get('cold_start_february_users', 0)):,}",
        f"Ineligible users    : {stats.get('ineligible_test_users', 0):,} "
        f"(< {stats.get('catalog_item_support_min', 10)} catalogue items)",
        f"Catalogue coverage  : {stats.get('catalog_coverage_of_test_rows', stats.get('catalog_coverage_of_february_rows', 0)):.1%} of "
        f"{domain.get('test_period_label', 'test period')} rows for warm {user_label}",
    ]
    return "\n".join(lines)
