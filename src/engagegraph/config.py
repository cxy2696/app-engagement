"""Typed configuration objects and YAML loading.

All experiment inputs are explicit and serialisable so that a run can be
reproduced from the config file alone.  The config schema is part of the
public contract: adding a field is backwards compatible, removing one is not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DomainConfig(StrictModel):
    """Human-facing vocabulary for a deployment domain.

    The model, matrices, and metrics are domain-agnostic.  This object only
    controls labels used by the API/dashboard and the optional serving blocklist.
    Keep it in the dataset config so an artefact carries its own vocabulary.
    """

    name: str = "generic"
    dataset_name: str = "interaction panel"
    user_label: str = "user"
    user_label_plural: str = "users"
    item_label: str = "item"
    item_label_plural: str = "items"
    interaction_label: str = "interaction"
    catalog_label: str = "catalogue"
    task_name: str = "next-period new-item adoption ranking"
    train_period_label: str = "train period"
    test_period_label: str = "test period"
    tagline: str = (
        "A reproducible pipeline for leakage-free temporal splits, reference baselines, "
        "and auditable next-item recommendations."
    )
    blocklist: tuple[str, ...] = ()

    @field_validator(
        "name",
        "dataset_name",
        "user_label",
        "user_label_plural",
        "item_label",
        "item_label_plural",
        "interaction_label",
        "catalog_label",
        "task_name",
        "train_period_label",
        "test_period_label",
        "tagline",
    )
    @classmethod
    def _non_empty_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("domain labels must not be empty")
        return value.strip()


class DataConfig(StrictModel):
    """How raw interaction exports are converted into a benchmark.

    Two equivalent path styles are accepted: ``train_path``/``test_path`` (the
    generic API) and the historical ``jan_path``/``feb_path`` aliases.  The
    generic style wins when both are present.
    """

    jan_path: str = "Jan-Feb-parquet/jan_aggregated.parquet"
    feb_path: str = "Jan-Feb-parquet/feb_aggregated.parquet"
    train_path: str | None = None
    test_path: str | None = None
    format: Literal["auto", "parquet", "csv", "jsonl"] = "auto"
    user_col: str = "PANELISTID"
    item_col: str = "APPDESCRIPTION"
    weight_col: str | None = "weight"
    time_col: str | None = None
    weight_unit: Literal["ms", "s", "count", "unknown"] = "ms"
    min_item_users: int = Field(default=10, ge=1)
    min_user_items: int = Field(default=3, ge=1)
    max_items: int | None = Field(default=10_000, ge=100)
    clip_quantile: float = Field(default=0.999, gt=0.5, lt=1.0)
    weight_transform: Literal["log1p", "binary", "share", "raw"] = "log1p"
    normalize_names: bool = True
    drop_empty: bool = True
    seed: int = 42
    domain: DomainConfig = Field(default_factory=DomainConfig)

    @property
    def train_file(self) -> str:
        return self.train_path or self.jan_path

    @property
    def test_file(self) -> str:
        return self.test_path or self.feb_path

    @model_validator(mode="after")
    def _different_periods(self) -> DataConfig:
        if not self.train_file.strip() or not self.test_file.strip():
            raise ValueError("train/test paths must not be empty")
        if Path(self.train_file).resolve() == Path(self.test_file).resolve():
            raise ValueError("train/test paths must refer to different files")
        if self.weight_col is not None and not self.weight_col.strip():
            raise ValueError("weight_col must be a column name or null")
        return self


class ModelConfig(StrictModel):
    """Model selection and hyper-parameters.

    ``params`` is deliberately free-form: each model documents the keys it
    accepts.  Unknown keys are rejected by the model constructor, not silently
    ignored, so a typo cannot masquerade as a default.
    """

    name: Literal["popularity", "itemknn", "als", "metapath2vec"]
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(StrictModel):
    k_values: tuple[int, ...] = (10, 20, 50)
    primary_k: int = 10
    bootstrap_samples: int = Field(default=1000, ge=0, le=20_000)
    confidence: float = Field(default=0.95, gt=0.5, lt=1.0)
    eval_batch_size: int = Field(default=512, ge=1)
    min_test_items: int = Field(default=1, ge=1)
    coverage_k: int = 10
    seed: int = 42

    @field_validator("k_values")
    @classmethod
    def _sorted_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(k <= 0 for k in value):
            raise ValueError("k_values must be positive")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def _primary_in_k(self) -> EvaluationConfig:
        if self.primary_k not in self.k_values:
            raise ValueError("primary_k must be present in k_values")
        return self


class ExperimentConfig(StrictModel):
    name: str = "experiment"
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: str = "artifacts/runs"
    processed_dir: str | None = "data/processed"

    def fingerprint(self) -> str:
        """Stable SHA-256 of the resolved, JSON-serialisable config."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class BenchmarkConfig(StrictModel):
    name: str = "benchmark"
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    models: list[ModelConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output_dir: str = "artifacts/runs"
    processed_dir: str | None = "data/processed"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_read_yaml(path))


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(_read_yaml(path))


def dump_config(config: BaseModel, path: str | Path) -> None:
    """Write *config* as canonical YAML (useful for run provenance)."""
    data = config.model_dump(mode="json")
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
