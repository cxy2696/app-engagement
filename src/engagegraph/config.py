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


class DataConfig(StrictModel):
    """How raw monthly interaction exports are converted into a benchmark."""

    jan_path: str = "Jan-Feb-parquet/jan_aggregated.parquet"
    feb_path: str = "Jan-Feb-parquet/feb_aggregated.parquet"
    user_col: str = "PANELISTID"
    item_col: str = "APPDESCRIPTION"
    weight_col: str = "weight"
    weight_unit: Literal["ms", "s", "unknown"] = "ms"
    min_item_users: int = Field(default=10, ge=1)
    min_user_items: int = Field(default=3, ge=1)
    max_items: int | None = Field(default=10_000, ge=100)
    clip_quantile: float = Field(default=0.999, gt=0.5, lt=1.0)
    weight_transform: Literal["log1p", "binary", "share", "raw"] = "log1p"
    normalize_names: bool = True
    drop_empty: bool = True
    seed: int = 42

    @field_validator("jan_path", "feb_path")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be empty")
        return value

    @model_validator(mode="after")
    def _different_months(self) -> DataConfig:
        if Path(self.jan_path).resolve() == Path(self.feb_path).resolve():
            raise ValueError("jan_path and feb_path must refer to different files")
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
