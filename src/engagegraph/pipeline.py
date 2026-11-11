"""Experiment orchestration: build once, fit, evaluate, persist, compare."""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from engagegraph import __version__
from engagegraph.artifacts import Artifact, ModelRegistry, save_artifact
from engagegraph.config import BenchmarkConfig, EvaluationConfig, ExperimentConfig, ModelConfig
from engagegraph.data.preprocess import Dataset, build_dataset
from engagegraph.evaluation.bootstrap import compare_results, holm_bonferroni
from engagegraph.evaluation.metrics import EvalResult, evaluate, format_metrics
from engagegraph.models.base import BaseRecommender, build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU environment
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class RunOutput:
    model_name: str
    artifact: Artifact
    evaluation: EvalResult
    train_seconds: float
    evaluate_seconds: float
    run_dir: Path
    skipped: bool = False
    reason: str = ""


def _evaluation_kwargs(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "k_values": config.k_values,
        "primary_k": config.primary_k,
        "batch_size": config.eval_batch_size,
        "coverage_k": config.coverage_k,
        "min_test_items": config.min_test_items,
    }


def fit_and_evaluate(
    dataset: Dataset,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    *,
    output_dir: str | Path = "artifacts/runs",
    seed: int = 42,
    run_label: str | None = None,
    save: bool = True,
) -> RunOutput:
    seed_everything(seed)
    model: BaseRecommender = build_model(model_config.name, model_config.params)
    started = time.perf_counter()
    model.fit(dataset)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = evaluate(model, dataset, **_evaluation_kwargs(evaluation_config))
    evaluate_seconds = time.perf_counter() - started

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = run_label or f"{model_config.name}-{dataset.fingerprint[:8]}"
    run_dir = Path(output_dir) / f"{label}-{stamp}"
    artifact = None
    if save:
        artifact = save_artifact(
            model,
            dataset,
            run_dir / "artifact",
            metrics=result.metrics,
            config={"model": model_config.model_dump(mode="json"), "seed": seed},
        )
        result.save(run_dir / "evaluation")
        run_record = {
            "package_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "dataset_fingerprint": dataset.fingerprint,
            "model": model_config.model_dump(mode="json"),
            "train_seconds": train_seconds,
            "evaluate_seconds": evaluate_seconds,
            "metrics": result.metrics,
        }
        (run_dir / "run.json").write_text(
            json.dumps(run_record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    else:
        # Still return a lightweight artefact object for callers that need the
        # fitted model without touching disk (used by tests and notebooks).
        from engagegraph.artifacts import Artifact

        artifact = Artifact(
            model=model,
            user_ids=dataset.user_ids,
            item_ids=dataset.item_ids,
            stats=dataset.stats,
            metrics=result.metrics,
            manifest={"model": model.meta()},
            path=run_dir,
            history=dataset.train,
        )
    return RunOutput(
        model_name=model_config.name,
        artifact=artifact,
        evaluation=result,
        train_seconds=train_seconds,
        evaluate_seconds=evaluate_seconds,
        run_dir=run_dir,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    dataset: Dataset | None = None,
    save_dataset: bool = True,
    output_dir: str | Path | None = None,
) -> RunOutput:
    dataset = dataset or build_dataset(config.data, verbose=True)
    if save_dataset and config.processed_dir:
        dataset.save(config.processed_dir)
    return fit_and_evaluate(
        dataset,
        config.model,
        config.evaluation,
        output_dir=output_dir or config.output_dir,
        seed=config.seed,
        run_label=config.name,
    )


def run_benchmark(
    config: BenchmarkConfig,
    *,
    output_dir: str | Path | None = None,
    save_dataset: bool = True,
) -> dict[str, Any]:
    """Fit every configured model on one dataset and compare against popularity."""

    dataset = build_dataset(config.data, verbose=True)
    if save_dataset and config.processed_dir:
        dataset.save(config.processed_dir)
    base_dir = Path(output_dir or config.output_dir)
    outputs: list[RunOutput] = []
    for model_config in config.models:
        try:
            output = fit_and_evaluate(
                dataset,
                model_config,
                config.evaluation,
                output_dir=base_dir,
                seed=config.seed,
                run_label=model_config.name,
            )
            outputs.append(output)
            print(format_metrics(output.evaluation))
        except Exception as exc:
            print(f"skipping {model_config.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            outputs.append(
                RunOutput(
                    model_name=model_config.name,
                    artifact=None,  # type: ignore[arg-type]
                    evaluation=None,  # type: ignore[arg-type]
                    train_seconds=0.0,
                    evaluate_seconds=0.0,
                    run_dir=base_dir,
                    skipped=True,
                    reason=str(exc),
                )
            )

    completed = [o for o in outputs if not o.skipped]
    if not completed:
        raise RuntimeError("all models failed; nothing to compare")
    results = {o.model_name: o.evaluation for o in completed}
    baseline = "popularity" if "popularity" in results else completed[0].model_name
    comparisons = compare_results(
        results,
        baseline=baseline,
        metric=f"ndcg@{config.evaluation.primary_k}",
        n_boot=config.evaluation.bootstrap_samples,
        seed=config.seed,
    )
    adjusted = holm_bonferroni({c.candidate: c.p_value for c in comparisons})
    comparison_records = [{**c.as_dict(), "p_holm": adjusted.get(c.candidate)} for c in comparisons]
    (base_dir / "comparisons.json").write_text(
        json.dumps(comparison_records, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    primary_metric = f"ndcg@{config.evaluation.primary_k}"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_stats": dataset.stats,
        "primary_metric": primary_metric,
        "models": {
            o.model_name: {
                "metrics": o.evaluation.metrics,
                "train_seconds": o.train_seconds,
                "evaluate_seconds": o.evaluate_seconds,
                "run_dir": str(o.run_dir),
                "artifact_fingerprint": o.artifact.manifest.get("artifact_fingerprint"),
            }
            for o in completed
        },
        "skipped": [{"model": o.model_name, "reason": o.reason} for o in outputs if o.skipped],
        "comparisons": comparison_records,
        "baseline": baseline,
    }
    summary_path = base_dir / "benchmark.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    best = max(completed, key=lambda o: o.evaluation.metrics[primary_metric])
    registry = ModelRegistry(base_dir.parent if base_dir.name == "runs" else base_dir)
    registry.register(best.artifact, make_active=True)
    print(
        f"\nBest model: {best.model_name} ({primary_metric}="
        f"{best.evaluation.metrics[primary_metric]:.4f})"
    )
    print(f"Benchmark summary: {summary_path}")
    return summary
