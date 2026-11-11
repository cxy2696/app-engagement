"""Command-line interface.

The CLI is the supported operational surface: every documented workflow has a
command, a config file, and a machine-readable output option.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from engagegraph import __version__
from engagegraph.artifacts import ModelRegistry, load_artifact
from engagegraph.config import DataConfig, load_benchmark_config, load_experiment_config
from engagegraph.data.preprocess import Dataset, build_dataset, dataset_summary
from engagegraph.evaluation.metrics import evaluate, format_metrics
from engagegraph.pipeline import run_benchmark, run_experiment
from engagegraph.recommend import RecommendationService


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _data_config(args: argparse.Namespace) -> DataConfig:
    if getattr(args, "config", None):
        import yaml

        payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("data config must be a YAML mapping")
        # Accept either a standalone DataConfig or a full Benchmark/Experiment
        # profile; the latter is convenient because it also carries the domain.
        config = DataConfig.model_validate(payload.get("data", payload))
    else:
        config = DataConfig()
    overrides = {
        "train_path": args.train,
        "test_path": args.test,
        "format": args.format,
        "min_item_users": args.min_item_users,
        "min_user_items": args.min_user_items,
        "max_items": args.max_items,
        "weight_transform": args.weight_transform,
        "weight_unit": args.weight_unit,
        "clip_quantile": args.clip_quantile,
    }
    # ``None`` means "keep the value from the config/default"; argparse uses
    # explicit sentinels for every numeric override.
    update = {key: value for key, value in overrides.items() if value is not None}
    return config.model_copy(update=update) if update else config


def cmd_data_build(args: argparse.Namespace) -> int:
    config = _data_config(args)
    dataset = build_dataset(config, verbose=True)
    target = Path(args.out)
    dataset.save(target)
    (target / "resolved-data-config.yaml").write_text(
        __import__("yaml").safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    print(dataset_summary(dataset))
    print(f"\nSaved processed dataset to {target}")
    return 0


def cmd_data_stats(args: argparse.Namespace) -> int:
    dataset = Dataset.load(args.dataset)
    if args.json:
        _print_json(dataset.stats)
    else:
        print(dataset_summary(dataset))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    output = run_experiment(config)
    print(f"\nRun directory: {output.run_dir}")
    print(f"Artifact     : {output.artifact.path}")
    print(f"Train time   : {output.train_seconds:.1f}s")
    if args.json:
        _print_json(output.evaluation.metrics)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = load_benchmark_config(args.config)
    summary = run_benchmark(config)
    if args.json:
        _print_json(summary)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    dataset = Dataset.load(args.dataset)
    artifact = load_artifact(args.artifact)
    if artifact.manifest.get("dataset", {}).get("fingerprint") != dataset.fingerprint:
        raise ValueError("artifact and dataset fingerprints do not match")
    result = evaluate(
        artifact.model,
        dataset,
        k_values=tuple(args.k),
        primary_k=args.k[0],
        batch_size=args.batch_size,
    )
    if args.out:
        result.save(args.out)
    if args.json:
        _print_json(result.metrics)
    else:
        print(format_metrics(result))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.artifact)
    service = RecommendationService(artifact)
    response = service.recommend(
        args.user, k=args.k, explain=not args.no_explain, diversify=args.diversify
    )
    if args.json:
        _print_json(response.as_dict())
    else:
        print(f"User {args.user} ({response.strategy}, {response.model})")
        for item in response.items:
            reasons = "; ".join(e.reason for e in item.explanations)
            print(f"  {item.rank:>2}. {item.item_id:<40} score={item.score:8.4f}  {reasons}")
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    artifact = load_artifact(args.artifact)
    service = RecommendationService(artifact)
    items = service.similar(args.item, k=args.k)
    if args.json:
        _print_json([item.as_dict() for item in items])
    else:
        print(f"Apps similar to {args.item}:")
        for item in items:
            print(f"  {item.rank:>2}. {item.item_id:<40} score={item.score:.4f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    artifact = args.artifact or os.environ.get("ENGAGEGRAPH_ARTIFACT")
    if artifact:
        os.environ["ENGAGEGRAPH_ARTIFACT"] = str(Path(artifact).resolve())
    if args.reload:
        uvicorn.run(
            "engagegraph.api.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        from engagegraph.api.app import create_app

        app = create_app(artifact)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from engagegraph.evaluation.report import build_report

    paths = build_report(args.benchmark, out_dir=args.out, figures=not args.no_figures)
    for path in paths:
        print(path)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("python", sys.version_info >= (3, 10), sys.version.split()[0]))
    for module in ("numpy", "pandas", "scipy", "sklearn", "fastapi"):
        try:
            imported = __import__(module)
            checks.append((module, True, getattr(imported, "__version__", "installed")))
        except ImportError as exc:
            checks.append((module, False, str(exc)))
    for optional in ("implicit", "torch", "torch_geometric"):
        try:
            imported = __import__(optional)
            checks.append((optional, True, getattr(imported, "__version__", "installed")))
        except ImportError:
            checks.append((optional, False, "optional, not installed"))
    data_ok = Path("data/raw/jan_aggregated.parquet").exists()
    checks.append(("legacy data files", data_ok, "data/raw/"))
    try:
        artifact = ModelRegistry().resolve()
        checks.append(("active artefact", True, str(artifact.path)))
    except FileNotFoundError:
        checks.append(("active artefact", False, "run `engagegraph benchmark` first"))
    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        print(f"{'OK ' if ok else 'WARN'}  {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 0 if failed == 0 else 1


def cmd_version(_: argparse.Namespace) -> int:
    print(f"engagegraph {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engagegraph",
        description=(
            "Reproducible temporal engagement analytics and next-item recommendation. "
            "Domain labels are configured; the engine is not app-specific."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="build and inspect processed datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build", help="aggregate interaction exports into a benchmark")
    build.add_argument("--config", default=None, help="optional DataConfig YAML")
    build.add_argument("--train", "--jan", dest="train", default=None)
    build.add_argument("--test", "--feb", dest="test", default=None)
    build.add_argument("--out", default="data/processed")
    build.add_argument("--format", choices=["auto", "parquet", "csv", "jsonl"], default=None)
    build.add_argument("--min-item-users", type=int, default=None)
    build.add_argument("--min-user-items", type=int, default=None)
    build.add_argument("--max-items", type=int, default=None)
    build.add_argument(
        "--weight-transform", choices=["log1p", "binary", "share", "raw"], default=None
    )
    build.add_argument("--weight-unit", choices=["ms", "s", "count", "unknown"], default=None)
    build.add_argument("--clip-quantile", type=float, default=None)
    build.set_defaults(func=cmd_data_build)

    stats = data_sub.add_parser("stats", help="print dataset statistics")
    stats.add_argument("--dataset", default="data/processed")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=cmd_data_stats)

    train = sub.add_parser("train", help="fit one model from a config")
    train.add_argument("--config", required=True)
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=cmd_train)

    benchmark = sub.add_parser("benchmark", help="fit and compare all configured models")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--json", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    evaluate_cmd = sub.add_parser("evaluate", help="evaluate a saved artifact")
    evaluate_cmd.add_argument("--artifact", required=True)
    evaluate_cmd.add_argument("--dataset", default="data/processed")
    evaluate_cmd.add_argument("--k", type=int, nargs="+", default=[10, 20, 50])
    evaluate_cmd.add_argument("--batch-size", type=int, default=512)
    evaluate_cmd.add_argument("--out", default=None)
    evaluate_cmd.add_argument("--json", action="store_true")
    evaluate_cmd.set_defaults(func=cmd_evaluate)

    recommend = sub.add_parser("recommend", help="recommend items for one user")
    recommend.add_argument("--artifact", required=True)
    recommend.add_argument("--user", required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--no-explain", action="store_true")
    recommend.add_argument("--diversify", action="store_true")
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=cmd_recommend)

    similar = sub.add_parser("similar", help="find similar items")
    similar.add_argument("--artifact", required=True)
    similar.add_argument("--item", required=True)
    similar.add_argument("--k", type=int, default=10)
    similar.add_argument("--json", action="store_true")
    similar.set_defaults(func=cmd_similar)

    serve = sub.add_parser("serve", help="run the HTTP API and dashboard")
    serve.add_argument("--artifact", default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    report = sub.add_parser("report", help="build a Markdown/PNG evaluation report")
    report.add_argument("--benchmark", default="artifacts/runs/benchmark.json")
    report.add_argument("--out", default="reports")
    report.add_argument("--no-figures", action="store_true")
    report.set_defaults(func=cmd_report)

    doctor = sub.add_parser("doctor", help="check the local environment")
    doctor.set_defaults(func=cmd_doctor)

    version = sub.add_parser("version", help="print the package version")
    version.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("ENGAGEGRAPH_DEBUG"):
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
