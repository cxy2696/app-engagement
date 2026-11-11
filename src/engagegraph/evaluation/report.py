"""Generate a human-readable evaluation report and figures from benchmark artefacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engagegraph.evaluation.bootstrap import bootstrap_ci
from engagegraph.evaluation.metrics import EvalResult


def _load_results(summary: dict[str, Any]) -> dict[str, EvalResult]:
    results: dict[str, EvalResult] = {}
    for name, record in summary.get("models", {}).items():
        run_dir = Path(record["run_dir"]) / "evaluation"
        if run_dir.exists():
            results[name] = EvalResult.load(run_dir)
    return results


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _table(results: dict[str, EvalResult], primary_k: int) -> str:
    header = (
        "| Model | NDCG@10 (95% CI) | Recall@10 | MAP@10 | HitRate@10 | "
        "Coverage@10 | Novelty@10 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = []
    for name, result in sorted(
        results.items(), key=lambda kv: -kv[1].metrics.get(f"ndcg@{primary_k}", 0.0)
    ):
        ci = bootstrap_ci(
            result.per_user[f"ndcg@{primary_k}"],
            n_boot=1000,
            seed=42,
        )
        lines.append(
            f"| `{name}` | {_fmt(result.metrics[f'ndcg@{primary_k}'])} "
            f"[{_fmt(ci.ci_low)}, {_fmt(ci.ci_high)}] | "
            f"{_fmt(result.metrics['recall@10'])} | {_fmt(result.metrics['map@10'])} | "
            f"{_fmt(result.metrics['hit_rate@10'])} | "
            f"{_fmt(result.metrics['coverage@10'], 3)} | "
            f"{_fmt(result.metrics['novelty@10'], 3)} |"
        )
    return header + "\n".join(lines)


def _comparison_table(summary: dict[str, Any]) -> str:
    comparisons = summary.get("comparisons", [])
    if not comparisons:
        return "_No comparisons were generated._\n"
    header = (
        "| Candidate vs. baseline | Δ NDCG@10 | 95% CI | p (bootstrap) | p (Holm) |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for record in comparisons:
        rows.append(
            f"| `{record['candidate']}` vs `{record['baseline']}` | "
            f"{record['difference']:+.4f} | "
            f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}] | "
            f"{record['p_value']:.4f} | {record.get('p_holm', float('nan')):.4f} |"
        )
    return header + "\n".join(rows)


def _per_k_table(results: dict[str, EvalResult]) -> str:
    k_values = sorted({k for result in results.values() for k in result.k_values})
    header = "| Model | " + " | ".join(f"NDCG@{k} | Recall@{k}" for k in k_values) + " |\n"
    header += "| --- | " + " | ".join("---: | ---:" for _ in k_values) + " |\n"
    rows = []
    for name, result in results.items():
        cells = []
        for k in k_values:
            cells.append(_fmt(result.metrics.get(f"ndcg@{k}", float("nan"))))
            cells.append(_fmt(result.metrics.get(f"recall@{k}", float("nan"))))
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return header + "\n".join(rows)


def _make_figures(
    results: dict[str, EvalResult], summary: dict[str, Any], out_dir: Path
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = ["#4f46e5", "#0f766e", "#e05d44", "#7c3aed", "#b45309", "#0369a1"]
    names = sorted(
        results,
        key=lambda name: -results[name].metrics.get("ndcg@10", 0.0),
    )

    for metric in ("ndcg", "recall"):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for index, name in enumerate(names):
            result = results[name]
            ks = sorted(result.k_values)
            values = [result.metrics[f"{metric}@{k}"] for k in ks]
            ax.plot(ks, values, marker="o", label=name, color=colors[index % len(colors)])
        ax.set_xlabel("K")
        ax.set_ylabel(metric.upper() + "@K")
        ax.set_title(f"{metric.upper()}@K by model")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / f"{metric}_at_k.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for index, name in enumerate(names):
        result = results[name]
        ax.scatter(
            result.metrics["coverage@10"],
            result.metrics["ndcg@10"],
            s=90,
            color=colors[index % len(colors)],
            label=name,
            zorder=3,
        )
        ax.annotate(
            name,
            (result.metrics["coverage@10"], result.metrics["ndcg@10"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    ax.set_xlabel("Coverage@10")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. catalogue coverage")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "accuracy_vs_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    stats = summary.get("dataset_stats", {})
    histogram = stats.get("target_count_histogram")
    if histogram:
        bins = histogram["bins"]
        counts = histogram["counts"]
        labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(counts))]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.bar(labels, counts, color="#4f46e5", alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("February adoptions per user (bin lower bound)")
        ax.set_ylabel("Users (log scale)")
        ax.set_title("Distribution of evaluation targets")
        fig.tight_layout()
        path = out_dir / "target_distribution.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    popularity = stats.get("item_popularity_quantiles")
    if popularity:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        labels = ["p50", "p75", "p90", "p95", "p99"]
        values = [popularity[label] for label in labels]
        ax.bar(labels, values, color="#0f766e", alpha=0.9)
        ax.set_ylabel("January users per app")
        ax.set_title("Catalogue support quantiles")
        fig.tight_layout()
        path = out_dir / "popularity_quantiles.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def build_report(
    benchmark_path: str | Path = "artifacts/runs/benchmark.json",
    out_dir: str | Path = "reports",
    *,
    figures: bool = True,
) -> list[Path]:
    benchmark_path = Path(benchmark_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"benchmark summary not found: {benchmark_path}; run `engagegraph benchmark` first"
        )
    summary = json.loads(benchmark_path.read_text(encoding="utf-8"))
    results = _load_results(summary)
    if not results:
        raise ValueError("benchmark summary contains no loadable evaluations")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_k = int(summary.get("primary_metric", "ndcg@10").split("@")[1])
    dataset = summary.get("dataset_stats", {})
    domain = dataset.get("domain", {}) or {}
    item_label = domain.get("item_label_plural", "items")
    user_label = domain.get("user_label_plural", "users")
    test_label = domain.get("test_period_label", "test period")
    cold_users = dataset.get("cold_start_test_users", dataset.get("cold_start_february_users", 0))
    comparison = _comparison_table(summary)
    table = _table(results, primary_k)
    per_k = _per_k_table(results)

    lines = [
        "# EngageGraph evaluation report",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}  ",
        f"Dataset fingerprint: `{summary.get('dataset_fingerprint')}`  ",
        f"Package: `{summary.get('package_version')}`  ",
        f"Domain: `{domain.get('name', 'generic')}`  ",
        f"Primary metric: **NDCG@{primary_k}**",
        "",
        "## Headline",
        "",
        table,
        "",
        "Coverage and novelty are included because accuracy alone can hide a degenerate "
        "popularity strategy. Confidence intervals are paired user-level bootstrap "
        "intervals over evaluation users (1,000 resamples, 95%).",
        "",
        "## Paired comparisons",
        "",
        comparison,
        "",
        f"`n = {dataset.get('eval_users', 0):,}` evaluation {user_label}; "
        f"`{cold_users:,}` {test_label} cold-start {user_label} are reported separately "
        "and are not in the personalised denominator."
        if isinstance(dataset.get("eval_users"), int)
        else "Cold-start counts are reported in the dataset manifest.",
        "",
        "## Per-K metrics",
        "",
        per_k,
        "",
        "## Dataset",
        "",
        f"- {user_label.title()} x {item_label}: **{dataset.get('n_users', 0):,} x {dataset.get('n_items', 0):,}**",
        f"- Train interactions: **{dataset.get('n_train_interactions', 0):,}**",
        f"- Test targets: **{dataset.get('n_test_interactions', 0):,}**",
        f"- Targets per evaluation user: mean **{dataset.get('mean_targets_per_eval_user', 0):.1f}**, "
        f"median **{dataset.get('median_targets_per_eval_user', 0):.0f}**",
        f"- Catalogue coverage of {test_label} rows (warm {user_label}): "
        f"**{dataset.get('catalog_coverage_of_test_rows', dataset.get('catalog_coverage_of_february_rows', 0)):.1%}**",
        f"- Weight transform: `{dataset.get('weight_transform', '—')}`; "
        f"clip: `{dataset.get('weight_clip_value', 0):,.0f}`",
        "",
        "## Protocol recap",
        "",
        "January is used for all fitting; February only defines binary adoption labels for "
        "apps the user did not use in January. The candidate set is the full January "
        "catalogue, and all January apps are masked at scoring time. Metrics are "
        "macro-averaged over users. Hyperparameters were fixed before the February "
        "evaluation. Full details: [`docs/methodology.md`](../docs/methodology.md).",
        "",
        "## Reading the result",
        "",
    ]

    if summary.get("comparisons"):
        strongest = max(
            summary["comparisons"],
            key=lambda record: record.get("difference", float("-inf")),
        )
        if strongest.get("p_holm", 1.0) < 0.05 and strongest.get("difference", 0) > 0:
            lines.append(
                f"The largest positive paired difference is `{strongest['candidate']}` "
                f"over `{strongest['baseline']}`: "
                f"**{strongest['difference']:+.4f} NDCG@{primary_k}** "
                f"(Holm p = {strongest['p_holm']:.4f}). The interval excludes zero, so the "
                "difference is credible under this protocol, but it remains specific to this "
                "panel and month pair."
            )
        else:
            lines.append(
                "No model shows a Holm-corrected positive difference over the baseline. "
                "That is a valid result: the simpler baseline is preferred."
            )
    else:
        lines.append("No paired comparisons were included in this benchmark run.")

    lines += [
        "",
        "Coverage differences matter: a model with higher NDCG but near-zero coverage is "
        "recommending the same few apps to everyone. The `accuracy_vs_coverage` figure makes "
        "that trade-off explicit.",
        "",
        "## Limitations",
        "",
        f"- One panel, one {domain.get('train_period_label', 'train')} -> "
        f"{test_label} transition; no claim of generalisation to other markets.",
        "- Binary relevance; graded engagement is out of scope for the primary metric.",
        "- Offline evaluation; no position bias or online A/B evidence.",
        "- The data licence is unresolved; see [`DATA_LICENSE.md`](../DATA_LICENSE.md).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "engagegraph data build --out data/processed",
        "engagegraph benchmark --config configs/benchmark.yaml",
        "engagegraph report --benchmark artifacts/runs/benchmark.json --out reports",
        "```",
        "",
        "The legacy AUC audit is in [`legacy-audit.md`](legacy-audit.md).",
        "",
    ]
    report_path = out_dir / "evaluation-report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    figure_paths: list[Path] = []
    if figures:
        figure_paths = _make_figures(results, summary, out_dir / "figures")
        if figure_paths:
            with report_path.open("a", encoding="utf-8") as handle:
                handle.write("\n## Figures\n\n")
                for path in figure_paths:
                    handle.write(f"![{path.stem}](figures/{path.name})\n\n")

    results_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": summary.get("dataset_fingerprint"),
        "primary_metric": summary.get("primary_metric"),
        "models": {
            name: {
                "metrics": result.metrics,
                "coverage": result.coverage,
                "n_eval_users": result.n_eval_users,
                "runtime_sec": result.runtime_sec,
            }
            for name, result in results.items()
        },
        "comparisons": summary.get("comparisons", []),
        "dataset": dataset,
    }
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results_payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return [report_path, results_path, *figure_paths]
