#!/usr/bin/env python
"""Inject the latest benchmark table into README and the model card."""

from __future__ import annotations

import json
from pathlib import Path

from engagegraph.evaluation.bootstrap import bootstrap_ci
from engagegraph.evaluation.metrics import EvalResult

START = "<!-- RESULTS_TABLE_START -->"
END = "<!-- RESULTS_TABLE_END -->"
MODEL_START = "<!-- MODEL_METRICS_START -->"
MODEL_END = "<!-- MODEL_METRICS_END -->"


def table() -> str:
    summary = json.loads(Path("artifacts/runs/benchmark.json").read_text())
    results = {}
    for name, record in summary["models"].items():
        directory = Path(record["run_dir"]) / "evaluation"
        if directory.exists():
            results[name] = EvalResult.load(directory)
    lines = [
        "| Model | NDCG@10 (95% CI) | Recall@10 | MAP@10 | Coverage@10 | Novelty@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in sorted(results.items(), key=lambda kv: -kv[1].metrics["ndcg@10"]):
        ci = bootstrap_ci(result.per_user["ndcg@10"], n_boot=1000, seed=42)
        lines.append(
            f"| `{name}` | **{result.metrics['ndcg@10']:.4f}** "
            f"[{ci.ci_low:.4f}, {ci.ci_high:.4f}] | "
            f"{result.metrics['recall@10']:.4f} | {result.metrics['map@10']:.4f} | "
            f"{result.metrics['coverage@10']:.3f} | {result.metrics['novelty@10']:.3f} |"
        )
    lines += [
        "",
        f"Dataset fingerprint: `{summary['dataset_fingerprint']}` · "
        f"Primary metric: `{summary['primary_metric']}` · "
        f"Evaluation users: {summary['dataset_stats']['eval_users']:,} · "
        f"Cold-start users: {summary['dataset_stats'].get('cold_start_test_users', summary['dataset_stats'].get('cold_start_february_users', 0)):,}. "
        "Paired bootstrap intervals (1,000 resamples) are computed over evaluation users.",
    ]
    return "\n".join(lines)


def replace(path: Path, start: str, end: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{body}\n{end}{tail}", encoding="utf-8")


def main() -> int:
    benchmark = Path("artifacts/runs/benchmark.json")
    if not benchmark.exists():
        print("no benchmark.json; run `engagegraph benchmark` first")
        return 0
    body = table()
    replace(Path("README.md"), START, END, body)
    replace(Path("README.zh-CN.md"), START, END, body)
    replace(Path("docs/model-card.md"), MODEL_START, MODEL_END, body)
    print("updated README.md, README.zh-CN.md, docs/model-card.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
