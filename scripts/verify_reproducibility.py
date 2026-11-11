#!/usr/bin/env python
"""End-to-end reproducibility and domain-invariance verification.

Runs the real reference configuration twice, checks byte-level identity of the
benchmark matrices, item-kNN state, and ranking metrics, then builds the
synthetic generic-retail profile and verifies that domain labels change only
metadata (not matrices or scores).

This is intentionally a local/protected job: it reads the panel exports and is
not run in CI.  The fast mathematical invariants are covered by
``tests/test_math_verification.py``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from engagegraph import __version__
from engagegraph.config import load_benchmark_config
from engagegraph.data.preprocess import build_dataset
from engagegraph.evaluation.metrics import evaluate
from engagegraph.models.itemknn import ItemKNNRecommender
from engagegraph.recommend import within_user_percentile


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:20]


def _matrix_digest(matrix: Any) -> str:
    return _digest(
        np.concatenate(
            [
                matrix.data.astype(np.float64),
                matrix.indices.astype(np.int64),
                matrix.indptr.astype(np.int64),
            ]
        )
    )


def main() -> int:
    started = time.perf_counter()
    config = load_benchmark_config("configs/benchmark.yaml")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    first = build_dataset(config.data, verbose=False)
    second = build_dataset(config.data, verbose=False)
    check(
        "dataset fingerprint",
        first.fingerprint == second.fingerprint,
        f"{first.fingerprint} == {second.fingerprint}",
    )
    check(
        "train matrix",
        _matrix_digest(first.train) == _matrix_digest(second.train),
        f"{first.train.nnz:,} nnz, digest {_matrix_digest(first.train)}",
    )
    check(
        "test matrix",
        _matrix_digest(first.test) == _matrix_digest(second.test),
        f"{first.test.nnz:,} nnz, digest {_matrix_digest(first.test)}",
    )

    model_a = ItemKNNRecommender(k=100).fit(first)
    model_b = ItemKNNRecommender(k=100).fit(second)
    state_a = model_a.get_state()
    state_b = model_b.get_state()
    check(
        "item-kNN state",
        all(np.array_equal(state_a[key], state_b[key]) for key in state_a),
        "all state arrays byte-identical",
    )
    users = np.arange(0, min(256, first.n_users), dtype=np.int64)
    scores_a = model_a.score_users(users, train=first.train)
    scores_b = model_b.score_users(users, train=second.train)
    check(
        "item-kNN scores",
        np.array_equal(scores_a, scores_b),
        f"max abs diff {float(np.max(np.abs(scores_a - scores_b))):.3e}",
    )

    result_a = evaluate(model_a, first, k_values=(10, 20, 50), primary_k=10, coverage_k=10)
    result_b = evaluate(model_b, second, k_values=(10, 20, 50), primary_k=10, coverage_k=10)
    metric_diff = max(
        abs(float(result_a.metrics[key]) - float(result_b.metrics[key])) for key in result_a.metrics
    )
    check("evaluation metrics", metric_diff == 0.0, f"max metric diff {metric_diff:.3e}")

    # Domain labels are metadata: same interaction table, different vocabulary.
    generic = load_benchmark_config("profiles/generic-interactions.yaml")
    generic_dataset = build_dataset(generic.data, verbose=False)
    renamed_panel = build_dataset(
        config.data.model_copy(update={"domain": generic.data.domain}), verbose=False
    )
    check(
        "domain labels do not change panel matrices",
        _matrix_digest(first.train) == _matrix_digest(renamed_panel.train)
        and _matrix_digest(first.test) == _matrix_digest(renamed_panel.test),
        "train/test CSR byte-identical under a different domain label",
    )
    check(
        "generic profile builds",
        generic_dataset.n_users > 0 and generic_dataset.n_items > 0,
        f"{generic_dataset.n_users} users x {generic_dataset.n_items} items, "
        f"domain={generic_dataset.stats['domain']['name']}",
    )
    check(
        "generic blocklist empty",
        generic_dataset.stats["domain"].get("blocklist", []) == [],
        "no app blocklist applied",
    )

    # Percentile is a rank statistic; verify monotone invariance on a real row.
    if first.eval_users.size:
        user = int(first.eval_users[0])
        scores = model_a.score_users(np.asarray([user]), train=first.train)[0]
        seen = set(first.train[user].indices)
        candidates = np.asarray([i for i in range(first.n_items) if i not in seen], dtype=np.int64)
        selected = candidates[np.argsort(-scores[candidates])[:20]]
        q1 = within_user_percentile(scores, candidates, selected)
        q2 = within_user_percentile(np.power(np.maximum(scores, 0.0), 3), candidates, selected)
        check(
            "percentile monotone invariance",
            np.allclose(q1, q2),
            f"max diff {float(np.max(np.abs(q1 - q2))):.3e}",
        )

    passed = all(item["passed"] for item in checks)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "package_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_sec": time.perf_counter() - started,
        "dataset_fingerprint": first.fingerprint,
        "n_users": first.n_users,
        "n_items": first.n_items,
        "checks": checks,
        "caveats": [
            "Item-kNN is deterministic; ALS/MetaPath2Vec are seeded but not guaranteed bitwise across hardware/library builds.",
            "The generic profile is synthetic and demonstrates metadata/domain invariance, not external validity.",
            "The panel benchmark is one month pair; reproducibility is not a claim of generalisation.",
        ],
    }
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Reproducibility verification",
        "",
        f"Generated: {payload['generated_at']}  ",
        f"Package: `{payload['package_version']}`  ",
        f"Python: `{platform.python_version()}`  ",
        f"Dataset fingerprint: `{first.fingerprint}`  ",
        f"Status: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        lines.append(
            f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {item['detail']} |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "The real January/February benchmark is built twice from the canonical config; "
        "matrices, item-kNN state, scores, and metrics are compared exactly. The generic "
        "retail profile is built from its synthetic CSV to confirm that domain labels and "
        "the empty blocklist do not alter the data contract. The fast mathematical "
        "invariants run in CI (`tests/test_math_verification.py`).",
        "",
        "## Caveats",
        "",
        *[f"- {item}" for item in payload["caveats"]],
        "",
    ]
    (out / "verification.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(line for line in lines if line.startswith("|") or "Status" in line))
    print(f"\nverification: {'PASS' if passed else 'FAIL'} ({payload['runtime_sec']:.1f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
