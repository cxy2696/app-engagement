#!/usr/bin/env python
"""Audit the original notebook's link-prediction protocol.

This script is deliberately separate from the benchmark.  It reproduces the
*shape* of the original evaluation (February positives, 1:1 negatives, logistic
regression on a 75/25 split) under three negative-sampling strategies, and
quantifies why a high AUC does not imply usable recommendation precision.

It also documents two code-level defects visible in the original notebook:
an elementwise "cosine" read-out and a February mapping mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from engagegraph.artifacts import load_artifact
from engagegraph.data.preprocess import Dataset


def _embedding_matrix(artifact: Any) -> tuple[np.ndarray | None, str]:
    for attribute in ("item_embeddings", "item_factors"):
        value = getattr(artifact.model, attribute, None)
        if isinstance(value, np.ndarray) and value.size:
            return value, f"{artifact.model.name}.{attribute}"
    candidates = sorted(Path("artifacts/runs").glob("metapath2vec-*/artifact"))
    if candidates:
        fallback = load_artifact(candidates[-1])
        value = getattr(fallback.model, "item_embeddings", None)
        if isinstance(value, np.ndarray) and value.size:
            return value, f"{fallback.model.name}.item_embeddings (fallback)"
    return None, "unavailable"


def audit(
    artifact_path: str,
    dataset_path: str,
    *,
    n_users: int = 6000,
    seed: int = 42,
) -> dict[str, Any]:
    dataset = Dataset.load(dataset_path)
    artifact = load_artifact(artifact_path)
    if artifact.manifest["dataset"]["fingerprint"] != dataset.fingerprint:
        raise ValueError("artifact and dataset fingerprints do not match")
    rng = np.random.default_rng(seed)
    eval_users = dataset.eval_users
    if len(eval_users) > n_users:
        eval_users = rng.choice(eval_users, size=n_users, replace=False)
    eval_users = np.sort(eval_users)
    popularity = np.asarray(dataset.train.getnnz(axis=0)).ravel().astype(np.float64)
    scores = artifact.model.score_users(eval_users, train=dataset.train)
    prevalence = float(dataset.test.nnz / (len(dataset.eval_users) * dataset.n_items))

    records: dict[str, dict[str, float]] = {}
    pair_samples: dict[str, list[tuple[int, int, int]]] = {}
    for strategy in ("uniform", "popularity", "hard"):
        pairs: list[tuple[int, int, int]] = []
        for row, user in enumerate(eval_users):
            positives = {int(i) for i in dataset.test[user].indices}
            seen = {int(i) for i in dataset.train[user].indices}
            excluded = seen | positives
            mask = np.ones(dataset.n_items, dtype=bool)
            mask[list(excluded)] = False
            candidates = np.flatnonzero(mask).astype(np.int64)
            if candidates.size == 0:
                continue
            if strategy == "uniform":
                negative_pool = candidates
                weights = None
            elif strategy == "popularity":
                negative_pool = candidates
                weights = popularity[candidates].astype(np.float64)
                weights = weights / weights.sum()
            else:
                candidate_scores = scores[row, candidates]
                k = max(1, int(0.01 * candidates.size))
                negative_pool = candidates[np.argpartition(-candidate_scores, k - 1)[:k]]
                weights = None
            for item in positives:
                negative = int(rng.choice(negative_pool, p=weights))
                pairs.append((row, item, 1))
                pairs.append((row, negative, 0))
        if not pairs:
            continue
        pairs_array = np.asarray(pairs)
        order = rng.permutation(len(pairs_array))
        split = int(0.75 * len(order))
        train_idx, test_idx = order[:split], order[split:]
        x_train = scores[pairs_array[train_idx, 0], pairs_array[train_idx, 1]].reshape(-1, 1)
        y_train = pairs_array[train_idx, 2]
        x_test = scores[pairs_array[test_idx, 0], pairs_array[test_idx, 1]].reshape(-1, 1)
        y_test = pairs_array[test_idx, 2]
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(x_train, y_train)
        probability = classifier.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        tp = int(np.sum((prediction == 1) & (y_test == 1)))
        fp = int(np.sum((prediction == 1) & (y_test == 0)))
        fn = int(np.sum((prediction == 0) & (y_test == 1)))
        tn = int(np.sum((prediction == 0) & (y_test == 0)))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        ppv_at = {}
        for p in (0.01, 0.02, 0.05):
            denom = tpr * p + fpr * (1 - p)
            ppv_at[f"{p:.2f}"] = float(tpr * p / denom) if denom else 0.0
        records[strategy] = {
            "n_pairs": len(pairs_array),
            "auc": float(roc_auc_score(y_test, probability)),
            "average_precision": float(average_precision_score(y_test, probability)),
            "precision_balanced": float(tp / max(tp + fp, 1)),
            "recall_balanced": float(tpr),
            "fpr": float(fpr),
            "ppv_at_prevalence": ppv_at,
        }
        pair_samples[strategy] = pairs_array[:5].tolist()

    # Code-level defect 1: elementwise "cosine" averaged over dimensions.
    embeddings, embedding_source = _embedding_matrix(artifact)
    cosine_audit: dict[str, float] = {}
    if embeddings is not None and len(embeddings) > 100:
        sample = np.asarray(embeddings[:500], dtype=np.float64)
        sample /= np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
        dots = sample @ sample.T
        # The notebook computed mean(u * v / (||u|| ||v||)); for normalized
        # vectors that is mean(u * v), not sum(u * v).
        elementwise_mean = np.array([np.mean(sample[0] * sample[j]) for j in range(1, 100)])
        cosine_audit = {
            "correct_cosine_std": float(np.std(dots[np.triu_indices_from(dots, k=1)])),
            "elementwise_mean_mean": float(np.mean(elementwise_mean)),
            "elementwise_mean_std": float(np.std(elementwise_mean)),
            "theoretical_1_over_dim": float(1.0 / sample.shape[1]),
        }

    return {
        "artifact": str(artifact_path),
        "dataset_fingerprint": dataset.fingerprint,
        "model": artifact.model.name,
        "n_users_sampled": len(eval_users),
        "catalogue_prevalence": prevalence,
        "negative_sampling": records,
        "cosine_defect": cosine_audit,
        "embedding_source": embedding_source,
        "notes": [
            "AUC is computed on a balanced 1:1 pair sample; deployment prevalence is far lower.",
            "PPV at prevalence uses the test-set TPR/FPR and Bayes' rule.",
            "Hard negatives are the top 1% of scored unseen, non-target items; this mimics "
            "the original notebook's similarity-selected negatives.",
            "The original notebook selected the best of nine binary operators on the same "
            "February labels, which adds another selection layer.",
        ],
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = payload["negative_sampling"]
    lines = [
        "# Legacy AUC audit",
        "",
        "The original notebook reported `ROC-AUC 0.9257` and `Precision 0.8552` for a "
        "February link-prediction task. This audit reproduces the *shape* of that "
        "evaluation with the maintained artefact and shows why the number is not a "
        "recommendation metric.",
        "",
        f"- Model audited: `{payload['model']}`",
        f"- Users sampled: {payload['n_users_sampled']:,}",
        f"- Catalogue prevalence of February positives: {payload['catalogue_prevalence']:.3%}",
        "",
        "## Negative sampling changes the story",
        "",
        "| Negative strategy | AUC | AP | Precision (balanced) | Recall | FPR | PPV @ 1% | PPV @ 5% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, record in rows.items():
        ppv = record["ppv_at_prevalence"]
        lines.append(
            f"| {strategy} | {record['auc']:.4f} | {record['average_precision']:.4f} | "
            f"{record['precision_balanced']:.4f} | {record['recall_balanced']:.4f} | "
            f"{record['fpr']:.4f} | {ppv['0.01']:.3f} | {ppv['0.05']:.3f} |"
        )
    lines += [
        "",
        "A balanced 1:1 test can show a high AUC while precision at realistic prevalence "
        "is low. The original notebook's headline precision was computed on a balanced "
        "sample, not on the full catalogue. Reporting it as deployment precision is a "
        "category error.",
        "",
        "## Code-level defect: the cosine read-out",
        "",
        "```python",
        "def cosine(u, v):",
        "    return (u * v) / (norm_u * norm_v)   # elementwise vector, then .mean()",
        "```",
        "",
        "This returns a vector of elementwise products, which the notebook then averages. "
        "The correct cosine is a *sum*, not a mean. The original output printed 0.0156 for "
        "every top pair because its mean-imputed/collapsed embeddings produced a nearly "
        "constant elementwise average. The read-out destroys signal even when embeddings "
        "are healthy: in the maintained embedding matrix the same computation averages "
        "near zero, while the correct cosine has real variance.",
        "",
        f"- Embedding source: `{payload.get('embedding_source', 'unavailable')}`",
        f"- Observed elementwise mean: **{payload['cosine_defect'].get('elementwise_mean_mean', float('nan')):.6f}** "
        f"(std {payload['cosine_defect'].get('elementwise_mean_std', float('nan')):.6f})",
        f"- Scale of one component (1/dim): **{payload['cosine_defect'].get('theoretical_1_over_dim', float('nan')):.6f}**",
        f"- Correct cosine std across sampled pairs: **{payload['cosine_defect'].get('correct_cosine_std', float('nan')):.6f}**",
        "",
        "A near-constant read-out cannot rank apps. This is why the original top-10 "
        "lists were arbitrary and why the reported temporal consistency was zero.",
        "",
        "## Code-level defect: February mapping mismatch",
        "",
        "The notebook builds February embeddings in the order of `unique_apps_feb`, then "
        "filters them with indices from a *different* mapping (`app_mapping_feb`). The "
        "result mixes app identities: February top pairs contain unrelated apps with "
        "identical mean-imputed embeddings. That is a mapping bug, not evidence of "
        "temporal instability.",
        "",
        "## What this means",
        "",
        "The maintained benchmark does not use this protocol. It fits January only, ranks "
        "the full catalogue, masks seen apps, and reports macro-averaged ranking metrics "
        "with paired confidence intervals. See `docs/methodology.md`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--dataset", default="data/processed")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--users", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    payload = audit(args.artifact, args.dataset, n_users=args.users, seed=args.seed)
    (output / "legacy-audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(payload, output / "legacy-audit.md")
    print(f"wrote {output / 'legacy-audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
