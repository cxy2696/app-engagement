#!/usr/bin/env python
"""Fit a compact demo artefact so the dashboard works out of the box.

This is not a benchmark run.  It fits item-kNN on the processed dataset,
evaluates once, writes `artifacts/demo`, and makes it the active registry entry.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from engagegraph.artifacts import ModelRegistry, save_artifact
from engagegraph.data.preprocess import Dataset
from engagegraph.evaluation.metrics import evaluate
from engagegraph.models.itemknn import ItemKNNRecommender


def main() -> int:
    dataset_path = Path("data/processed")
    if not (dataset_path / "manifest.json").exists():
        raise SystemExit("run `engagegraph data build --config configs/dataset.yaml` first")
    dataset = Dataset.load(dataset_path)
    model = ItemKNNRecommender(k=100).fit(dataset)
    result = evaluate(model, dataset, k_values=(10, 20, 50), primary_k=10)
    target = Path("artifacts/demo")
    if target.exists():
        shutil.rmtree(target)
    artifact = save_artifact(
        model,
        dataset,
        target,
        metrics=result.metrics,
        config={"demo": True, "model": "itemknn"},
    )
    result.save(target / "evaluation")
    registry = ModelRegistry("artifacts")
    registry.register(artifact, make_active=True)
    (target / "demo.json").write_text(
        json.dumps({"metrics": result.metrics, "sample_users": []}, indent=2), encoding="utf-8"
    )
    print(f"demo artefact: {target}")
    print(f"NDCG@10: {result.metrics['ndcg@10']:.4f}")
    print("serve with: engagegraph serve --artifact artifacts/demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
