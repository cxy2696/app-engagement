"""Portable model artefacts.

An artefact is a directory containing only JSON and NumPy ``.npz`` files.  No
pickle is used, so loading an artefact from an untrusted registry cannot execute
code.  The manifest records the dataset fingerprint, package version, and the
metrics observed at build time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from engagegraph import __version__
from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import BaseRecommender, build_model

ARTIFACT_SCHEMA_VERSION = "2.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    env = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class Artifact:
    model: BaseRecommender
    user_ids: list[str]
    item_ids: list[str]
    stats: dict[str, Any]
    metrics: dict[str, float]
    manifest: dict[str, Any]
    path: Path
    history: sparse.csr_matrix

    @property
    def user_index(self) -> dict[str, int]:
        return {user: idx for idx, user in enumerate(self.user_ids)}

    @property
    def item_index(self) -> dict[str, int]:
        return {item: idx for idx, item in enumerate(self.item_ids)}

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.manifest.get("model", {}),
            "dataset": self.manifest.get("dataset", {}),
            "metrics": self.metrics,
            "created_at": self.manifest.get("created_at"),
            "package_version": self.manifest.get("package_version"),
            "fingerprint": self.manifest.get("artifact_fingerprint"),
        }


def save_artifact(
    model: BaseRecommender,
    dataset: Dataset,
    directory: str | Path,
    *,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Artifact:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.get_state()
    np.savez_compressed(directory / "model.npz", **state)  # type: ignore[arg-type]
    mappings = {"user_ids": dataset.user_ids, "item_ids": dataset.item_ids}
    (directory / "mappings.json").write_text(
        json.dumps(mappings, ensure_ascii=False), encoding="utf-8"
    )
    # The January interaction matrix is required for seen-item masking, exact
    # item-kNN scoring, and co-usage explanations at serving time.
    np.savez_compressed(
        directory / "history.npz",
        data=dataset.train.data.astype(np.float32, copy=False),
        indices=dataset.train.indices.astype(np.int32, copy=False),
        indptr=dataset.train.indptr.astype(np.int64, copy=False),
        shape=np.asarray(dataset.train.shape, dtype=np.int64),
    )
    manifest: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "model": model.meta(),
        "dataset": {
            "fingerprint": dataset.fingerprint,
            "n_users": dataset.n_users,
            "n_items": dataset.n_items,
            "stats": dataset.stats,
            "config": dataset.config,
        },
        "config": config or {},
        "metrics": metrics or {},
        "artifact_fingerprint": "",
    }
    if extra:
        manifest.update(extra)
    manifest_path = directory / "manifest.json"

    # Write once to compute the fingerprint over deterministic payloads.
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    digest = hashlib.sha256()
    digest.update((directory / "model.npz").read_bytes())
    digest.update((directory / "mappings.json").read_bytes())
    digest.update((directory / "history.npz").read_bytes())
    manifest["artifact_fingerprint"] = digest.hexdigest()[:20]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return Artifact(
        model=model,
        user_ids=list(dataset.user_ids),
        item_ids=list(dataset.item_ids),
        stats=dataset.stats,
        metrics=manifest["metrics"],
        manifest=manifest,
        path=directory,
        history=dataset.train,
    )


def load_artifact(directory: str | Path) -> Artifact:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"not an EngageGraph artefact: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artefact schema: {manifest.get('artifact_schema_version')!r}"
        )
    mappings = json.loads((directory / "mappings.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "model.npz", allow_pickle=False)
    model_meta = manifest["model"]
    model = build_model(model_meta["model_type"], model_meta.get("params", {}))
    model.set_state({key: arrays[key] for key in arrays.files})
    model.n_users = len(mappings["user_ids"])
    model.n_items = len(mappings["item_ids"])
    history_arrays = np.load(directory / "history.npz", allow_pickle=False)
    history_shape = tuple(int(x) for x in history_arrays["shape"])
    history = sparse.csr_matrix(
        (
            history_arrays["data"],
            history_arrays["indices"],
            history_arrays["indptr"],
        ),
        shape=history_shape,
    ).astype(np.float32)
    return Artifact(
        model=model,
        user_ids=list(mappings["user_ids"]),
        item_ids=list(mappings["item_ids"]),
        stats=manifest["dataset"]["stats"],
        metrics=manifest.get("metrics", {}),
        manifest=manifest,
        path=directory,
        history=history,
    )


class ModelRegistry:
    """Tiny JSON registry with one active artefact and an append-only history."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)
        self.path = self.root / "registry.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"active": None, "models": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def register(self, artifact: Artifact, *, make_active: bool = True) -> dict[str, Any]:
        payload = self.load()
        entry = {
            "name": artifact.model.name,
            "path": str(artifact.path),
            "fingerprint": artifact.manifest.get("artifact_fingerprint"),
            "created_at": artifact.manifest.get("created_at"),
            "metrics": artifact.metrics,
        }
        payload["models"] = [m for m in payload["models"] if m.get("path") != entry["path"]]
        payload["models"].append(entry)
        if make_active:
            payload["active"] = str(artifact.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def resolve(self) -> Artifact:
        payload = self.load()
        active = payload.get("active")
        if not active:
            raise FileNotFoundError(
                "no active artefact; run `engagegraph train ...` or set artifacts/registry.json"
            )
        return load_artifact(active)


def artifact_digest(directory: str | Path) -> str:
    directory = Path(directory)
    return _sha256(directory / "model.npz")[:20]
