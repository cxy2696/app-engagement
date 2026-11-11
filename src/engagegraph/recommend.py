"""Serving-side recommendation logic shared by the CLI and HTTP API.

The service is deliberately thin: it validates identifiers, applies seen-item
masking, delegates scoring to the fitted model, and attaches auditable reasons.
All model and data provenance stays in the artefact.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import sparse

from engagegraph.artifacts import Artifact
from engagegraph.data.preprocess import Dataset
from engagegraph.models.base import Explanation, Recommendation

# Reference blocklist for the app-engagement profile.  It is *not* applied by
# default: the active dataset's ``domain.blocklist`` controls serving.  Keeping
# the list here makes the app profile importable from Python while other
# domains stay untouched.
APP_ENGAGEMENT_BLOCKLIST = frozenset(
    {
        "Android System",
        "Settings",
        "Permission controller",
        "System UI",
        "Phone",
        "Call",
        "Camera",
        "Samsung Gallery",
        "Google Play services",
        "Samsung One UI Home",
        "Clock by Samsung Electronics Co.,  Ltd.",
        "Google Photos",
        "VpnDialogs",
        "CaptivePortalLogin",
        "Media and devices",
        "MobileXpression Panel",
    }
)

# Backwards-compatible alias for callers that imported the old constant.
DEFAULT_BLOCKLIST = APP_ENGAGEMENT_BLOCKLIST


@dataclass
class RecommendationResponse:
    user_id: str
    strategy: str
    model: str
    items: list[Recommendation]
    latency_ms: float
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "strategy": self.strategy,
            "model": self.model,
            "generated_at": self.generated_at,
            "latency_ms": round(self.latency_ms, 3),
            "items": [item.as_dict() for item in self.items],
        }


@dataclass(frozen=True)
class SimilarItem:
    item_id: str
    score: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": round(float(self.score), 6),
            "rank": self.rank,
        }


class RecommendationService:
    """Read-only recommendation facade over one artefact."""

    def __init__(self, artifact: Artifact, blocklist: set[str] | None = None) -> None:
        self.artifact = artifact
        self.domain = dict(artifact.stats.get("domain") or {})
        self.user_label = self.domain.get("user_label", "user")
        self.user_label_plural = self.domain.get("user_label_plural", "users")
        self.item_label = self.domain.get("item_label", "item")
        self.item_label_plural = self.domain.get("item_label_plural", "items")
        self.train_period_label = self.domain.get("train_period_label", "train period")
        self.model = artifact.model
        self.history = artifact.history
        self.user_index = artifact.user_index
        self.item_index = artifact.item_index
        self.user_ids = artifact.user_ids
        self.item_ids = artifact.item_ids
        self.n_users, self.n_items = self.history.shape
        if blocklist is None:
            if artifact.stats.get("domain") is None:
                # Legacy 2.0 artefact: the only shipped data is the app panel,
                # so preserve its historical serving behaviour.
                blocklist = set(APP_ENGAGEMENT_BLOCKLIST)
            else:
                blocklist = set(self.domain.get("blocklist") or ())
        env_blocklist = os.environ.get("ENGAGEGRAPH_BLOCKLIST", "")
        if env_blocklist:
            blocklist = set(blocklist) | {
                name.strip() for name in env_blocklist.split(",") if name.strip()
            }
        self.blocked = {self.item_index[name] for name in blocklist if name in self.item_index}
        self.context = Dataset(
            train=self.history,
            test=sparse.csr_matrix((self.n_users, self.n_items), dtype=np.float32),
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            stats=artifact.stats,
            config={},
            fingerprint=artifact.manifest.get("dataset", {}).get("fingerprint", ""),
        )
        pop = np.asarray(self.history.getnnz(axis=0)).ravel().astype(np.float64)
        self.popularity = pop
        self._sample_users = self._choose_sample_users()

    def _choose_sample_users(self) -> list[str]:
        lengths = np.diff(self.history.indptr)
        candidates = np.flatnonzero((lengths >= 10) & (lengths <= 150))
        if candidates.size == 0:
            candidates = np.arange(self.n_users)
        rng = np.random.default_rng(7)
        chosen = rng.choice(candidates, size=min(6, candidates.size), replace=False)
        return [self.user_ids[int(i)] for i in chosen]

    @property
    def sample_users(self) -> list[str]:
        return list(self._sample_users)

    def _popular_items(self, k: int, exclude: set[int] | None = None) -> list[Recommendation]:
        order = np.argsort(-self.popularity)
        items: list[Recommendation] = []
        for item_idx in order:
            if exclude and int(item_idx) in exclude:
                continue
            if int(item_idx) in self.blocked:
                continue
            items.append(
                Recommendation(
                    item_id=self.item_ids[int(item_idx)],
                    score=float(np.log1p(self.popularity[int(item_idx)])),
                    rank=len(items) + 1,
                    explanations=[
                        Explanation(
                            item_id=self.item_ids[int(item_idx)],
                            reason=(
                                f"used by {int(self.popularity[int(item_idx)]):,} "
                                f"{self.user_label_plural} in {self.train_period_label}"
                            ),
                            contribution=float(self.popularity[int(item_idx)]),
                        )
                    ],
                )
            )
            if len(items) >= k:
                break
        return items

    def recommend(
        self,
        user_id: str,
        *,
        k: int = 10,
        explain: bool = True,
        diversify: bool = False,
        diversity_weight: float = 0.2,
    ) -> RecommendationResponse:
        started = time.perf_counter()
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        if user_id not in self.user_index:
            items = self._popular_items(k)
            return RecommendationResponse(
                user_id=user_id,
                strategy="cold_start_popularity",
                model=self.model.name,
                items=items,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        user_idx = self.user_index[user_id]
        scores = self.model.score_users(np.asarray([user_idx]), train=self.history)[0]
        if scores.shape[0] != self.n_items:
            raise RuntimeError("model returned an unexpected score vector length")
        seen = self.history.indices[
            self.history.indptr[user_idx] : self.history.indptr[user_idx + 1]
        ]
        scores = scores.astype(np.float64, copy=True)
        scores[seen] = -np.inf
        if self.blocked:
            scores[list(self.blocked)] = -np.inf
        finite = np.isfinite(scores)
        if not np.any(finite):
            items = self._popular_items(k, exclude=set(map(int, seen)))
            return RecommendationResponse(
                user_id=user_id,
                strategy="exhausted_catalog_popularity",
                model=self.model.name,
                items=items,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        candidates = np.flatnonzero(finite)
        if diversify:
            selected = self._mmr(user_idx, scores, candidates, k, diversity_weight)
        else:
            top = candidates[np.argsort(-scores[candidates])[:k]]
            selected = [int(i) for i in top]

        items = []
        for rank, item_idx in enumerate(selected, start=1):
            reasons = self._safe_explain(user_idx, item_idx) if explain else []
            items.append(
                Recommendation(
                    item_id=self.item_ids[item_idx],
                    score=float(scores[item_idx]),
                    rank=rank,
                    explanations=reasons,
                )
            )
        return RecommendationResponse(
            user_id=user_id,
            strategy="personalized",
            model=self.model.name,
            items=items,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _safe_explain(self, user_idx: int, item_idx: int) -> list[Explanation]:
        try:
            reasons = self.model.explain(user_idx, item_idx, self.context, k=2)
        except (NotImplementedError, ValueError, IndexError):
            reasons = []
        blocked_names = {self.item_ids[index] for index in self.blocked}
        reasons = [reason for reason in reasons if reason.item_id not in blocked_names]
        if not reasons:
            reasons = self.model.explain_popularity(item_idx, self.context)
        return reasons

    def _item_vectors(self) -> np.ndarray | None:
        for attribute in ("item_embeddings", "item_factors"):
            vectors = getattr(self.model, attribute, None)
            if isinstance(vectors, np.ndarray) and vectors.size:
                return vectors.astype(np.float64, copy=False)
        sim = getattr(self.model, "_sim", None)
        if isinstance(sim, sparse.csr_matrix):
            return None
        return None

    def _mmr(
        self,
        user_idx: int,
        scores: np.ndarray,
        candidates: np.ndarray,
        k: int,
        weight: float,
    ) -> list[int]:
        weight = float(np.clip(weight, 0.0, 1.0))
        vectors = self._item_vectors()
        if vectors is None or weight == 0.0:
            order = candidates[np.argsort(-scores[candidates])[:k]]
            return [int(i) for i in order]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms
        pool = list(map(int, candidates[np.argsort(-scores[candidates])[: max(k * 10, 50)]]))
        selected: list[int] = []
        max_score = max(float(scores[pool[0]]), 1e-12)
        while pool and len(selected) < k:
            if not selected:
                choice = pool.pop(0)
            else:
                chosen_vectors = normalized[selected]
                similarities = normalized[pool] @ chosen_vectors.T
                redundancy = similarities.max(axis=1)
                utility = scores[pool] / max_score - weight * redundancy
                best = int(np.argmax(utility))
                choice = pool.pop(best)
            selected.append(choice)
        return selected

    def similar(self, item_id: str, k: int = 10) -> list[SimilarItem]:
        if item_id not in self.item_index:
            raise KeyError(item_id)
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        item_idx = self.item_index[item_id]
        try:
            neighbours = self.model.similar_items(item_idx, k)
        except NotImplementedError:
            neighbours = self._cooccurrence_neighbours(item_idx, k)
        return [
            SimilarItem(item_id=self.item_ids[idx], score=score, rank=rank)
            for rank, (idx, score) in enumerate(neighbours, start=1)
        ]

    def _cooccurrence_neighbours(self, item_idx: int, k: int) -> list[tuple[int, float]]:
        column = self.history[:, item_idx]
        cooccurrence = np.asarray((self.history.T @ column).toarray()).ravel()
        cooccurrence[item_idx] = 0.0
        support = np.maximum(np.asarray(self.history.getnnz(axis=0)).ravel(), 1)
        score = cooccurrence / np.sqrt(support * max(support[item_idx], 1))
        order = np.argsort(-score)[:k]
        return [(int(i), float(score[i])) for i in order if score[i] > 0]

    def popular(self, k: int = 10) -> list[Recommendation]:
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        return self._popular_items(k)

    def profile(self, user_id: str) -> dict[str, Any]:
        if user_id not in self.user_index:
            raise KeyError(user_id)
        user_idx = self.user_index[user_id]
        start, end = self.history.indptr[user_idx], self.history.indptr[user_idx + 1]
        indices = self.history.indices[start:end]
        weights = self.history.data[start:end]
        order = np.argsort(-weights)[:10]
        top = [
            {"item_id": self.item_ids[int(indices[i])], "engagement": float(weights[i])}
            for i in order
        ]
        return {
            "user_id": user_id,
            "n_items": int(end - start),
            "top_items": top,
            # Backwards-compatible keys for clients built against 2.0.
            "n_apps": int(end - start),
            "top_apps": top,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model.meta(),
            "dataset": {
                "fingerprint": self.artifact.manifest.get("dataset", {}).get("fingerprint"),
                "n_users": self.n_users,
                "n_items": self.n_items,
                "stats": self.artifact.stats,
            },
            "metrics": self.artifact.metrics,
            "artifact": {
                "created_at": self.artifact.manifest.get("created_at"),
                "package_version": self.artifact.manifest.get("package_version"),
                "schema": self.artifact.manifest.get("artifact_schema_version"),
                "fingerprint": self.artifact.manifest.get("artifact_fingerprint"),
            },
            "domain": self.domain,
            "labels": {
                "user": self.user_label,
                "user_plural": self.user_label_plural,
                "item": self.item_label,
                "item_plural": self.item_label_plural,
                "train_period": self.train_period_label,
            },
            "sample_users": self.sample_users,
            "serving_blocklist_size": len(self.blocked),
        }
