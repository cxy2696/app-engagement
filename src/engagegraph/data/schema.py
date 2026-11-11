"""Input schema, validation, and defensive cleaning for interaction logs.

The canonical input is a *long* table with one row per (user, item) pair and a
non-negative engagement weight.  Raw daily exports typically contain duplicate
pairs; the cleaner aggregates them.  The cleaner never silently imputes user or
item identifiers: rows without both identifiers are rejected and counted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e]")
_WHITESPACE = re.compile(r"\s+")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE = re.compile(r"^\+?\d{10,15}$")


class SchemaError(ValueError):
    """Raised when an input cannot be interpreted as an interaction table."""


@dataclass(frozen=True)
class DataQualityReport:
    """Auditable record of every row removed or transformed by the cleaner."""

    rows_in: int
    rows_out: int
    users: int
    items: int
    dropped_missing_id: int = 0
    dropped_missing_item: int = 0
    dropped_missing_weight: int = 0
    dropped_negative_weight: int = 0
    dropped_empty_name: int = 0
    duplicates_collapsed: int = 0
    names_normalised: int = 0
    suspected_pii_user_ids: int = 0
    weight_min: float | None = None
    weight_median: float | None = None
    weight_p99: float | None = None
    weight_max: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_label(value: str) -> str:
    """Unicode-normalise and collapse whitespace without changing semantics.

    NFKC is chosen over NFC because panel exports frequently mix full-width and
    half-width characters.  Control characters and bidi overrides are removed;
    they are never meaningful in an app label and can be used to spoof UI text.
    """

    text = unicodedata.normalize("NFKC", str(value))
    text = _CONTROL_CHARS.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def looks_like_pii(value: str) -> bool:
    """Heuristic guardrail for accidental personal identifiers in ID columns."""

    return bool(_EMAIL.match(value) or _PHONE.match(value))


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"missing required column(s): {', '.join(missing)}; "
            f"found: {', '.join(map(str, frame.columns))}"
        )


def clean_interactions(
    frame: pd.DataFrame,
    *,
    user_col: str = "PANELISTID",
    item_col: str = "APPDESCRIPTION",
    weight_col: str = "weight",
    normalize_names: bool = True,
    drop_empty: bool = True,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate and normalise an interaction table.

    Returns a copy with exactly three columns (``user``, ``item``, ``weight``,
    renamed for stability), sorted by user and item.  The function is pure: the
    input frame is never mutated.
    """

    _require_columns(frame, [user_col, item_col, weight_col])
    if frame.empty:
        raise SchemaError("input table is empty")

    rows_in = len(frame)
    work = frame[[user_col, item_col, weight_col]].copy()
    work.columns = ["user", "item", "weight"]

    original = work.copy()
    missing_id = int(work["user"].isna().sum())
    missing_item = int(work["item"].isna().sum())
    missing_weight = int(work["weight"].isna().sum())
    work = work.dropna(subset=["user", "item"])
    work = work.dropna(subset=["weight"])

    work["weight"] = pd.to_numeric(work["weight"], errors="coerce")
    bad_weight = int(work["weight"].isna().sum())
    work = work.dropna(subset=["weight"])
    negative = int((work["weight"] < 0).sum())
    work = work[work["weight"] >= 0]

    work["user"] = work["user"].astype(str).str.strip()
    work["item"] = work["item"].astype(str).str.strip()
    empty_user = int((work["user"] == "").sum())
    if empty_user:
        work = work[work["user"] != ""]
    empty_item = int((work["item"] == "").sum())
    if drop_empty:
        work = work[work["item"] != ""]

    if normalize_names:
        before = work["item"].copy()
        work["item"] = work["item"].map(normalize_label)
        names_normalised = int((before != work["item"]).sum())
    else:
        names_normalised = 0

    work = work[work["item"] != ""]

    duplicates = int(work.duplicated(["user", "item"]).sum())
    if duplicates:
        work = (
            work.groupby(["user", "item"], as_index=False, sort=False)["weight"]
            .sum()
            .astype({"weight": "float64"})
        )

    work = work.sort_values(["user", "item"], kind="stable").reset_index(drop=True)

    warnings: list[str] = []
    pii = int(work["user"].head(10_000).map(looks_like_pii).sum())
    if pii:
        warnings.append(
            f"{pii} user identifier(s) look like email addresses or phone numbers; "
            "hash or pseudonymise identifiers before deployment."
        )
    if bad_weight:
        warnings.append(f"{bad_weight} non-numeric weight(s) were dropped.")
    if negative:
        warnings.append(f"{negative} negative weight(s) were dropped.")
    if empty_user:
        warnings.append(f"{empty_user} empty user identifier(s) were dropped.")
    if empty_item:
        warnings.append(f"{empty_item} empty item label(s) were dropped.")

    report = DataQualityReport(
        rows_in=rows_in,
        rows_out=len(work),
        users=int(work["user"].nunique()) if len(work) else 0,
        items=int(work["item"].nunique()) if len(work) else 0,
        dropped_missing_id=missing_id + empty_user,
        dropped_missing_item=missing_item,
        dropped_missing_weight=missing_weight + bad_weight,
        dropped_negative_weight=negative,
        dropped_empty_name=empty_item,
        duplicates_collapsed=duplicates,
        names_normalised=names_normalised,
        suspected_pii_user_ids=pii,
        weight_min=float(work["weight"].min()) if len(work) else None,
        weight_median=float(work["weight"].median()) if len(work) else None,
        weight_p99=float(work["weight"].quantile(0.99)) if len(work) else None,
        weight_max=float(work["weight"].max()) if len(work) else None,
        warnings=warnings,
    )

    # Keep a reference so linters know the copy is intentional for auditability.
    del original
    return work, report


def validate_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Assert that no (user, item) pair appears in both train and test.

    This is a cheap, explicit guard against the most common temporal-split bug.
    """

    train_pairs = set(map(tuple, train[["user", "item"]].to_numpy()))
    test_pairs = set(map(tuple, test[["user", "item"]].to_numpy()))
    overlap = train_pairs & test_pairs
    if overlap:
        sample = list(overlap)[:3]
        raise AssertionError(f"train/test leakage detected for {len(overlap)} pairs: {sample}")


def summarize_weight_unit(values: np.ndarray) -> str:
    """Best-effort description of the weight column, never a silent conversion."""

    if values.size == 0:
        return "empty"
    median = float(np.median(values))
    if median >= 60_000:
        return "likely milliseconds (median >= 60,000)"
    if median >= 60:
        return "likely seconds (median >= 60)"
    return "unknown or already normalised"
