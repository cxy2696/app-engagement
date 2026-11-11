# Verification

This page records the checks used to confirm that EngageGraph is **generalisable
by construction** and **reproducible on the reference benchmark**. The latest
machine-generated evidence is in [`reports/verification.md`](../reports/verification.md)
and [`reports/verification.json`](../reports/verification.json).

## 1. Commands

```bash
make check    # lint, type check, 45 fast tests
make verify   # real-data determinism + domain-invariance report
```

`make verify` requires the panel exports and is intentionally run locally or in
a protected job, not in CI. CI runs the fast mathematical invariants on synthetic
fixtures.

## 2. What is checked

| Check | Method | Pass criterion |
| --- | --- | --- |
| Dataset determinism | Build the canonical config twice | Identical fingerprint and byte-identical CSR matrices |
| Model determinism | Fit item-kNN twice | All state arrays byte-identical |
| Score determinism | Score the same 256 users twice | Exact float equality |
| Metric determinism | Evaluate both models | Zero metric difference |
| Domain invariance | Same panel, different `DomainConfig` | Identical train/test matrices |
| Generic profile | Build synthetic retail CSV | Valid dataset, empty blocklist, non-zero matrices |
| Percentile math | Rank transform on a real score row | Invariant under a strictly increasing transform |
| Math invariants | Fast tests: brute-force cosine, NDCG/MAP, percentile ties, test-matrix independence | All assertions pass |

## 3. Mathematical invariants in CI

`tests/test_math_verification.py` checks:

- item-kNN scores equal a brute-force cosine computation on a dense fixture;
- percentile ties use the upper rank and are monotone-invariant;
- NDCG/Recall/MAP/HitRate are unchanged by a strictly increasing score transform;
- changing only the domain vocabulary leaves matrices and scores identical;
- altering the test matrix does not change fitted scores.

These are properties of the implementation, not empirical claims. They are
fast enough to run on every commit.

## 4. Reproducibility scope

- **Deterministic:** dataset construction, popularity, item-kNN, evaluation,
  artefact serialization, report tables.
- **Seeded but not bitwise:** ALS (BLAS/version dependent) and MetaPath2Vec
  (PyG random-walk order/threads). See [`reproducibility.md`](reproducibility.md).
- **Data-dependent:** the panel exports are fixed files with SHA-256 hashes in
  the dataset manifest; new months are new snapshots.
- **Not claimed:** generalisation to other markets or months, causal effects,
  or online business lift.

## 5. Interpreting a PASS

A PASS means the reference protocol is deterministic on the recorded
environment and that domain labels are metadata rather than algorithmic inputs.
It does **not** mean any model is correct for a new domain. For a new dataset,
rerun `engagegraph data build`, `benchmark`, and the verification script.
