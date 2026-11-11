# Reproducibility verification

Generated: 2026-11-11T17:29:12.225885+00:00  
Package: `2.2.0`  
Python: `3.12.3`  
Dataset fingerprint: `2b8d9290d5614afa22f6`  
Status: **PASS**

| Check | Result | Detail |
| --- | --- | --- |
| dataset fingerprint | PASS | 2b8d9290d5614afa22f6 == 2b8d9290d5614afa22f6 |
| train matrix | PASS | 2,077,959 nnz, digest 358d16c6d8c459de799e |
| test matrix | PASS | 561,231 nnz, digest 3077922bb79997291bbf |
| item-kNN state | PASS | all state arrays byte-identical |
| item-kNN scores | PASS | max abs diff 0.000e+00 |
| evaluation metrics | PASS | max metric diff 0.000e+00 |
| domain labels do not change panel matrices | PASS | train/test CSR byte-identical under a different domain label |
| generic profile builds | PASS | 40 users x 15 items, domain=generic-retail |
| generic blocklist empty | PASS | no app blocklist applied |
| percentile monotone invariance | PASS | max diff 0.000e+00 |

## Scope

The real January/February benchmark is built twice from the canonical config; matrices, item-kNN state, scores, and metrics are compared exactly. The generic retail profile is built from its synthetic CSV to confirm that domain labels and the empty blocklist do not alter the data contract. The fast mathematical invariants run in CI (`tests/test_math_verification.py`).

## Caveats

- Item-kNN is deterministic; ALS/MetaPath2Vec are seeded but not guaranteed bitwise across hardware/library builds.
- The generic profile is synthetic and demonstrates metadata/domain invariance, not external validity.
- The panel benchmark is one month pair; reproducibility is not a claim of generalisation.
