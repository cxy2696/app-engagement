# Reproducibility

EngageGraph is designed so that a reported number can be traced to a config, a dataset fingerprint, a package version, and an artefact. This document lists the controls and the known limits.

## 1. What is pinned

- **Config:** every run stores the full resolved YAML/JSON in `run.json` and the artefact manifest.
- **Dataset:** `data/processed/manifest.json` records source SHA-256 hashes, filtering config, quality counters, and a fingerprint derived from mappings, matrices, and source hashes.
- **Model:** the artefact stores model type, hyperparameters, random seed, and the exact NumPy state.
- **Environment:** `run.json` records Python, platform, and package version. `requirements-lock.txt` pins the tested dependency set for the reference run.
- **Git:** the manifest records `GIT_COMMIT`/`GITHUB_SHA` when available.

## 2. Seeds and determinism

- Python `random`, NumPy, and PyTorch seeds are set from `seed` before fitting.
- Popularity and item-kNN are deterministic given the matrix.
- ALS uses a fixed `random_state`; the `implicit` solver is deterministic on the same BLAS/CPU build in practice, but we do not claim bitwise reproducibility across hardware.
- MetaPath2Vec random walks and negative sampling are seeded, but PyG/C++ walk order and thread scheduling can produce small differences across versions. Treat graph results as reproducible within a tolerance, not bitwise.

## 3. Reference environment

```text
Python 3.12
pandas 3.0.x, numpy 2.5.x, scipy 1.18.x, scikit-learn 1.9.x,
implicit 0.7.x, fastapi 0.162.x, pyarrow 26.x
```

Install the tested set:

```bash
pip install -r requirements-lock.txt
pip install -e .
```

The lock file is regenerated deliberately when dependencies change and the benchmark is re-run.

## 4. Expected outputs

After `engagegraph data build` with the default config:

- 34,024 users, 9,848 items, 2,077,959 training interactions.
- 561,231 February targets across 28,286 evaluation users.
- 10,568 February cold-start users; 307 ineligible users.
- 93.4% catalogue coverage of February rows for warm users.

After `engagegraph benchmark`, each model writes an artefact and an `evaluation/` directory. `benchmark.json` contains aggregate metrics, timings, and paired comparisons. Exact metric values depend on the data snapshot, but the ordering and approximate magnitudes should be stable; see `reports/results.json` for the reference run.

## 5. Reproducing a single number

```bash
git checkout <commit>
pip install -r requirements-lock.txt && pip install -e .
engagegraph data build --config configs/benchmark.yaml --out data/processed
engagegraph evaluate \
  --artifact artifacts/runs/<run>/artifact \
  --dataset data/processed \
  --k 10 20 50
```

`evaluate` refuses to run if the artefact and dataset fingerprints differ. That guard prevents the common failure of evaluating a model against a silently rebuilt dataset.

## 6. Verification

`make verify` rebuilds the benchmark twice from `configs/benchmark.yaml` and
compares the results, then builds the synthetic generic-retail profile. It requires
the panel exports and runs locally or in a protected job, not in CI.

| Check | Pass criterion |
| --- | --- |
| Dataset determinism | Identical fingerprint and byte-identical CSR matrices |
| Item-kNN state | All state arrays byte-identical |
| Scores | Exact float equality on the sampled users |
| Metrics | Zero difference across all reported metrics |
| Domain invariance | Only the `DomainConfig` changes; train/test matrices stay byte-identical |
| Generic profile | Builds 40 × 15 with an empty blocklist |
| Percentile | Invariant under a strictly increasing score transform |

The latest result is in [`reports/verification.md`](../reports/verification.md)
and [`reports/verification.json`](../reports/verification.json). A PASS means
the reference protocol is deterministic on the recorded environment and that
domain labels are metadata, not model inputs. It does not mean the model is
valid for another market or month.

The fast mathematical invariants run in CI (`tests/test_math_verification.py`):
item-kNN equals a brute-force cosine computation on a dense fixture; percentile
ties use the upper rank; NDCG/Recall/MAP/HitRate are unchanged by a strictly
increasing score transform; changing the domain vocabulary leaves matrices and
scores identical; changing the test matrix does not change fitted scores.

## 7. CI

The GitHub Actions workflow runs:

1. `ruff check` and `ruff format --check`.
2. `mypy` on the package (non-blocking initially; see `continue-on-error`).
3. `pytest` with coverage on a synthetic dataset (no panel data required).
4. Package build and import smoke test.
5. An optional graph job that installs torch/PyG and runs the marked graph test.

CI never downloads the panel data; the tests use synthetic fixtures. The real benchmark is reproduced locally or in a protected job with access to the exports.

## 8. Known non-reproducibility

- BLAS threading can change floating-point summation order; ranking ties may flip.
- PyG random-walk order can vary across versions.
- App labels can change between exports (vendor renaming); the catalogue is snapshot-specific.
- Panel composition changes monthly; comparing across snapshots requires re-running the full protocol, not just the model.

## 9. Reporting checklist

When publishing a result, include:

- [ ] Git commit and package version.
- [ ] Dataset fingerprint and source hashes.
- [ ] Config file (or resolved `run.json`).
- [ ] Number of evaluation users and cold-start users.
- [ ] Primary metric with a paired confidence interval.
- [ ] Coverage/novelty, not just accuracy.
- [ ] Any test-set decisions disclosed.
