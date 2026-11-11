# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-11-11

### Added

- Typed `engagegraph` package with data, model, evaluation, serving, and CLI layers.
- Leakage-free January → February temporal benchmark with a published protocol and dataset fingerprint.
- Four reference models: popularity, item-kNN, implicit ALS, and MetaPath2Vec.
- Full-catalogue ranking evaluation with NDCG, Recall, MAP, HitRate, MRR, coverage, Gini, and novelty.
- Paired user-level bootstrap confidence intervals and Holm–Bonferroni comparisons.
- Pickle-free artefact format (`manifest.json`, `model.npz`, `history.npz`, `mappings.json`) with a registry.
- FastAPI service with cold-start fallback, explanations, batch scoring, feedback capture, health checks, and Prometheus metrics.
- Zero-build dashboard with panelist lookup, recommendation reasons, similarity search, and model provenance.
- Dockerfile, Compose file, Makefile, CI, pre-commit, tests, data card, model card, methodology, deployment, operations, and governance docs.
- Reproducible report generator and LaTeX technical report.
- Legacy AUC audit documenting why the original 0.9257 is not comparable.

### Changed

- Replaced the original notebook's transductive February evaluation with a temporal ranking protocol.
- Replaced the elementwise "cosine" read-out with the canonical metapath2vec dot-product objective.
- Documented the inferred millisecond weight unit and added 99.9th-percentile clipping.

### Removed

- Runtime dependence on the original monolithic notebook. The notebook remains in git history for provenance.

### Fixed

- Constant 0.0156 cosine scores and zero common app pairs in the original analysis.
- Silent cold-start imputation; cold-start users are now reported and served with an explicit fallback.

## [1.0.0] — 2025-09-30

- Initial research upload: aggregated January/February exports, MetaPath2Vec graph and weights, prediction notebook, and a Word summary of app-switching probabilities.
