# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [2.3.0] — 2026-11-12

### Added

- Formal ranking framework in `docs/algorithm.md`: fit → score → mask → rank/policy → evaluate, with domain configuration explicitly outside the model.
- `min_percentile` on `/v1/users/{id}/recommendations` and batch scoring: keeps only candidates above an empirical quantile; the returned percentile is still computed against the full unseen set.
- Shareable dashboard state (`user`, `k`, `explain`, `diversify`, `min_percentile`, `mode`, `item`), a copy-link button, a minimum-percentile slider, and a per-row `SIMILAR` action.
- API tests for threshold filtering and deep-link interaction checks.
- Documentation for the threshold policy, complexity, and query semantics.

### Changed

- Default result-card layout now exposes the percentile and a direct similar-item path.
- `docs/showcase.md` reduced to screenshots, links, and the reference result.

## [2.2.1] — 2026-11-12

### Fixed

- Cold-start count now excludes the 307 February users who have January history but fewer than three catalogue items. Both groups are reported separately in the dataset manifest, README, data/model cards, reports, and paper. The benchmark metrics are unchanged; the dataset fingerprint changes because the recorded metadata changes.

### Changed

- Documentation pass: shorter README, plain-language protocol notes, corrected build commands (`--config` retained), and an explicit distinction between cold-start and ineligible users.
- Removed marketing wording and an unverifiable AUC figure from the report prose.
- Institutional contact: `xc535@cornell.edu`.

## [2.2.0] — 2026-11-12

### Added

- Instrument-cluster dashboard: dark telemetry UI, central NDCG gauge, mini hit-rate/recall gauges, provenance strip, and responsive layout inspired by automotive clusters.
- Within-user score percentile (`percentile` in the recommendation response; `TOP x%` in the UI) as a distribution-free, cross-user confidence signal.
- `docs/algorithm.md` documenting scoring, percentile math, decision policy, MMR, complexity, and explicit non-claims.
- `docs/showcase.md` with screenshots and links to release, report, audit, API, and profiles.
- Optional `data.domain.blocklist` filtering in user profiles.

### Changed

- Dashboard labels and reasons now use domain vocabulary and training-period labels.
- Removed duplicated `paper/figures/` copies, the duplicate `reports/engagegraph-technical-report.pdf`, and the duplicate full `profiles/app-engagement.yaml` (the canonical app profile is `configs/benchmark.yaml`).
- Removed the metrics table from the dashboard; the reference table lives in the showcase and report.

### Fixed

- `[hidden]` modes no longer render due to CSS display rules.
- Gauge labels scale dynamically when the NDCG axis maximum changes.

## [2.1.0] — 2026-11-12

### Added

- Domain profiles: `data.domain` controls user/item vocabulary, task labels, display tagline, and an optional serving blocklist.
- Generic data paths and formats: `train_path`/`test_path`, CSV/TSV/JSONL/Parquet, optional `weight_col` (unit weights), optional `time_col`.
- `configs/benchmark.yaml` (app reference), `profiles/generic-interactions.yaml`, and `profiles/ecommerce.example.yaml`, plus a synthetic generic CSV example.
- Canonical `/v1/items/similar` endpoint; `/v1/apps/similar` remains a backwards-compatible alias.
- Generic profile fields (`n_items`, `top_items`) while preserving 2.0 keys (`n_apps`, `top_apps`).
- Domain-aware dashboard labels driven by `/v1/meta`.
- `docs/generality.md` and tests for CSV ingestion, unit weights, domain blocklists, and API aliases.

### Changed

- Blocklist moved from a hard-coded serving default into the dataset domain profile. Legacy 2.0 artefacts still receive the historical app blocklist.
- `ENGAGEGRAPH_BLOCKLIST` environment override added for operations.
- CLI `data build` accepts a full profile via `--config` and `--train`/`--test` aliases.

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
