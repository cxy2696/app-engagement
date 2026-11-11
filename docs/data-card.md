# Data card: January–February 2024 app-engagement panel exports

This card follows the spirit of *Datasheets for Datasets* (Gebru et al., 2021). It documents what is known, what is inferred, and what is **not** known about the data shipped in `data/raw/`.

## 1. Identity

| Field | Value |
| --- | --- |
| Files | `jan_aggregated.parquet` (19.8 MB), `feb_aggregated.parquet` (22.1 MB) |
| SHA-256 | Recorded automatically in `data/processed/manifest.json` at build time |
| Format | Apache Parquet, one row per `(panelist, app)` pair per month |
| Schema | `PANELISTID: string`, `APPDESCRIPTION: string`, `weight: decimal` |
| Grain | Monthly aggregate; **not** sessions, events, or daily observations |
| Row counts | January 2,210,089; February 2,429,195 |
| Distinct panelists | January 34,639; February 40,001 |
| Distinct app labels | January 74,754; February 80,277 |
| Weight range | 1,001 – 109,614,008,181 (median 415,120) |

## 2. Provenance

The files are aggregated monthly exports derived from daily files named `data/2024/01/data_202401DD.parquet` in the original notebook. Those daily files are **not** included in this repository. The original notebook sums `FOREGROUNDDURATION` by `(PANELISTID, APPDESCRIPTION)`; the shipped files are the result of that aggregation.

**What is not documented:** the panel vendor, recruitment and consent process, sampling frame, weighting, device/platform coverage, app-label taxonomy, collection dates, or redistribution licence. The repository owner should add a vendor citation and licence before any redistribution beyond research reproducibility. Until then, the data is marked **research use only** in [`DATA_LICENSE.md`](../DATA_LICENSE.md).

## 3. Weight semantics

`weight` is interpreted as **foreground duration in milliseconds**. Evidence:

- Median 415,120 → ≈ 6.9 minutes per app per panelist per month, plausible for foreground time.
- 90th percentile 55.9 million → ≈ 15.5 hours; plausible for heavy apps.
- Maximum 109.6 billion → ≈ 3.5 years, clearly an outlier or a sentinel value.
- The original notebook names the source column `FOREGROUNDDURATION` and never converts units.

This is an **inference**, not a codebook fact. All benchmark metrics use relative weights (and binary relevance), so a global unit error would not change the rankings. Do not report absolute hours from this dataset without vendor confirmation.

## 4. Quality profile

- **Missing values:** none in the three columns after decimal parsing.
- **Duplicates:** none at `(PANELISTID, APPDESCRIPTION)` grain in the shipped monthly files.
- **Outliers:** heavy right tail; the 99.9th percentile clip is applied in the pipeline. The maximum is treated as a data-quality outlier.
- **Labels:** free-text app names, including non-English names, mojibake (`ParamÃ¨tres`), trailing spaces, full-width characters, and generic system apps (`Android System`, `Permission controller`). NFKC normalisation and whitespace collapsing are applied; semantic aliases are **not** merged.
- **Coverage:** 93.4% of February rows for warm users fall inside the January catalogue (apps with ≥10 January users, capped at 9,848 items). The remaining 6.6% are February-only or low-support apps and are out of scope.
- **Panel churn:** 10,568 February users have no January history; 307 have January history but fewer than three catalogue items and are ineligible; 28,286 users have at least one eligible February adoption.

## 5. Benchmark construction

| Step | January | February |
| --- | --- | --- |
| Clean | drop missing/negative, NFKC, dedupe | same |
| Catalogue | apps with ≥10 users, cap 10,000 | restricted to January catalogue |
| Users | ≥3 catalogue apps | January users only for personalisation |
| Matrix | `log1p(clip(w, q99.9))` | binary relevance for unseen apps |
| Final size | 34,024 × 9,848; 2,077,959 interactions | 561,231 targets; 28,286 eval users |

Full protocol: [`methodology.md`](methodology.md).

## 6. Privacy and ethics

- Identifiers are pseudonymous panelist IDs, not names. They are still potentially re-identifiable when combined with external data.
- The data contains no demographics, location, content, or device identifiers in the shipped columns.
- App-usage telemetry can reveal sensitive interests (health, dating, finance, religion). Do not attempt re-identification, do not join to external data, and do not publish per-panelist profiles.
- The demo API accepts raw panelist IDs because it is intended for local research use. A public deployment must add authentication, hash identifiers at ingest, and restrict bulk export.
- The panel is opt-in and may not represent the general population. Any product decision based on it should be validated on the operator's own telemetry.

## 7. Intended use

**In scope:** benchmarking temporal next-app recommendation, comparing collaborative filtering and graph embeddings, teaching reproducible ML engineering, and prototyping an internal recommendation service.

**Out of scope:** causal claims about app switching, advertising targeting, individual-level profiling, clinical or financial decisions, and any use that requires demographic fairness guarantees (the data has no demographics).

## 8. Maintenance

- The raw exports are immutable; new months should be added as new files with hashes, not overwritten.
- Rebuild the processed benchmark with `engagegraph data build`; the manifest records source hashes, config, and quality counters.
- If the vendor provides a codebook, update the weight-unit field and re-run the sensitivity analysis; do not silently change units.
- Report data issues at <https://github.com/cxy2696/app-engagement/issues>.

## 9. Other domains

The data card describes the app reference dataset. For the generic data
contract and how to adapt the engine to products, content, or SaaS features,
see [`generality.md`](generality.md). The app blocklist and millisecond weight
assumption live in the app profile, not in the engine.

## 10. Citation

If you use the data, cite the repository and, when available, the panel vendor. The repository's [`CITATION.cff`](../CITATION.cff) provides a software citation; it does not grant rights to the data.
