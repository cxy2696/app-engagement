# Architecture

EngageGraph is intentionally a modular monolith: one installable package, one CLI, one HTTP service, and one artefact format. The goal is operational simplicity without sacrificing the boundaries that make the system testable and replaceable.

## 1. Data flow

```mermaid
flowchart TD
    A[Monthly Parquet exports] --> B[data.schema.clean_interactions]
    B --> C[data.preprocess.build_dataset]
    C --> D[(data/processed<br/>dataset.npz · mappings.json · manifest.json)]
    D --> E[models.fit]
    E --> F[(artefact<br/>model.npz · history.npz · manifest.json)]
    D --> G[evaluation.evaluate]
    F --> G
    G --> H[reports/ · metrics.json · per_user.npz]
    F --> I[recommend.RecommendationService]
    I --> J[FastAPI /v1]
    J --> K[Dashboard]
    J --> L[/metrics]
    J --> M[Feedback JSONL]
```

## 2. Boundaries

| Boundary | Responsibility | Why it matters |
| --- | --- | --- |
| `data.schema` | validation, Unicode normalisation, row-level audit | Bad input fails before it reaches a model |
| `data.preprocess` | temporal split, catalogue, CSR matrices, fingerprints | Leakage controls live in one auditable place |
| `models` | fit and score only; no access to February labels | Models cannot accidentally train on test data |
| `evaluation` | masking, metrics, bootstrap, comparisons | One protocol for every model |
| `artifacts` | pickle-free persistence, manifest, registry | Reproducible serving and rollback |
| `recommend` | cold start, explanations, diversity | Serving policy is separate from model math |
| `api` | HTTP contracts, health, feedback | Operational surface with stable semantics |
| `cli` | operator workflows | Everything scriptable in CI and cron |

The evaluator masks seen items, not the model. This is deliberate: a model that forgets to mask cannot silently win, and the serving layer uses the same masking code path.

## 3. Artefact format

```text
artifact/
├── manifest.json    # schema version, package version, git commit, model meta,
│                    # dataset fingerprint, metrics, hashes, creation time
├── mappings.json    # ordered panelist and item identifiers
├── model.npz        # model-specific NumPy arrays (factors, neighbours, scores)
└── history.npz      # January CSR matrix for masking, exact item-kNN scoring,
                     # co-usage explanations, and profiles
```

No pickle is used. Loading an artefact cannot execute arbitrary code. The manifest records an `artifact_fingerprint` over the model, mappings, and history payloads so a served model can be traced to an evaluation run.

## 4. Serving path

1. `create_app()` resolves an artefact from `--artifact`, `ENGAGEGRAPH_ARTIFACT`, `artifacts/registry.json`, or `artifacts/demo`.
2. `RecommendationService` loads mappings, history, and model once at startup.
3. A request validates `k`, resolves the panelist, scores the full catalogue, masks January items, ranks, and attaches up to two co-usage reasons.
4. Unknown panelists receive popularity recommendations with `strategy="cold_start_popularity"`; the response shape is unchanged.
5. The response carries `X-Request-ID` and `X-Response-Time-ms`; counters feed `/metrics`.

The service is read-only. Feedback is appended to JSONL and never fed back into the model online; retraining is an explicit offline workflow.

## 5. Scaling notes

- **Scoring:** full-catalogue scoring is batched by `evaluation.eval_batch_size` (default 512). At 9,848 items this is ~5M floats per batch, well within a small container.
- **Serving:** the current artefact is fully in memory (~20 MB for ALS plus history). A 10k-item catalogue is comfortably single-process; for millions of items, replace `score_users` with an ANN candidate generator and keep the same evaluation contract.
- **Training:** ALS and item-kNN run in seconds to minutes on CPU. MetaPath2Vec is CPU-bound by random walks; the graph job is optional in CI.
- **Storage:** artefacts are content-addressed by fingerprint; keep the last N versions and prune by policy.

## 6. Extension points

- **New model:** implement `BaseRecommender` (`fit`, `score_users`, `get_state`, `set_state`) and register it in `models.base.build_model`.
- **New metric:** add a per-user array in `evaluation.metrics.evaluate`; it automatically participates in bootstrap comparisons if named consistently.
- **New data source:** adapt to the three canonical columns and call `build_dataset`; the rest of the stack is source-agnostic.
- **New surface:** the service is transport-independent; a batch scorer or gRPC adapter can reuse `RecommendationService`.

## 7. Failure modes

| Failure | Detection | Behaviour |
| --- | --- | --- |
| Missing artefact | `/healthz` degraded, startup log | `/v1/*` returns 503; dashboard shows instructions |
| Corrupt artefact | manifest/schema check at load | startup fails loudly; previous version stays on disk |
| Unknown user | lookup miss | popularity fallback, `strategy` labelled |
| Unknown app | lookup miss | 404 with a clear message |
| Invalid `k` | Pydantic/FastAPI validation | 422 |
| Feedback store unwritable | write exception | 500; event not acknowledged |
| Model regression | benchmark comparison + CI | release blocked; rollback via registry |

## 8. Design decisions

- **Why not a notebook-first design?** Notebooks are excellent for exploration but poor at enforcing leakage controls, versioning, and failure semantics. The package is the product; notebooks are downstream consumers.
- **Why popularity and item-kNN?** They are cheap, interpretable, and strong on adjacent-month adoption. A complex model must beat them on paired users, not just on a balanced AUC.
- **Why no deep sequence model?** The export is monthly aggregate, not event sequences. A session model would require data that is not in the repository; pretending otherwise would violate the provenance contract.
- **Why JSON + NPZ?** Portable, auditable, and safe by default. Pickle would be smaller but creates a code-execution surface and version-coupling problems.
