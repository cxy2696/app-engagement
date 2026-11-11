# Showcase

EngageGraph is a **temporal recommendation instrument cluster**: a leakage-free
benchmark, calibrated evaluation, and an operable API behind a dashboard built
for at-a-glance decisions.

## Links

| Destination | Link |
| --- | --- |
| GitHub repository | <https://github.com/cxy2696/app-engagement> |
| Release / wheel | <https://github.com/cxy2696/app-engagement/releases> |
| Technical report (PDF) | [paper/engagegraph-report.pdf](../paper/engagegraph-report.pdf) |
| Evaluation report | [reports/evaluation-report.md](../reports/evaluation-report.md) |
| Legacy AUC audit | [reports/legacy-audit.md](../reports/legacy-audit.md) |
| Reproducibility verification | [reports/verification.md](../reports/verification.md) |
| API documentation | <http://127.0.0.1:8000/docs> (local service) |
| Generality guide | [generality.md](generality.md) |
| Domain profiles | [profiles/](../profiles/) |

## Screens

### App-engagement reference cluster

![App-engagement cluster](assets/dashboard.png)

### Same engine, generic retail domain

![Generic retail cluster](assets/generic-dashboard.png)

The second screen uses `profiles/generic-interactions.yaml`: 40 synthetic
shoppers, 15 products, event counts, a different support threshold, and no app
blocklist. No frontend code changed between the two screens.

## Reference result

| Model | NDCG@10 (95% CI) | Recall@10 | Coverage@10 |
| --- | ---: | ---: | ---: |
| **item-kNN** | **0.2685 [0.2660, 0.2710]** | 0.1522 | 0.013 |
| Popularity | 0.2351 [0.2327, 0.2376] | 0.1347 | 0.006 |
| MetaPath2Vec | 0.1639 [0.1622, 0.1660] | 0.0933 | 0.015 |
| ALS | 0.0993 [0.0974, 0.1013] | 0.0798 | 0.317 |

Dataset fingerprint: `2b8d9290d5614afa22f6`. 34,024 users × 9,848 apps;
28,286 evaluation users; 10,875 cold-start users. See the
[evaluation report](../reports/evaluation-report.md) for CIs, Holm-corrected
comparisons, and coverage/novelty trade-offs.

## What to look at

- **Model telemetry:** NDCG@10 gauge, hit rate, recall, coverage.
- **Workspace:** panelist lookup, sample IDs, `k`, explanation and diversity
  controls.
- **Ranked slate:** score, within-user percentile (`TOP x%`), and up to two
  January co-usage reasons per item.
- **Provenance strip:** dataset fingerprint, cold-start count, model, latency.
- **Links:** report, release, source, API, showcase.

## Run it locally

```bash
make demo
make serve
# open http://127.0.0.1:8000
```

Or serve any artefact:

```bash
engagegraph serve --artifact artifacts/runs/<run>/artifact --port 8000
```

## Reproduce the screens

The screenshots are captured with `scripts/screenshot.py` from the running API.
The app screen uses `artifacts/demo`; the generic screen uses
`artifacts/runs/generic/<run>/artifact` after:

```bash
engagegraph data build --config profiles/generic-interactions.yaml --out data/processed-generic
engagegraph benchmark --config profiles/generic-interactions.yaml
```
