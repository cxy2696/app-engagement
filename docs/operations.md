# Operations runbook

This runbook is for the person on call for an EngageGraph deployment. It assumes the API is behind a reverse proxy and Prometheus scrapes `/metrics`.

## 1. Service level objectives

| SLO | Target | Measurement |
| --- | --- | --- |
| Availability | 99.5% monthly | non-5xx responses / total |
| Read latency | p95 < 250 ms, p99 < 750 ms | `X-Response-Time-ms` / proxy histogram |
| Error rate | < 1% of requests | 5xx / total |
| Freshness | model ≤ 45 days old | artefact `created_at` |
| Feedback durability | ≤ 5 min loss | JSONL flush + shipping lag |

These are starting points, not contractual SLAs. Adjust after measuring real traffic.

## 2. Dashboards and alerts

Scrape `/metrics` and chart:

- request rate by route/status;
- mean latency (replace with a histogram in production);
- feedback events;
- process uptime;
- artefact age (from `/v1/meta`).

Alert candidates:

| Alert | Condition | First response |
| --- | --- | --- |
| API down | `/readyz` failing for 2 min | check process, artefact path, disk |
| Elevated 5xx | > 2% for 5 min | check logs for traceback, roll back if new release |
| Latency regression | p95 > 500 ms for 10 min | check CPU/GC, batch size, concurrent load |
| No feedback | 0 events for 24 h during business hours | check client instrumentation |
| Stale model | artefact age > 45 days | schedule retrain |
| Coverage collapse | `coverage@10` drops > 50% vs baseline | inspect model/popularity drift |

## 3. Routine cadence

**Daily:** confirm health, error rate, latency, and feedback ingestion.

**Weekly:** inspect top recommendations and explanations for a sample of panelists; check for obviously inappropriate apps; review feedback click-through; confirm artefact age.

**Monthly:** retrain on the latest complete month; evaluate on the next month; compare against the incumbent with paired bootstrap; update the registry; archive the previous artefact; review drift.

**Quarterly:** dependency upgrades, security review, data licence review, documentation refresh, and a fresh look at the evaluation protocol.

## 4. Retraining

```bash
# 1. Build a new benchmark from the latest complete months.
engagegraph data build --jan <new-jan> --feb <new-feb> --out data/processed-<version>

# 2. Run the benchmark; keep all artefacts.
ENGAGEGRAPH_DATA=data/processed-<version> engagegraph benchmark --config configs/benchmark.yaml

# 3. Compare against the incumbent on the same held-out month.
engagegraph evaluate --artifact <incumbent>/artifact --dataset data/processed-<version>
engagegraph evaluate --artifact <candidate>/artifact --dataset data/processed-<version>

# 4. Promote only if the paired difference is credible and coverage is stable.
```

Promotion is a registry update plus a restart. Keep the previous artefact and the exact config for rollback.

## 5. Drift review

Monitor inputs and outputs separately:

- **Input drift:** catalogue size, new-app share, item support distribution, weight quantiles, panelist count.
- **Output drift:** score distribution, coverage@10, Gini@10, novelty@10, top-app share, cold-start rate.
- **Label drift:** next-month adoption rate, repeat rate, app churn.

A drift alert is a prompt to investigate, not an automatic retrain. Retraining on a shifted population can bake in the shift.

## 6. Incident runbook

### 6.1 API returns 503

1. `GET /healthz` and `GET /readyz`.
2. Check `ENGAGEGRAPH_ARTIFACT` / registry path exists and `manifest.json` is readable.
3. Verify the artefact schema version matches the package.
4. Roll back to the previous artefact directory and restart.
5. If the artefact is corrupt, restore from backup/release; do not edit in place.

### 6.2 High 5xx rate

1. Check logs for tracebacks and the first failing route.
2. If it started after a deploy, roll back the image and artefact.
3. If it is isolated to one route, disable it at the proxy and open an incident.
4. Capture the request ID and a redacted payload; never log raw panelist IDs.

### 6.3 Latency spike

1. Check CPU, memory, and worker count.
2. Confirm traffic volume and batch sizes.
3. Reduce `k` or batch size at the proxy/client if possible.
4. Scale workers or replicas; restart stuck workers.
5. If latency is model-specific, compare `X-Response-Time-ms` across routes.

### 6.4 Recommendations look wrong

1. Reproduce with a known panelist and `explain=true`.
2. Check the artefact fingerprint and creation date on `/v1/meta`.
3. Confirm the item catalogue and mappings match the dataset.
4. Run the offline benchmark on the same artefact; if metrics are normal, suspect a client/integration issue.
5. If metrics regressed, roll back and schedule a root-cause review.

### 6.5 Data corruption or leak suspicion

1. Stop writes to the affected store; preserve logs.
2. Rotate credentials if exposure is possible.
3. Notify the data owner and follow the security policy.
4. Do not attempt to "clean" the only copy; restore from an immutable backup.

## 7. Backup and rollback

- Artefacts: immutable directories, keep last 6 monthly versions plus the current.
- Processed data: rebuildable; keep manifests and source hashes.
- Feedback: daily files, shipped to durable storage; retain per privacy policy.
- Rollback: point the registry `active` field or `ENGAGEGRAPH_ARTIFACT` at the previous artefact and restart. No database migration is involved.

## 8. Security operations

- Rotate `ENGAGEGRAPH_API_KEY` on a schedule and after any suspected exposure.
- Keep dependencies patched; rebuild images monthly even without code changes.
- Restrict egress from the service.
- Review access to the feedback store and raw exports quarterly.
- Follow [`SECURITY.md`](../SECURITY.md) for disclosure.
