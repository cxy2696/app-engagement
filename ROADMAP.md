# Roadmap

This roadmap is directional, not a commitment. Priorities are informed by issues and by what makes the benchmark more trustworthy.

## Now (2.x)

- [x] Leakage-free temporal benchmark and four reference models.
- [x] Paired bootstrap comparisons and Holm correction.
- [x] Pickle-free artefacts, registry, API, dashboard, Docker, CI.
- [x] Data card, model card, methodology, operations runbook.
- [ ] Add a second time window (e.g., February → March) to test protocol stability.
- [ ] Add an EASE or BPR baseline with a tuned, pre-registered protocol.

## Next

- [ ] Cold-start model using app-name text embeddings (no demographics).
- [ ] Sequential model once event-level data is available and licensed.
- [ ] Approximate nearest-neighbour candidate generation for >100k catalogues.
- [ ] Online evaluation hooks: impression logging, position-bias-aware estimators.
- [ ] Fairness audit with an appropriately licensed demographic slice.
- [ ] Parquet/Arrow-native pipeline for >100M interactions.

## Later

- [ ] Multi-tenant artefact registry with signed manifests.
- [ ] Kubernetes Helm chart and autoscaling guidance.
- [ ] Reproducible benchmark leaderboard with submission validation.
- [ ] Integration with experiment-tracking systems (MLflow/W&B) as optional adapters.

## Non-goals

- Public individual-level profiling.
- Causal claims of app switching from observational panel data.
- Replacing a domain-specific production recommender without online validation.
