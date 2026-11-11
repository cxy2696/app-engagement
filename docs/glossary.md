# Glossary

Short definitions for terms used across the documentation. Formulas are in
[methodology.md](methodology.md); serving policy is in [algorithm.md](algorithm.md).

| Term | Meaning |
| --- | --- |
| Training period | The only period used to fit mappings, catalogue statistics, and model parameters. In the reference run: January 2024. |
| Test period | The period used once to create adoption labels. In the reference run: February 2024. |
| Candidate set \(C_u\) | Catalogue items not used by user \(u\) in the training period. |
| Adoption label | A test-period interaction with a catalogue item the user did not use in training. |
| Cold-start user | A test-period user with no training-period history. Reference count: 10,568. |
| Ineligible user | A user with training history but fewer than three catalogue items. Reference count: 307. |
| Item-kNN | Item-based collaborative filtering: score an item by the similarity mass from the user's history. |
| NDCG@K | Normalised discounted cumulative gain at K with binary relevance; primary metric. |
| Recall@K | Fraction of a user's adoption set retrieved in the top K. |
| Coverage@K | Fraction of catalogue items appearing in at least one user's top K. |
| Novelty@K | Mean self-information of recommended items under training-period popularity. |
| Percentile \(q_{u,i}\) | Fraction of the user's unseen candidates scored at or below item \(i\). Relative confidence, not a probability. |
| `min_percentile` | Serving filter that keeps candidates at or above the empirical quantile. |
| Paired bootstrap | User-level resampling of per-user metric differences, used for confidence intervals and p-values. |
| Artefact | Pickle-free model directory: manifest, mappings, model arrays, training history. |
| Domain profile | Configuration of vocabulary, paths, and optional serving blocklist; does not change the model. |
| MMR | Maximal marginal relevance: relevance minus a penalty for similarity to already selected items. |
