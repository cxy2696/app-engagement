# FAQ

### Is the 0.9257 AUC from the original notebook wrong?

It is not comparable to a recommendation metric. It was computed on February edges, with a 75/25 split of those edges, and negatives selected by similarity. The model also applied an elementwise operation and averaged it, producing the constant 0.0156 observed in the notebook output. The number can be reproduced as a link-prediction diagnostic, but it does not estimate next-month ranking quality. See [`reports/legacy-audit.md`](../reports/legacy-audit.md).

### Why is item-kNN stronger than ALS / MetaPath2Vec?

The task rewards apps that are popular *and* co-used with a user's existing repertoire. Item-kNN directly encodes that signal. ALS optimises reconstruction and often surfaces niche items (high novelty, high coverage, low NDCG). MetaPath2Vec's learned user vectors are a compressed summary that is not optimised for top-K ranking. This is a finding, not a bug; the report discusses the trade-off.

### Why is the weight unit milliseconds?

The median weight (≈ 415,120) and the original column name `FOREGROUNDDURATION` make milliseconds the most plausible unit. The vendor codebook is not in the repository, so this is documented as an inference. All primary metrics are invariant to a global rescale.

### Why exclude February-only apps?

Including them would leak the future catalogue into the candidate set. The benchmark measures adoption among apps that existed and had enough support in January. February-only apps are counted as out-of-catalogue.

### Why are cold-start users not in the headline number?

A January-fitted personalised model has no signal for a user with no January history. Including them in the same denominator would mix personalisation with a popularity prior. The API returns an explicit `cold_start_popularity` strategy; the data card reports the count.

### Can I use this on my own data?

Yes, if you have lawful access. Convert to the three canonical columns (`user`, `item`, `weight`), set the paths in a `DataConfig`, and run `engagegraph data build`. The package does not require the panel data. Review [`DATA_LICENSE.md`](../DATA_LICENSE.md) for the shipped exports.

### How do I add a model?

Implement `BaseRecommender` (`fit`, `score_users`, `get_state`, `set_state`), register it in `models/base.py`, add a config, and run the benchmark. A new model must beat the baselines on paired users, not just on a balanced AUC.

### Is the API ready for a production deployment?

It is a reference implementation with the operational basics: health checks, request IDs, metrics, feedback capture, Docker, and runbooks. Before public deployment, add authentication, rate limiting at the proxy, TLS, and a privacy review. See [`docs/deployment.md`](deployment.md).

### Why no deep sequence model?

The exports are monthly aggregates, not event sequences. A sequence model would require data that is not in the repository. Adding one without that data would be a misleading benchmark.
