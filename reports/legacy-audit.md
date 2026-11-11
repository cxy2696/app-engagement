# Legacy AUC audit

The original notebook reported `ROC-AUC 0.9257` and `Precision 0.8552` for a February link-prediction task. This audit reproduces the *shape* of that evaluation with the maintained artefact and shows why the number is not a recommendation metric.

- Model audited: `itemknn`
- Users sampled: 5,000
- Catalogue prevalence of February positives: 0.201%

## Negative sampling changes the story

| Negative strategy | AUC | AP | Precision (balanced) | Recall | FPR | PPV @ 1% | PPV @ 5% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 0.8311 | 0.8354 | 0.9704 | 0.4776 | 0.0145 | 0.250 | 0.635 |
| popularity | 0.5656 | 0.5238 | 0.5715 | 0.7157 | 0.5384 | 0.013 | 0.065 |
| hard | 0.7433 | 0.8272 | 0.5950 | 0.7404 | 0.5051 | 0.015 | 0.072 |

A balanced 1:1 test can show a high AUC while precision at realistic prevalence is low. The original notebook's headline precision was computed on a balanced sample, not on the full catalogue. Reporting it as deployment precision is a category error.

## Code-level defect: the cosine read-out

```python
def cosine(u, v):
    return (u * v) / (norm_u * norm_v)   # elementwise vector, then .mean()
```

This returns a vector of elementwise products, which the notebook then averages. The correct cosine is a *sum*, not a mean. The original output printed 0.0156 for every top pair because its mean-imputed/collapsed embeddings produced a nearly constant elementwise average. The read-out destroys signal even when embeddings are healthy: in the maintained embedding matrix the same computation averages near zero, while the correct cosine has real variance.

- Embedding source: `metapath2vec.item_embeddings (fallback)`
- Observed elementwise mean: **0.000045** (std 0.001930)
- Scale of one component (1/dim): **0.015625**
- Correct cosine std across sampled pairs: **0.136036**

A near-constant read-out cannot rank apps. This is why the original top-10 lists were arbitrary and why the reported temporal consistency was zero.

## Code-level defect: February mapping mismatch

The notebook builds February embeddings in the order of `unique_apps_feb`, then filters them with indices from a *different* mapping (`app_mapping_feb`). The result mixes app identities: February top pairs contain unrelated apps with identical mean-imputed embeddings. That is a mapping bug, not evidence of temporal instability.

## What this means

The maintained benchmark does not use this protocol. It fits January only, ranks the full catalogue, masks seen apps, and reports macro-averaged ranking metrics with paired confidence intervals. See `docs/methodology.md`.
