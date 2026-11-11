# Evaluation guide

This guide gives the exact commands, definitions, and expected outputs for reproducing the benchmark.

## 1. Build the dataset

```bash
engagegraph data build --config configs/benchmark.yaml --out data/processed
```

Expected summary (v2.3.1, default config):

```text
Domain              : app-engagement (apps)
Users x items       : 34,024 x 9,848
Train interactions  : 2,077,959 (density 0.62%)
Test targets        : 561,231 across 28,286 users
Targets per user    : mean 19.8, median 15
February cold users : 10,568
Ineligible users    : 307
Catalogue coverage  : 93.4% of February rows for warm users
```

The fingerprint depends on the source files and config; it is printed and stored in `data/processed/manifest.json`.

## 2. Run the benchmark

```bash
engagegraph benchmark --config configs/benchmark.yaml
```

Outputs:

```text
artifacts/runs/
├── benchmark.json                  # aggregate results, comparisons, timings
├── comparisons.json                # paired bootstrap table
├── popularity-<timestamp>/
│   ├── artifact/                   # served model
│   ├── evaluation/metrics.json
│   ├── evaluation/per_user.npz
│   └── run.json
├── itemknn-<timestamp>/
├── als-<timestamp>/
└── metapath2vec-<timestamp>/
```

`benchmark.json` is the machine-readable source of truth. `run.json` records the Python/platform versions, seed, model config, dataset fingerprint, and timings.

## 3. Metric definitions

| Metric | Definition | Interpretation |
| --- | --- | --- |
| `recall@k` | fraction of a user's February new apps retrieved in top k | coverage of the adoption set |
| `ndcg@k` | DCG / ideal DCG with binary relevance | early placement quality; **primary** |
| `map@k` | mean average precision at k | precision-weighted ranking quality |
| `hit_rate@k` | 1 if at least one hit | product-style success rate |
| `precision@k` | hits / k | catalog efficiency |
| `mrr@k` | reciprocal rank of first hit | how early the first success appears |
| `coverage@10` | unique recommended items / catalogue | catalog breadth |
| `gini@10` | inequality of recommendation exposure | concentration / filter bubble |
| `novelty@10` | mean −log₂ January support share | how non-obvious the recommendations are |

All metrics are macro-averaged over evaluation users. See [`methodology.md`](methodology.md) §4 for formulas.

## 4. Statistical comparison

`benchmark.json` contains, for each model:

```json
{
  "candidate": "itemknn",
  "baseline": "popularity",
  "metric": "ndcg@10",
  "difference": 0.0334,
  "ci_low": 0.0301,
  "ci_high": 0.0368,
  "p_value": 0.001,
  "p_holm": 0.003,
  "n_users": 28286
}
```

The interval is a paired user-level bootstrap over 1,000 resamples. `p_holm` is the Holm–Bonferroni adjustment across the family of comparisons against the baseline. A difference is reported as evidence, not proof: it is conditional on this panel, month pair, and protocol.

## 5. Slices and sensitivity

- **Cold start:** 10,568 February users have no January history. A further 307 have January history but fewer than three catalogue items; they are ineligible and reported separately. Neither group is in the personalised ranking denominator. `engagegraph data stats` reports both counts; the API returns `strategy="cold_start_popularity"` for any identifier outside the fitted cohort.
- **Weight transform:** rerun with `--weight-transform binary` and a different `--out`/config to test sensitivity. The primary benchmark uses `log1p`.
- **Support threshold:** `--min-item-users 5` or `20` changes the catalogue; rerun the benchmark and report the change rather than mixing catalogues.
- **Model hyperparameters:** primary configs fixed before February. Post-hoc MetaPath2Vec and item-kNN diagnostics are disclosed in [`methodology.md`](methodology.md) §3.5; exploratory sweeps are labelled, not used to select the headline model.

## 6. Generate the report

```bash
engagegraph report \
  --benchmark artifacts/runs/benchmark.json \
  --out reports
```

This writes `reports/evaluation-report.md`, `reports/results.json`, and figures under `reports/figures/`. The technical report in `docs/paper/` consumes the same JSON.

## 7. Interpreting a result

A useful checklist before believing a number:

1. Is the dataset fingerprint the same across models?
2. Are seen items masked? (The evaluator does this centrally.)
3. Is the model fitted on January only?
4. Is the metric macro-averaged, and is the user count reported?
5. Is the difference larger than the paired bootstrap interval?
6. Does coverage/novelty reveal a degenerate popularity strategy?
7. Is the cold-start population reported separately?

If any answer is no, the comparison is not valid under this protocol.
