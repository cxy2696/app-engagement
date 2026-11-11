# Domain profiles

EngageGraph's engine is domain-agnostic. A **profile** supplies the vocabulary,
file layout, and optional serving blocklist for a domain.

| Profile | Domain | Notes |
| --- | --- | --- |
| [`app-engagement.yaml`](app-engagement.yaml) | mobile apps | Reference benchmark; blocklists OS components. |
| [`generic-interactions.yaml`](generic-interactions.yaml) | synthetic retail | CSV + event counts; no app assumptions. |
| [`ecommerce.example.yaml`](ecommerce.example.yaml) | e-commerce | Template; merge into a full config. |

## How to adapt

1. Copy a profile and change `data.*` to your paths and column names.
2. Set `domain.*` labels. They appear in the API, dashboard, and artefact metadata.
3. Set `domain.blocklist` for items that should never be recommended (optional).
4. Set `processed_dir` and `output_dir` to domain-specific folders.
5. Run `engagegraph data build --config <profile>` and `engagegraph benchmark --config <profile>`.
6. Keep `weight_col: null` to treat each row as a binary interaction.

The model code never reads `domain`; it only changes labels and the optional
serving blocklist. This means the same evaluation and artefact format apply to
apps, products, articles, courses, or SaaS features.

See [`docs/generality.md`](../docs/generality.md) for the data contract and
domain-specific caveats.
