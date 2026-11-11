# Generic synthetic example

A tiny **synthetic** dataset used by `profiles/generic-interactions.yaml` and the
generality tests. It is intentionally not real user data.

Schema: `user_id`, `item_id`, `event_count`, `event_time`.

```bash
engagegraph data build --config profiles/generic-interactions.yaml --out data/processed-generic
engagegraph benchmark --config profiles/generic-interactions.yaml
engagegraph serve --artifact artifacts/runs/<run>/artifact --port 8001
```
