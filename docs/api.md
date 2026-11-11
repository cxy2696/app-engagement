# API reference

Base URL: `http://127.0.0.1:8000`. Interactive schema: `/docs`. OpenAPI JSON: `/openapi.json`. All responses are JSON unless noted.

The API is domain-agnostic. `item` is the canonical resource; `app` aliases are
kept for the reference dataset. Domain labels are reported by `/v1/meta` and the
dashboard adapts automatically.

## 1. Conventions

- `k` is an integer in `[1, 100]`; larger requests are rejected by validation.
- Identifiers are URL-encoded; app names may contain spaces and punctuation. Use the query parameter form for similarity search.
- Successful responses include `X-Request-ID` and `X-Response-Time-ms`.
- Errors return `{"detail": "..."}` with 4xx/5xx status codes.
- There is no API version in the path beyond `/v1`; breaking changes will use `/v2`.

## 2. Endpoints

### `GET /healthz`

Liveness and model status. Always 200 if the process is alive.

```json
{"status": "ok", "model_loaded": true, "version": "2.1.0", "error": null}
```

### `GET /readyz`

Readiness. 200 when an artefact is loaded, 503 otherwise.

### `GET /version`

```json
{"name": "engagegraph", "version": "2.1.0"}
```

### `GET /v1/meta`

Model, dataset, and artefact provenance. The dashboard uses this on load.

```json
{
  "model": {"model_type": "itemknn", "params": {"k": 100}},
  "domain": {"name": "app-engagement", "item_label": "app", "item_label_plural": "apps"},
  "labels": {"user": "panelist", "item": "app", "train_period": "January 2024"},
  "dataset": {"fingerprint": "822b17a4209c607144d1", "n_users": 34024, "n_items": 9848, "stats": {...}},
  "metrics": {"ndcg@10": 0.2685, "recall@10": 0.1522, ...},
  "artifact": {"created_at": "...", "package_version": "2.1.0", "fingerprint": "..."},
  "sample_users": ["...", "..."]
}
```

### `GET /v1/popular?k=10`

Global popularity ranking from the artefact's January history.

```json
{"items": [{"item_id": "Facebook", "score": 9.1, "rank": 1, "explanations": [...]}]}
```

### `GET /v1/items/similar?item_id=Spotify&k=10`

Nearest catalogue items. `/v1/apps/similar` is a backwards-compatible alias.
Returns 404 for an unknown item.

```json
{"item_id": "Spotify", "items": [{"item_id": "YouTube Music", "score": 0.42, "rank": 1}]}
```

### `GET /v1/users/{user_id}/recommendations?k=10&explain=true&diversify=false`

Personalised ranking, with January items masked. Unknown users receive a popularity fallback with a different `strategy`.

```json
{
  "user_id": "1000123",
  "strategy": "personalized",
  "model": "itemknn",
  "generated_at": "2026-11-11T12:00:00+00:00",
  "latency_ms": 3.2,
  "items": [
    {
      "item_id": "Discord",
      "score": 1.234,
      "rank": 1,
      "explanations": [
        {"item_id": "Twitch", "reason": "users of Twitch also used this app", "contribution": 0.11}
      ]
    }
  ]
}
```

`strategy` is one of `personalized`, `cold_start_popularity`, or `exhausted_catalog_popularity`.

The serving layer removes a small blocklist of OS components and measurement agents (for example `Android System`, `Settings`, `Permission controller`) that are not actionable app choices. The offline benchmark keeps the full catalogue so its metrics remain reproducible; `/v1/meta` reports `serving_blocklist_size`. Pass `blocklist=set()` when constructing `RecommendationService` to disable the filter in an internal experiment.

### `GET /v1/users/{user_id}`

Training-period profile: number of items and top items by engagement. 404 for
unknown users. Generic keys are canonical; `n_apps`/`top_apps` remain for 2.0
compatibility.

```json
{
  "user_id": "1000123",
  "n_items": 42,
  "top_items": [{"item_id": "YouTube", "engagement": 9.8}],
  "n_apps": 42,
  "top_apps": [{"item_id": "YouTube", "engagement": 9.8}]
}
```

### `POST /v1/recommendations`

Batch scoring for up to 100 users.

```json
{"user_ids": ["1000123", "1000456"], "k": 5, "explain": true}
```

Response: `{"results": [...], "count": 2}`. Invalid users fall back per item; the batch does not fail as a whole.

### `POST /v1/feedback`

Append-only feedback capture. Returns 202. If `ENGAGEGRAPH_API_KEY` is set, send `X-API-Key`.

```json
{
  "impression_id": "imp-7f3a",
  "user_id": "1000123",
  "item_id": "Discord",
  "event": "click",
  "surface": "dashboard",
  "dwell_seconds": 12.5,
  "metadata": {"rank": 1}
}
```

Events: `impression`, `click`, `install`, `dwell`, `dismiss`, `view`,
`purchase`, `like`, `skip`. The `install` name is retained for app deployments;
other domains can use `purchase`, `view`, or `like`. Records are written to
`ENGAGEGRAPH_FEEDBACK_DIR/YYYY-MM-DD.jsonl` with a hashed server-side user
identifier.

### `GET /metrics`

Prometheus text format: request counters by route/status, mean latency, feedback count, uptime.

## 3. Error semantics

| Status | Meaning |
| --- | --- |
| 400 | Invalid value that passed schema validation (e.g., `k` out of range) |
| 401 | Missing/invalid feedback API key |
| 404 | Unknown user or app |
| 422 | Request validation error (FastAPI) |
| 503 | No artefact loaded / service not ready |
| 500 | Unexpected server error; check logs with the request ID |

## 4. Client examples

```bash
curl -s "http://127.0.0.1:8000/v1/users/1000123/recommendations?k=5" | jq
curl -s "http://127.0.0.1:8000/v1/apps/similar?item_id=Spotify&k=5" | jq
curl -s -X POST "http://127.0.0.1:8000/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{"impression_id":"imp-1","user_id":"1000123","item_id":"Discord","event":"click"}'
```

Python:

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000") as client:
    response = client.get("/v1/users/1000123/recommendations", params={"k": 5})
    response.raise_for_status()
    for item in response.json()["items"]:
        print(item["rank"], item["item_id"], item["score"])
```

## 5. Rate limits and authentication

The reference service does not rate-limit reads; apply limits at the proxy. Feedback writes can be protected with a shared API key. For multi-tenant or public deployments, add OIDC/JWT authentication at the gateway and map external principals to opaque panelist tokens.
