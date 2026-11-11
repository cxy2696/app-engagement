# Deployment guide

EngageGraph is a single Python service plus a static dashboard. It can run on a laptop, in a container, or behind a reverse proxy. This guide covers the supported paths.

## 1. Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[als]"
engagegraph data build --out data/processed
engagegraph benchmark --config configs/benchmark.yaml
engagegraph serve --artifact artifacts/runs/<run>/artifact --port 8000
```

Open <http://127.0.0.1:8000>. The dashboard calls the same API it documents; there is no separate frontend build step.

## 2. Docker

```bash
docker build -t engagegraph:2.0.0 .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts:/app/artifacts:ro" \
  engagegraph:2.0.0
```

The image runs as a non-root user, exposes port 8000, and has a `HEALTHCHECK` against `/readyz`. Mount artefacts read-only; the service never writes to them.

Build the artefact before starting the container:

```bash
engagegraph train --config configs/als.yaml
docker run ... -e ENGAGEGRAPH_ARTIFACT=/app/artifacts/runs/<run>/artifact ...
```

`compose.yaml` provides the same setup with a bind mount and restart policy:

```bash
docker compose up --build
```

## 3. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENGAGEGRAPH_ARTIFACT` | registry / `artifacts/demo` | Path to the artefact directory |
| `ENGAGEGRAPH_FEEDBACK_DIR` | `var/feedback` | Append-only feedback JSONL directory |
| `ENGAGEGRAPH_API_KEY` | unset | If set, required in `X-API-Key` for `POST /v1/feedback` |
| `ENGAGEGRAPH_CORS_ORIGINS` | unset | Comma-separated allowed origins |
| `ENGAGEGRAPH_DEBUG` | unset | Re-raise CLI exceptions for debugging |
| `WEB_CONCURRENCY` | uvicorn default | Worker count (see scaling) |

The model is loaded once per worker at startup. There is no per-request disk I/O except feedback writes.

## 4. Reverse proxy

Terminate TLS at the proxy and forward `X-Request-ID` when present. Example Caddy:

```caddy
recommendations.example.com {
    encode gzip
    request_body {
        max_size 1MB
    }
    reverse_proxy 127.0.0.1:8000
}
```

Nginx equivalent:

```nginx
server {
    listen 443 ssl;
    server_name recommendations.example.com;
    client_max_body_size 1m;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-Request-ID $request_id;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

The API is read-only apart from feedback. If the service is public, add authentication at the proxy or in a gateway; do not expose raw panelist IDs publicly.

## 5. Scaling and availability

- **Vertical first.** Full-catalogue scoring is a few milliseconds for one user and ~200 ms for a batch of 512 on a laptop CPU. A single worker handles modest traffic.
- **Horizontal.** Run multiple uvicorn workers or replicas behind the proxy. The service is stateless; feedback writes go to a shared volume or should be replaced with a queue/warehouse sink.
- **Artefact rollout.** Publish a new artefact directory, then update `ENGAGEGRAPH_ARTIFACT` (or the registry `active` pointer) and restart workers. Keep the previous directory for instant rollback.
- **Warmup.** The first request after startup pays page-cache costs; send a `/v1/meta` probe in the readiness check.
- **Rate limits.** Apply them at the proxy. The built-in counters are for observability, not admission control.

## 6. Persistence and backup

- Artefacts are immutable and content-addressed; back them up with the release.
- `data/processed/` can be rebuilt from the raw exports; back up the raw files and the manifest.
- Feedback JSONL is the only mutable state. Mount it on durable storage, rotate daily, and ship it to the warehouse. Losing a day of feedback is preferable to blocking recommendations.

## 7. Security checklist

- [ ] TLS terminated; HTTP redirected.
- [ ] Authentication/authorisation in front of `/v1/*` if panelist IDs are sensitive.
- [ ] `ENGAGEGRAPH_API_KEY` set for feedback writes.
- [ ] Container runs as non-root, read-only root filesystem where possible.
- [ ] No raw panelist IDs in access logs; use request IDs and hashed IDs.
- [ ] Dependency scanning and image rebuilds on a schedule.
- [ ] Artefact provenance verified against the manifest before rollout.
- [ ] Feedback directory permissions restricted.

## 8. Health checks

| Endpoint | Use |
| --- | --- |
| `GET /healthz` | Liveness; returns 200 even when degraded, with `model_loaded` |
| `GET /readyz` | Readiness; 503 until an artefact is loaded |
| `GET /metrics` | Prometheus scrape |
| `GET /version` | Build/version probe |

Kubernetes probes should use `/healthz` for liveness and `/readyz` for readiness, with a 2-second timeout and 10-second period.
