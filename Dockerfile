# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENGAGEGRAPH_ARTIFACT=/app/artifacts/demo \
    ENGAGEGRAPH_FEEDBACK_DIR=/app/var/feedback

WORKDIR /app

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install . \
    && rm -rf /root/.cache /tmp/*

COPY configs ./configs
RUN mkdir -p /app/artifacts /app/var/feedback && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"

CMD ["engagegraph", "serve", "--host", "0.0.0.0", "--port", "8000"]
