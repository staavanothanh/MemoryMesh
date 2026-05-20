FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

RUN mkdir -p /app/db/chroma /app/db
ENV CHROMA_DB_PATH=/app/db/chroma \
    FTS_DB_PATH=/app/db/memory_fts.db \
    SESSION_DB_PATH=/app/db/sessions.db \
    LOG_LEVEL=INFO

VOLUME ["/app/db"]
EXPOSE 8090

ENTRYPOINT ["memorymesh"]
