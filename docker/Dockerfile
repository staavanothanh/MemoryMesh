FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir '/tmp/*.whl[local]' && rm -rf /tmp/*.whl

RUN mkdir -p /app/db
ENV VEC_DB_PATH=/app/db/memory.db \
    SESSION_DB_PATH=/app/db/sessions.db \
    INSTINCT_DB_PATH=/app/db/instincts.db \
    LOG_LEVEL=INFO

VOLUME ["/app/db"]
EXPOSE 8090

ENTRYPOINT ["memorymesh", "serve"]
