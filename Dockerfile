# Hosted solmem MCP server — zero pip dependencies (stdlib only).
FROM python:3.13-slim

# git is optional but lets the server push the store to a backup remote on `sync`.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY selfmod/ ./selfmod/
COPY solmem/ ./solmem/
COPY solmem_http.py ./

# Run unprivileged; the shared store lives on a mounted volume.
RUN useradd -m app && mkdir -p /data && chown app /data
USER app

ENV SOLMEM_HOME=/data \
    SOLMEM_PORT=8730 \
    SOLMEM_HTTP_HOST=0.0.0.0
EXPOSE 8730
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8730/health',timeout=2).status==200 else 1)"

CMD ["python", "solmem_http.py"]
