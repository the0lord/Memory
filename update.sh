#!/usr/bin/env bash
# Update solmem on the server: pull latest code, rebuild, restart.
# Auto-detects whether you're running the plain or the HTTPS (Caddy) stack.
#   ./update.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "Pulling latest..."
git pull --ff-only

FILE="docker-compose.yml"
if docker ps --format '{{.Names}}' | grep -q -- '-caddy-'; then
  FILE="docker-compose.https.yml"
fi

echo "Rebuilding & restarting ($FILE)..."
docker compose -f "$FILE" up -d --build

echo "Waiting for health..."
cid=$(docker compose -f "$FILE" ps -q solmem)
for i in $(seq 1 20); do
  s=$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "")
  [ "$s" = "healthy" ] && { echo "solmem: healthy ($FILE)"; exit 0; }
  sleep 2
done
echo "warning: not healthy yet — docker compose -f $FILE logs --tail=30 solmem"
