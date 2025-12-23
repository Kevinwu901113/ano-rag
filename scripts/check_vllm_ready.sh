#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/check_vllm_ready.sh [BASE_URL] [RETRIES] [SLEEP_SEC]
#
# BASE_URL can be:
#   - http://127.0.0.1:8000
#   - http://127.0.0.1:8000/v1
#   - http://127.0.0.1:8000/v1/models

BASE_URL="${1:-http://127.0.0.1:8000}"
RETRIES="${2:-90}"
SLEEP_SEC="${3:-2}"

base="${BASE_URL%/}"
if [[ "${base}" == */v1/models ]]; then
  url="${base}"
elif [[ "${base}" == */v1 ]]; then
  url="${base}/models"
else
  url="${base}/v1/models"
fi

i=0
while (( i < RETRIES )); do
  if curl --noproxy '*' -sSf -m 2 "${url}" >/dev/null; then
    echo "vLLM ready: ${url}"
    exit 0
  fi
  sleep "${SLEEP_SEC}"
  i=$((i + 1))
done

echo "vLLM not ready (timeout): ${url}" >&2
exit 1
