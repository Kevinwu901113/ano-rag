#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-mirage}"
DATA_DIR="${DATA_DIR:-data/mirage}"
VLLM_MODEL="${VLLM_MODEL:-qwen2.5-7b-instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT0="${VLLM_PORT0:-8001}"
VLLM_PORT1="${VLLM_PORT1:-8002}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

NOTES_DIR="${NOTES_DIR:-notes}"
IDX_DIR="${IDX_DIR:-indexes}"
OUT_SHARD0="${OUT_SHARD0:-${NOTES_DIR}/notes.mirage.shard0.jsonl}"
OUT_SHARD1="${OUT_SHARD1:-${NOTES_DIR}/notes.mirage.shard1.jsonl}"
OUT_MERGED="${OUT_MERGED:-${NOTES_DIR}/notes.jsonl}"

LMSTUDIO_ENDPOINT="${LMSTUDIO_ENDPOINT:-http://127.0.0.1:1234/v1}"
LMSTUDIO_MODEL="${LMSTUDIO_MODEL:-openai/gpt-oss-20b}"
QUESTION="${QUESTION:-Who is the spouse of the Green performer?}"
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"

log() { printf "\033[1;34m[%s]\033[0m %s\n" "$(date +'%H:%M:%S')" "$*"; }

wait_http_ok() {
  local url="$1" retries="${2:-60}" sleep_s="${3:-2}"
  local i=0
  while (( i < retries )); do
    if curl -sSf -m 2 "$url" >/dev/null; then return 0; fi
    sleep "$sleep_s"; i=$((i+1))
  done
  return 1
}

wait_port_closed() {
  local host="$1" port="$2" retries="${3:-60}" sleep_s="${4:-2}"
  local i=0
  while (( i < retries )); do
    if ! nc -z "$host" "$port" 2>/dev/null; then return 0; fi
    sleep "$sleep_s"; i=$((i+1))
  done
  return 1
}

kill_and_wait() {
  local pidfile="$1"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" || true)"
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

start_vllm_dual() {
  mkdir -p "$NOTES_DIR" "$IDX_DIR"
  log "Starting vLLM on GPU${GPU0}:${VLLM_PORT0} and GPU${GPU1}:${VLLM_PORT1}"

  CUDA_VISIBLE_DEVICES="${GPU0}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 --port "${VLLM_PORT0}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    > vllm_gpu0.log 2>&1 & echo $! > vllm_gpu0.pid

  CUDA_VISIBLE_DEVICES="${GPU1}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 --port "${VLLM_PORT1}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    > vllm_gpu1.log 2>&1 & echo $! > vllm_gpu1.pid

  log "Waiting for vLLM endpoints ready ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT0}/v1/models" 90 2 || { log "GPU0 endpoint not ready"; exit 1; }
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT1}/v1/models" 90 2 || { log "GPU1 endpoint not ready"; exit 1; }
  log "vLLM endpoints are healthy."
}

build_notes_dual() {
  log "Launching shard0 -> ${OUT_SHARD0}"
  CUDA_VISIBLE_DEVICES="${GPU0}" python main_build_notes.py \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --out "${OUT_SHARD0}" \
    --indexes_dir "${IDX_DIR}.tmp0" \
    --vllm_endpoint "http://${VLLM_HOST}:${VLLM_PORT0}/v1" \
    --vllm_model "${VLLM_MODEL}" \
    --shard-idx 0 --shard-cnt 2 \
    > build_shard0.log 2>&1 & echo $! > build_shard0.pid

  log "Launching shard1 -> ${OUT_SHARD1}"
  CUDA_VISIBLE_DEVICES="${GPU1}" python main_build_notes.py \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --out "${OUT_SHARD1}" \
    --indexes_dir "${IDX_DIR}.tmp1" \
    --vllm_endpoint "http://${VLLM_HOST}:${VLLM_PORT1}/v1" \
    --vllm_model "${VLLM_MODEL}" \
    --shard-idx 1 --shard-cnt 2 \
    > build_shard1.log 2>&1 & echo $! > build_shard1.pid

  log "Waiting both shards to finish ..."
  wait $(cat build_shard0.pid) || { log "shard0 failed"; exit 1; }
  wait $(cat build_shard1.pid) || { log "shard1 failed"; exit 1; }
  rm -f build_shard0.pid build_shard1.pid

  log "Merging shard notes -> ${OUT_MERGED}"
  OUT_SHARD0="${OUT_SHARD0}" OUT_SHARD1="${OUT_SHARD1}" OUT_MERGED="${OUT_MERGED}" python - <<'PY'
import json, os
paths = [os.getenv("OUT_SHARD0"), os.getenv("OUT_SHARD1")]
merged = os.getenv("OUT_MERGED")
seen = set()
with open(merged, "w", encoding="utf-8") as fout:
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                note = json.loads(line)
                nid = note["note_id"]
                if nid in seen:
                    continue
                seen.add(nid)
                fout.write(json.dumps(note, ensure_ascii=False) + "\n")
print(f"merged {len(seen)} notes to {merged}")
PY

  log "Building final indexes from merged notes ..."
  OUT_MERGED="${OUT_MERGED}" IDX_DIR="${IDX_DIR}" python - <<'PY'
import os
from indexer.index_builder import IndexBuilder
notes = os.getenv("OUT_MERGED")
idx_dir = os.getenv("IDX_DIR")
builder = IndexBuilder()
builder.build_from_jsonl(notes)
builder.dump(idx_dir)
print("indexes built into", idx_dir)
PY
}

stop_vllm_and_wait() {
  log "Stopping vLLM instances ..."
  kill_and_wait "vllm_gpu0.pid"
  kill_and_wait "vllm_gpu1.pid"

  log "Waiting ports to close ..."
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT0}" 90 2 || { log "port ${VLLM_PORT0} still busy"; exit 1; }
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT1}" 90 2 || { log "port ${VLLM_PORT1} still busy"; exit 1; }
  log "vLLM ports closed."
}

run_query() {
  local q="${1:-$QUESTION}"
  log "Query: ${q}"
  python main_query.py \
    --question "${q}" \
    --indexes_dir "${IDX_DIR}" \
    --notes "${OUT_MERGED}" \
    --lmstudio_endpoint "${LMSTUDIO_ENDPOINT}" \
    --lmstudio_model "${LMSTUDIO_MODEL}"
}

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") all
  $(basename "$0") extract
  $(basename "$0") query "<question>"
USAGE
}

cmd="${1:-all}"
case "$cmd" in
  all)
    start_vllm_dual
    build_notes_dual
    stop_vllm_and_wait
    run_query "$QUESTION"
    ;;
  extract)
    start_vllm_dual
    build_notes_dual
    stop_vllm_and_wait
    ;;
  query)
    shift || true
    run_query "${1:-$QUESTION}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
