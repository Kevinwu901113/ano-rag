#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-mirage}"
DATA_DIR="${DATA_DIR:-data/${DATASET}_sample}"
VLLM_MODEL="${VLLM_MODEL:-qwen2.5-7b-instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT0="${VLLM_PORT0:-8001}"
VLLM_PORT1="${VLLM_PORT1:-8002}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
RESULT_ROOT="${RESULT_ROOT:-result}"
SHARD_CNT=${SHARD_CNT:-2}
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"

NEW_RUN=0
WORK_DIR=""
LOG_DIR=""
NOTES_DIR=""
IDX_DIR=""
OUT_SHARD0=""
OUT_SHARD1=""
OUT_MERGED=""
IDX_TMP0=""
IDX_TMP1=""
VLLM_LOG0=""
VLLM_LOG1=""
VLLM_PID0=""
VLLM_PID1=""
BUILD_LOG0=""
BUILD_LOG1=""
BUILD_PID0=""
BUILD_PID1=""

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

ensure_workspace() {
  mkdir -p "$RESULT_ROOT"
  local selection=""
  local latest=""
  local max_index=0
  while IFS= read -r dir; do
    base="$(basename "$dir")"
    if [[ $base =~ ^([0-9]{3})-(.*)$ ]]; then
      local idx=${BASH_REMATCH[1]}
      local ds=${BASH_REMATCH[2]}
      (( idx > max_index )) && max_index=$idx
      if [[ $ds == "$DATASET" ]]; then
        latest="$dir"
      fi
    fi
  done < <(find "$RESULT_ROOT" -maxdepth 1 -mindepth 1 -type d | sort)

  if (( NEW_RUN )); then
    local next=$((max_index + 1))
    local id=$(printf "%03d" "$next")
    selection="$RESULT_ROOT/${id}-${DATASET}"
  else
    if [[ -n "$latest" ]]; then
      selection="$latest"
    else
      local next=$((max_index + 1))
      local id=$(printf "%03d" "$next")
      selection="$RESULT_ROOT/${id}-${DATASET}"
    fi
  fi

  WORK_DIR="$selection"
  mkdir -p "$WORK_DIR"

  LOG_DIR="$WORK_DIR/logs"
  mkdir -p "$LOG_DIR"

  NOTES_DIR="$WORK_DIR/notes"
  IDX_DIR="$WORK_DIR/indexes"
  mkdir -p "$NOTES_DIR" "$IDX_DIR"

  OUT_SHARD0="$NOTES_DIR/notes.${DATASET}.shard0.jsonl"
  OUT_SHARD1="$NOTES_DIR/notes.${DATASET}.shard1.jsonl"
  OUT_MERGED="$NOTES_DIR/notes.${DATASET}.jsonl"
  IDX_TMP0="$WORK_DIR/indexes.tmp0"
  IDX_TMP1="$WORK_DIR/indexes.tmp1"

  VLLM_LOG0="$LOG_DIR/vllm_gpu0.log"
  VLLM_LOG1="$LOG_DIR/vllm_gpu1.log"
  VLLM_PID0="$WORK_DIR/vllm_gpu0.pid"
  VLLM_PID1="$WORK_DIR/vllm_gpu1.pid"
  BUILD_LOG0="$LOG_DIR/build_shard0.log"
  BUILD_LOG1="$LOG_DIR/build_shard1.log"
  BUILD_PID0="$WORK_DIR/build_shard0.pid"
  BUILD_PID1="$WORK_DIR/build_shard1.pid"

  log "Workspace: $WORK_DIR"
  log "Dataset dir: $DATA_DIR"
  log "Notes dir: $NOTES_DIR"
  log "Indexes dir: $IDX_DIR"
}

start_vllm_dual() {
  log "Starting vLLM on GPU${GPU0}:${VLLM_PORT0} and GPU${GPU1}:${VLLM_PORT1}"

  extra_args=()
  if [[ -n "$VLLM_DOWNLOAD_DIR" ]]; then
    extra_args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
  fi

  CUDA_VISIBLE_DEVICES="${GPU0}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 --port "${VLLM_PORT0}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${extra_args[@]} \
    > "$VLLM_LOG0" 2>&1 & echo $! > "$VLLM_PID0"

  CUDA_VISIBLE_DEVICES="${GPU1}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL}" \
    --host 0.0.0.0 --port "${VLLM_PORT1}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${extra_args[@]} \
    > "$VLLM_LOG1" 2>&1 & echo $! > "$VLLM_PID1"

  log "Waiting for vLLM endpoints ready ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT0}/v1/models" 90 2 || { log "GPU0 endpoint not ready"; exit 1; }
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT1}/v1/models" 90 2 || { log "GPU1 endpoint not ready"; exit 1; }
  log "vLLM endpoints are healthy."
}

build_notes_dual() {
  if (( SHARD_CNT != 2 )); then
    log "Current script supports SHARD_CNT=2 (got ${SHARD_CNT})"
    exit 1
  fi

  rm -rf "$IDX_TMP0" "$IDX_TMP1"
  mkdir -p "$IDX_TMP0" "$IDX_TMP1"

  log "Launching shard0 -> ${OUT_SHARD0}"
  CUDA_VISIBLE_DEVICES="${GPU0}" python main_build_notes.py \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --out "${OUT_SHARD0}" \
    --indexes_dir "${IDX_TMP0}" \
    --vllm_endpoint "http://${VLLM_HOST}:${VLLM_PORT0}/v1" \
    --vllm_model "${VLLM_MODEL}" \
    --shard-idx 0 --shard-cnt "${SHARD_CNT}" \
    > "$BUILD_LOG0" 2>&1 & echo $! > "$BUILD_PID0"

  log "Launching shard1 -> ${OUT_SHARD1}"
  CUDA_VISIBLE_DEVICES="${GPU1}" python main_build_notes.py \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --out "${OUT_SHARD1}" \
    --indexes_dir "${IDX_TMP1}" \
    --vllm_endpoint "http://${VLLM_HOST}:${VLLM_PORT1}/v1" \
    --vllm_model "${VLLM_MODEL}" \
    --shard-idx 1 --shard-cnt "${SHARD_CNT}" \
    > "$BUILD_LOG1" 2>&1 & echo $! > "$BUILD_PID1"

  log "Waiting both shards to finish ..."
  wait "$(cat "$BUILD_PID0")" || { log "shard0 failed"; exit 1; }
  wait "$(cat "$BUILD_PID1")" || { log "shard1 failed"; exit 1; }
  rm -f "$BUILD_PID0" "$BUILD_PID1"

  log "Merging shards -> ${OUT_MERGED}"
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

  log "Building final indexes ..."
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

  rm -rf "$IDX_TMP0" "$IDX_TMP1"
}

stop_vllm_and_wait() {
  log "Stopping vLLM instances ..."
  kill_and_wait "$VLLM_PID0"
  kill_and_wait "$VLLM_PID1"
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT0}" 90 2 || { log "port ${VLLM_PORT0} still busy"; exit 1; }
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT1}" 90 2 || { log "port ${VLLM_PORT1} still busy"; exit 1; }
  log "vLLM ports closed."
}

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--new]

Environment overrides: DATA_DIR, DATASET, RESULT_ROOT, VLLM_MODEL, ...
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --new)
      NEW_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

ensure_workspace
start_vllm_dual
build_notes_dual
stop_vllm_and_wait

log "Notes written to ${OUT_MERGED}"
log "Indexes stored at ${IDX_DIR}"
