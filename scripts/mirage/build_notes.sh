#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-mirage}"
DATA_DIR="${DATA_DIR:-data/${DATASET}_sample}"
VLLM_MODEL="${VLLM_MODEL:-qwen2.5-7b-instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
RESULT_ROOT="${RESULT_ROOT:-result_relrag}"
WORKDIR="${WORKDIR:-}"
RUN_ID="${RUN_ID:-}"
USE_GUIDED_JSON="${USE_GUIDED_JSON:-1}"
JSON_SCHEMA_NAME="${JSON_SCHEMA_NAME:-ano-note}"
VLLM_GUIDED_BACKEND="${VLLM_GUIDED_BACKEND:-xgrammar}"
# Single-process build by default
SHARD_CNT=${SHARD_CNT:-1}
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-${HOME:-/root}/.cache/huggingface}"
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.7}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
STARTED_VLLM=0

VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-}"
if [[ -z "${VLLM_CUDA_DEVICES}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    VLLM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
  elif [[ -n "${GPU1}" ]]; then
    VLLM_CUDA_DEVICES="${GPU0},${GPU1}"
  else
    VLLM_CUDA_DEVICES="${GPU0}"
  fi
fi

NEW_RUN=0
WORK_DIR=""
LOG_DIR=""
NOTES_DIR=""
IDX_DIR=""
OUT_MERGED=""
VLLM_LOG=""
VLLM_PID=""
BUILD_LOG=""
BUILD_PID=""
PROG_SINGLE=""
RUN_CONFIG=""

log() { printf "\033[1;34m[%s]\033[0m %s\n" "$(date +'%H:%M:%S')" "$*"; }

prepare_run_config() {
  local base_cfg="$ROOT_DIR/config.yaml"
  local faiss_dir="$IDX_DIR/faiss"
  local bm25_store="$IDX_DIR/bm25/notes"
  mkdir -p "$faiss_dir" "$bm25_store"
  RUN_CONFIG="$ARTIFACTS_DIR/config.override.yaml"
  python - "$base_cfg" "$RUN_CONFIG" "$OUT_MERGED" "$IDX_DIR" "$faiss_dir" "$bm25_store" <<'PY'
import os, sys, yaml
base_cfg, out_cfg, notes_path, idx_dir, faiss_dir, bm25_store = sys.argv[1:7]
cfg = {}
if os.path.exists(base_cfg):
    with open(base_cfg, 'r', encoding='utf-8') as handle:
        cfg = yaml.safe_load(handle) or {}
notes_cfg = cfg.setdefault('notes', {})
notes_cfg['out_path'] = notes_path
notes_cfg['indexes_dir'] = idx_dir
retr = cfg.setdefault('retriever', {})
emb = retr.setdefault('embedding', {})
emb['offline_index_path'] = os.path.join(faiss_dir, 'notes.faiss')
emb['meta_path'] = os.path.join(faiss_dir, 'notes.meta.parquet')
bm25 = retr.setdefault('bm25', {})
bm25['store_path'] = bm25_store
# Guided JSON config for generator/parsing
use_guided = os.environ.get("USE_GUIDED_JSON", "0") not in {"0", "false", "False", ""}
schema_name = os.environ.get("JSON_SCHEMA_NAME", "ano-note")
json_mode = cfg.setdefault('vllm', {}).setdefault('json_mode', {})
json_mode.setdefault('schema_name', schema_name)
if use_guided:
    json_mode['use_guided_json'] = True
    json_mode.setdefault('use_response_format', False)
    cfg.setdefault('parsing', {}).setdefault('assume_valid_json', True)
with open(out_cfg, 'w', encoding='utf-8') as handle:
    yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
PY
  export ANO_RAG_CONFIG="$RUN_CONFIG"
  log "Using run config: $ANO_RAG_CONFIG"
}

auto_build_indexes_if_needed() {
  local flag
  flag=$(python - <<'PY'
from config import config as loader
cfg = loader.load_config()
emb = (cfg.get("retriever") or {}).get("embedding") or {}
print("1" if emb.get("auto_build") else "0")
PY
  )
  if [[ "$flag" == "1" ]]; then
    log "retriever.embedding.auto_build=true -> running scripts/build_indexes.sh"
    bash "$ROOT_DIR/scripts/build_indexes.sh"
  fi
}

# Read progress JSON and output: total processed notes
progress_values() {
  local path="$1"
  python - "$path" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p, 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f"{d.get('total_chunks', 0)} {d.get('processed_chunks', 0)} {d.get('notes_written', 0)} {d.get('current_workers', 0)}")
except Exception:
    print("0 0 0 0")
PY
}

wait_http_ok() {
  local url="$1" retries="${2:-60}" sleep_s="${3:-2}"
  local i=0
  while (( i < retries )); do
    if curl --noproxy '*' -sSf -m 2 "$url" >/dev/null; then return 0; fi
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

resolve_model_path() {
  local candidate="$1"
  if [[ ! -d "$candidate" ]]; then
    echo "$candidate"
    return
  fi

  if [[ -f "$candidate/config.json" ]]; then
    echo "$candidate"
    return
  fi

  local config_path
  config_path=$(find "$candidate" -maxdepth 4 -type f -name "config.json" | head -n 1)
  if [[ -n "$config_path" ]]; then
    echo "$(dirname "$config_path")"
  else
    echo "$candidate"
  fi
}

ensure_workspace() {
  local run_id="${RUN_ID}"
  if [[ -z "$run_id" ]]; then
    run_id="$(date +%Y%m%d_%H%M%S)"
  fi

  if [[ -n "$WORKDIR" ]]; then
    WORK_DIR="$WORKDIR"
  else
    WORK_DIR="${RESULT_ROOT}/run_${run_id}/${DATASET}"
  fi
  mkdir -p "$WORK_DIR"

  ARTIFACTS_DIR="$WORK_DIR/artifacts"
  LOG_DIR="$ARTIFACTS_DIR/logs"
  mkdir -p "$LOG_DIR"

  NOTES_DIR="$ARTIFACTS_DIR/notes"
  IDX_DIR="$ARTIFACTS_DIR/indexes"
  mkdir -p "$NOTES_DIR" "$IDX_DIR"

  OUT_MERGED="$NOTES_DIR/notes.${DATASET}.jsonl"

  VLLM_LOG="$LOG_DIR/vllm.log"
  VLLM_PID="$ARTIFACTS_DIR/vllm.pid"
  BUILD_LOG="$LOG_DIR/build_single.log"
  BUILD_PID="$ARTIFACTS_DIR/build_single.pid"
  PROG_SINGLE="$ARTIFACTS_DIR/progress.json"

  log "Workspace: $WORK_DIR"
  log "Dataset dir: $DATA_DIR"
  log "Notes dir: $NOTES_DIR"
  log "Indexes dir: $IDX_DIR"
}

start_vllm_tp() {
  local cuda_devices="${VLLM_CUDA_DEVICES//[[:space:]]/}"
  if [[ -z "${cuda_devices}" ]]; then
    cuda_devices="${GPU0},${GPU1}"
  fi

  local tp_size=0
  IFS=',' read -r -a _devs <<< "${cuda_devices}"
  for d in "${_devs[@]}"; do
    [[ -n "${d}" ]] && tp_size=$((tp_size + 1))
  done
  (( tp_size < 1 )) && tp_size=1

  log "Starting vLLM (TP=${tp_size}) on CUDA_VISIBLE_DEVICES=${cuda_devices} -> ${VLLM_HOST}:${VLLM_PORT}"

  VLLM_MODEL_RESOLVED=$(resolve_model_path "$VLLM_MODEL")
  if [[ "$VLLM_MODEL_RESOLVED" != "$VLLM_MODEL" ]]; then
    log "Resolved model path: $VLLM_MODEL_RESOLVED"
  fi

  extra_args=()
  if [[ -n "${VLLM_DOWNLOAD_DIR}" ]]; then
    extra_args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
  fi
  if [[ -n "$VLLM_GPU_MEMORY_UTIL" ]]; then
    extra_args+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTIL")
  fi
  if [[ "$USE_GUIDED_JSON" != "0" && -n "$VLLM_GUIDED_BACKEND" ]]; then
    extra_args+=(--guided-decoding-backend "$VLLM_GUIDED_BACKEND")
  fi

  if nc -z "${VLLM_HOST}" "${VLLM_PORT}" 2>/dev/null; then
    log "Port ${VLLM_PORT} already in use; aborting."
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="${cuda_devices}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL_RESOLVED}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --tensor-parallel-size "${tp_size}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --uvicorn-log-level info \
    "${extra_args[@]}" ${VLLM_EXTRA_ARGS} \
    > "$VLLM_LOG" 2>&1 & echo $! > "$VLLM_PID"

  log "Waiting for vLLM endpoints ready ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" 90 2 || { log "vLLM endpoint not ready"; exit 1; }
  log "vLLM endpoint is healthy."
  STARTED_VLLM=1
}

build_notes_single() {
  if (( SHARD_CNT != 1 )); then
    log "Single-process mode: please set SHARD_CNT=1 (got ${SHARD_CNT})"
    exit 1
  fi

  # Ensure NoteGenerator uses the single endpoint passed via CLI (no multi-endpoint probing)
  for var in ${!VLLM_ENDPOINT@}; do
    unset "$var"
  done

  log "Launching single builder -> ${OUT_MERGED}"
  CUDA_VISIBLE_DEVICES="${VLLM_CUDA_DEVICES}" python "$ROOT_DIR/main_build_notes.py" \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --out "${OUT_MERGED}" \
    --indexes_dir "${IDX_DIR}" \
    --vllm_endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --vllm_model "${VLLM_MODEL}" \
    --shard-cnt 1 \
    --progress-path "${PROG_SINGLE}" \
    > "$BUILD_LOG" 2>&1 & echo $! > "$BUILD_PID"

  # Progress monitor loop
  log "Waiting build to finish ..."
  start_time=$(date +%s)
  pid=$(cat "$BUILD_PID")
  while kill -0 "$pid" 2>/dev/null; do
    total=0; done=0; notes=0; workers=0
    if [[ -f "$PROG_SINGLE" ]]; then
      read -r total done notes workers < <(progress_values "$PROG_SINGLE")
    fi
    now=$(date +%s)
    elapsed=$(( now - start_time ))
    rate_str="0.00"
    if (( elapsed > 0 )); then
      rate_str=$(awk -v d="$done" -v e="$elapsed" 'BEGIN{printf "%.2f", d/e}')
    fi
    remaining=$(( total > done ? total - done : 0 ))
    eta="N/A"
    if [[ "$rate_str" != "0.00" && $remaining -gt 0 ]]; then
      eta_sec=$(awk -v r="$rate_str" -v rem="$remaining" 'BEGIN{printf "%d", rem/r}')
      eta="${eta_sec}s"
    fi
    width=40
    filled=0
    if (( total > 0 )); then
      filled=$(( done * width / total ))
    fi
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    empty=$(printf '%*s' $(( width - filled )) '' | tr ' ' '-')
    printf "\rProgress [%s%s] %d/%d chunks | notes=%d | workers=%d | elapsed=%ds | eta=%s | rate=%s chunk/s" "$bar" "$empty" "$done" "$total" "$notes" "$workers" "$elapsed" "$eta" "$rate_str"
    sleep 2
  done
  echo
  wait "$pid" || { log "build failed"; exit 1; }
  rm -f "$BUILD_PID"

  log "Notes written to ${OUT_MERGED}"
  log "Indexes stored at ${IDX_DIR}"
}

stop_vllm_and_wait() {
  log "Stopping vLLM instances ..."
  kill_and_wait "$VLLM_PID"
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT}" 90 2 || { log "port ${VLLM_PORT} still busy"; exit 1; }
  log "vLLM port closed."
  STARTED_VLLM=0
}

cleanup() {
  if [[ $STARTED_VLLM -eq 1 ]]; then
    stop_vllm_and_wait || true
  fi
}

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--new] [--workdir <path>]

Environment overrides: DATA_DIR, DATASET, RESULT_ROOT, WORKDIR, RUN_ID, VLLM_MODEL, ...
USAGE
}

trap cleanup EXIT

SKIP_VLLM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --new)
      NEW_RUN=1
      shift
      ;;
    --skip-vllm)
      SKIP_VLLM=1
      shift
      ;;
    --workdir|--work-dir)
      WORKDIR="${2:-}"
      shift 2
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
prepare_run_config
start_vllm_tp
build_notes_single
stop_vllm_and_wait
auto_build_indexes_if_needed

log "Notes written to ${OUT_MERGED}"
log "Indexes stored at ${IDX_DIR}"
