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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
RESULT_ROOT="${RESULT_ROOT:-result}"
# Single-process build by default
SHARD_CNT=${SHARD_CNT:-1}
VLLM_BIN="${VLLM_BIN:-python -m vllm.entrypoints.openai.api_server}"
VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-}"
STARTED_VLLM=0

NEW_RUN=0
WORK_DIR=""
LOG_DIR=""
NOTES_DIR=""
IDX_DIR=""
OUT_MERGED=""
VLLM_LOG0=""
VLLM_LOG1=""
VLLM_PID0=""
VLLM_PID1=""
BUILD_LOG=""
BUILD_PID=""
PROG_SINGLE=""

log() { printf "\033[1;34m[%s]\033[0m %s\n" "$(date +'%H:%M:%S')" "$*"; }

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
  mkdir -p "$RESULT_ROOT"
  local selection=""
  local latest=""
  local max_index=0
  while IFS= read -r dir; do
    base="$(basename "$dir")"
    if [[ $base =~ ^([0-9]{3})-(.*)$ ]]; then
      local idx_raw=${BASH_REMATCH[1]}
      local ds=${BASH_REMATCH[2]}
      local idx=$((10#$idx_raw))
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

  OUT_MERGED="$NOTES_DIR/notes.${DATASET}.jsonl"

  VLLM_LOG0="$LOG_DIR/vllm_gpu0.log"
  VLLM_LOG1="$LOG_DIR/vllm_gpu1.log"
  VLLM_PID0="$WORK_DIR/vllm_gpu0.pid"
  VLLM_PID1="$WORK_DIR/vllm_gpu1.pid"
  BUILD_LOG="$LOG_DIR/build_single.log"
  BUILD_PID="$WORK_DIR/build_single.pid"
  PROG_SINGLE="$WORK_DIR/progress.json"

  log "Workspace: $WORK_DIR"
  log "Dataset dir: $DATA_DIR"
  log "Notes dir: $NOTES_DIR"
  log "Indexes dir: $IDX_DIR"
}

start_vllm_dual() {
  log "Starting vLLM on GPU${GPU0}:${VLLM_PORT0} and GPU${GPU1}:${VLLM_PORT1}"

  VLLM_MODEL_RESOLVED=$(resolve_model_path "$VLLM_MODEL")
  if [[ "$VLLM_MODEL_RESOLVED" != "$VLLM_MODEL" ]]; then
    log "Resolved model path: $VLLM_MODEL_RESOLVED"
  fi

  extra_args=()
  if [[ -n "$VLLM_DOWNLOAD_DIR" ]]; then
    extra_args+=(--download-dir "$VLLM_DOWNLOAD_DIR")
  fi

  CUDA_VISIBLE_DEVICES="${GPU0}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL_RESOLVED}" \
    --host 0.0.0.0 --port "${VLLM_PORT0}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${extra_args[@]} \
    > "$VLLM_LOG0" 2>&1 & echo $! > "$VLLM_PID0"

  CUDA_VISIBLE_DEVICES="${GPU1}" nohup ${VLLM_BIN} \
    --model "${VLLM_MODEL_RESOLVED}" \
    --host 0.0.0.0 --port "${VLLM_PORT1}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    ${extra_args[@]} \
    > "$VLLM_LOG1" 2>&1 & echo $! > "$VLLM_PID1"

  log "Waiting for vLLM endpoints ready ..."
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT0}/v1/models" 90 2 || { log "GPU0 endpoint not ready"; exit 1; }
  wait_http_ok "http://${VLLM_HOST}:${VLLM_PORT1}/v1/models" 90 2 || { log "GPU1 endpoint not ready"; exit 1; }
  log "vLLM endpoints are healthy."
  STARTED_VLLM=1
}

build_notes_single() {
  if (( SHARD_CNT != 1 )); then
    log "Single-process mode: please set SHARD_CNT=1 (got ${SHARD_CNT})"
    exit 1
  fi

  # Export two endpoints to be picked up by NoteGenerator (env priority)
  export VLLM_ENDPOINT0="http://${VLLM_HOST}:${VLLM_PORT0}/v1"
  export VLLM_ENDPOINT1="http://${VLLM_HOST}:${VLLM_PORT1}/v1"

  log "Launching single builder -> ${OUT_MERGED}"
  CUDA_VISIBLE_DEVICES="${GPU0},${GPU1}" python main_build_notes.py \
    --dataset "${DATASET}" \
    --data_dir "${DATA_DIR}" \
    --out "${OUT_MERGED}" \
    --indexes_dir "${IDX_DIR}" \
    --vllm_endpoint "http://${VLLM_HOST}:${VLLM_PORT0}/v1" \
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
  kill_and_wait "$VLLM_PID0"
  kill_and_wait "$VLLM_PID1"
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT0}" 90 2 || { log "port ${VLLM_PORT0} still busy"; exit 1; }
  wait_port_closed "${VLLM_HOST}" "${VLLM_PORT1}" 90 2 || { log "port ${VLLM_PORT1} still busy"; exit 1; }
  log "vLLM ports closed."
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
  $(basename "$0") [--new]

Environment overrides: DATA_DIR, DATASET, RESULT_ROOT, VLLM_MODEL, ...
USAGE
}

trap cleanup EXIT

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
build_notes_single
stop_vllm_and_wait

log "Notes written to ${OUT_MERGED}"
log "Indexes stored at ${IDX_DIR}"
