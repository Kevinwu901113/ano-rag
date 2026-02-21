#!/usr/bin/env bash
set -euo pipefail

EVAL_SCRIPT="scripts/eval_retrieval_v3.py"
LOG_FILE="retrieval_eval.log"
EXPECTED_COUNT=500

DATASETS=("hotpotqa" "musique" "2wiki")
METHODS=("bm25" "dense" "lightrag" "raptor" "graphrag")
BACKENDS=("qwen")

WATCH=0
INTERVAL=60
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)
            WATCH=1
            shift
            ;;
        --interval)
            INTERVAL="${2:-60}"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

log() {
    local msg="$1"
    echo "[$(timestamp)] $msg" | tee -a "$LOG_FILE"
}

gold_file_for_dataset() {
    local dataset="$1"
    case "$dataset" in
        hotpotqa) echo "baseline/data/hotpotqa/qa.jsonl" ;; # Distractor500 baseline
        musique) echo "baseline/data/musique/qa.jsonl" ;;
        2wiki) echo "baseline/data/2wiki/qa.jsonl" ;;
        *)
            echo "Unknown dataset: $dataset" >&2
            return 1
            ;;
    esac
}

resolve_pred_file() {
    local method="$1"
    local dataset="$2"
    local backend="$3"
    local candidates=(
        "baseline/results/${method}/${dataset}/${backend}/pred_retrieval.jsonl"
        "baseline/results/${method}/${dataset}/${backend}/pred.jsonl"
        "baseline/results_retrieval_fix/${method}/${dataset}/${backend}/pred_retrieval.jsonl"
        "baseline/results_retrieval_fix/${method}/${dataset}/${backend}/pred.jsonl"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

evaluate_once() {
    local pending=0
    local method dataset backend pred_file line_count out_dir metrics_file gold_file

    for method in "${METHODS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            for backend in "${BACKENDS[@]}"; do
                if ! pred_file="$(resolve_pred_file "$method" "$dataset" "$backend")"; then
                    log "WAIT $method/$dataset/$backend: prediction file not found"
                    pending=$((pending + 1))
                    continue
                fi

                line_count="$(wc -l < "$pred_file")"
                if [[ "$line_count" -lt "$EXPECTED_COUNT" ]]; then
                    log "WAIT $method/$dataset/$backend: $line_count/$EXPECTED_COUNT lines"
                    pending=$((pending + 1))
                    continue
                fi

                out_dir="$(dirname "$pred_file")"
                metrics_file="${out_dir}/metrics_retrieval.json"
                gold_file="$(gold_file_for_dataset "$dataset")"

                if [[ "$FORCE" -eq 0 && -f "$metrics_file" ]]; then
                    log "SKIP $method/$dataset/$backend: metrics already exist"
                    continue
                fi

                log "EVAL $method/$dataset/$backend: $pred_file"
                python3 "$EVAL_SCRIPT" \
                    --pred_file "$pred_file" \
                    --gold_file "$gold_file" \
                    --expected_count "$EXPECTED_COUNT" \
                    > "$metrics_file"
                log "DONE $method/$dataset/$backend -> $metrics_file"
            done
        done
    done

    return "$pending"
}

log "Starting retrieval evaluation (v3). watch=$WATCH interval=${INTERVAL}s force=$FORCE"
while true; do
    if evaluate_once; then
        log "All evaluations finished."
        break
    else
        pending_count="$?"
    fi
    if [[ "$WATCH" -eq 0 ]]; then
        log "Pending evaluations: $pending_count"
        break
    fi
    log "Pending evaluations: $pending_count, sleeping ${INTERVAL}s"
    sleep "$INTERVAL"
done
