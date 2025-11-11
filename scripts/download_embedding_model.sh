#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: scripts/download_embedding_model.sh [--model <repo_id>] [--cache-dir <path>] [--local-dir <path>] [--revision <rev>]

Defaults are loaded from config.retriever.embedding (model, cache_dir, download_dir/model_path_override).
Set EMB_CACHE_DIR/EMB_MODEL_PATH/EMB_DOWNLOAD_DIR or config.yaml to override permanently.
EOF
}

MODEL_OVERRIDE=""
CACHE_OVERRIDE=""
LOCAL_OVERRIDE=""
REVISION="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_OVERRIDE="$2"
      shift 2
      ;;
    --local-dir|--download-dir)
      LOCAL_OVERRIDE="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found; pip install huggingface_hub>=0.23" >&2
  exit 1
fi

readarray -t CFG_VALUES < <(python - <<'PY'
from config import config as loader
cfg = loader.load_config()
emb = (cfg.get("retriever") or {}).get("embedding") or {}
def _fmt(value):
    return value if isinstance(value, str) and value.strip() else ""
print(_fmt(emb.get("model")))
print(_fmt(emb.get("cache_dir")))
print(_fmt(emb.get("download_dir")))
print(_fmt(emb.get("model_path_override")))
PY
)

CFG_MODEL="${CFG_VALUES[0]:-}"
CFG_CACHE="${CFG_VALUES[1]:-}"
CFG_DOWNLOAD="${CFG_VALUES[2]:-}"
CFG_OVERRIDE_PATH="${CFG_VALUES[3]:-}"

MODEL_ID="${MODEL_OVERRIDE:-$CFG_MODEL}"
CACHE_DIR="${CACHE_OVERRIDE:-$CFG_CACHE}"
LOCAL_DIR="${LOCAL_OVERRIDE:-${CFG_DOWNLOAD:-}}"
if [[ -z "$LOCAL_DIR" && -n "$CFG_OVERRIDE_PATH" ]]; then
  LOCAL_DIR="$CFG_OVERRIDE_PATH"
fi

if [[ -n "$CACHE_DIR" && "$CACHE_DIR" == ~* ]]; then
  CACHE_DIR="${CACHE_DIR/#\~/$HOME}"
fi
if [[ -n "$LOCAL_DIR" && "$LOCAL_DIR" == ~* ]]; then
  LOCAL_DIR="${LOCAL_DIR/#\~/$HOME}"
fi

if [[ -z "$MODEL_ID" ]]; then
  echo "Embedding model id is empty; set via --model or config.retriever.embedding.model" >&2
  exit 1
fi

HF_ARGS=(download "$MODEL_ID" --revision "$REVISION" --resume-download)

if [[ -n "$CACHE_DIR" ]]; then
  mkdir -p "$CACHE_DIR"
  HF_ARGS+=(--cache-dir "$CACHE_DIR")
fi

if [[ -n "$LOCAL_DIR" ]]; then
  mkdir -p "$LOCAL_DIR"
  HF_ARGS+=(--local-dir "$LOCAL_DIR" --local-dir-use-symlinks False)
fi

echo "[download_embedding_model] huggingface-cli ${HF_ARGS[*]}"
huggingface-cli "${HF_ARGS[@]}"

echo
echo "Model downloaded. Update config.retriever.embedding.model_path_override or cache_dir if needed:"
echo "  model_path_override: ${LOCAL_DIR:-<keep remote repo>}"
echo "  cache_dir: ${CACHE_DIR:-<default hf cache>}"
