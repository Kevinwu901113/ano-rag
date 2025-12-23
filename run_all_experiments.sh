#!/bin/bash
set -e

# Configuration
DATA_DIR="data/mirage_sample_200"
DOC_POOL="${DATA_DIR}/doc_pool.json"
DATASET="${DATA_DIR}/dataset.json"
LM_ENDPOINT="http://127.0.0.1:8000/v1"
LM_MODEL="qwen3-30b-a3b"
RESULT_ROOT="result"

# Ensure directories exist
mkdir -p "$RESULT_ROOT"

echo "Starting Experiments..."

# 1. Direct LLM
echo "Running Direct LLM..."
python scripts/mirage/run_direct_llm.py \
  --dataset-path "$DATASET" \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_direct_200"

# 2. Naive RAG
echo "Running Naive RAG..."
# Build Index
python scripts/mirage/build_naive_index.py \
  --doc-pool "$DOC_POOL" \
  --out-dir "${RESULT_ROOT}/mirage_naive_index_200"

# Run QA
python scripts/mirage/run_naive_rag.py \
  --dataset-path "$DATASET" \
  --index-dir "${RESULT_ROOT}/mirage_naive_index_200" \
  --topk 5 \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_naive_rag_200"

# 3. Vanilla RAG (Standard)
echo "Running Vanilla RAG..."
python scripts/mirage/run_vanilla_rag.py \
  --dataset-path "$DATASET" \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_vanilla_rag_200"

# 4. FiD RAG
echo "Running FiD RAG..."
# Build Index (uses naive index usually, but script has build_fid_index)
python scripts/mirage/build_fid_index.py \
  --doc-pool "$DOC_POOL" \
  --out-dir "${RESULT_ROOT}/mirage_fid_index_200"

# Run QA
python scripts/mirage/run_fid_rag.py \
  --dataset-path "$DATASET" \
  --index-dir "${RESULT_ROOT}/mirage_fid_index_200" \
  --topk 5 \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_fid_rag_200"

# 5. Simple Self-RAG
echo "Running Simple Self-RAG..."
python scripts/mirage/run_simple_selfrag.py \
  --dataset-path "$DATASET" \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_simple_selfrag_200"

# 6. Simple Raptor
echo "Running Simple Raptor..."
# Build Index
python scripts/mirage/build_simple_raptor_index.py \
  --doc-pool "$DOC_POOL" \
  --out-dir "${RESULT_ROOT}/mirage_raptor_index_200" \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL"

# Run QA
python scripts/mirage/run_simple_raptor.py \
  --dataset-path "$DATASET" \
  --index-dir "${RESULT_ROOT}/mirage_raptor_index_200" \
  --topk 5 \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_raptor_run_200"

# 7. Simple GraphRAG
echo "Running Simple GraphRAG..."
python scripts/mirage/run_simple_graphrag.py \
  --dataset-path "$DATASET" \
  --result-root "$RESULT_ROOT" \
  --index-dir "${RESULT_ROOT}/mirage_simple_graphrag_index_200" \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --new \
  --work-dir "${RESULT_ROOT}/mirage_graphrag_run_200"

# 8. Project Pipeline (Ano-RAG)
echo "Running Project Pipeline (Ano-RAG)..."
# Build Notes
export DATA_DIR="$DATA_DIR"
export DATASET="mirage"
export VLLM_MODEL="Qwen/Qwen3-30B-A3B"
# Force a new run for notes
bash scripts/mirage/build_notes.sh --new

# Determine the latest run directory created by build_notes.sh
LATEST_RUN=$(ls -td result/*-mirage | head -1)
echo "Using project run directory: $LATEST_RUN"

# Run Query
python scripts/mirage/query_dataset.py \
  --dataset mirage \
  --dataset-path "$DATASET" \
  --work-dir "$LATEST_RUN" \
  --lm-endpoint "$LM_ENDPOINT" \
  --lm-model "$LM_MODEL" \
  --out "$LATEST_RUN/answers.json"

# Symlink for evaluation script
ln -sfn "$(readlink -f "$LATEST_RUN")" "${RESULT_ROOT}/mirage_project_200"

echo "Experiments Completed."
