#!/bin/bash
set -e

# Configuration
DATASET="data/mirage_sample/dataset.json"
DOC_POOL="data/mirage_sample/doc_pool.json"
RESULT_ROOT="result"
LM_ENDPOINT="http://127.0.0.1:8000/v1"
LM_MODEL="qwen3-30b-a3b"

echo "Using vLLM endpoint ${LM_ENDPOINT} with model ${LM_MODEL}."

# 1. Vanilla RAG
echo "Running Vanilla RAG..."
python scripts/mirage/run_vanilla_rag.py \
  --dataset-path $DATASET \
  --lm-endpoint $LM_ENDPOINT \
  --lm-model $LM_MODEL \
  --new

# 2. FiD RAG
echo "Running FiD RAG..."
# Build index
python scripts/mirage/build_fid_index.py \
  --doc-pool $DOC_POOL \
  --out-dir $RESULT_ROOT/mirage_fid_index \
  --embed-device cpu \
  --embed-model /home/wjk/models/qwen3-emb

# Run QA
python scripts/mirage/run_fid_rag.py \
  --dataset-path $DATASET \
  --index-dir $RESULT_ROOT/mirage_fid_index \
  --lm-endpoint $LM_ENDPOINT \
  --lm-model $LM_MODEL \
  --work-dir $RESULT_ROOT/mirage_fid_rag \
  --new

# 3. Simple Self-RAG
echo "Running Simple Self-RAG..."
python scripts/mirage/run_simple_selfrag.py \
  --dataset-path $DATASET \
  --lm-endpoint $LM_ENDPOINT \
  --lm-model $LM_MODEL \
  --new

# 4. Simple Raptor
echo "Running Simple Raptor..."
# Build index
python scripts/mirage/build_simple_raptor_index.py \
  --doc-pool $DOC_POOL \
  --out-dir $RESULT_ROOT/mirage_raptor_index \
  --lm-endpoint $LM_ENDPOINT \
  --lm-model $LM_MODEL

# Run QA
python scripts/mirage/run_simple_raptor.py \
  --dataset-path $DATASET \
  --index-dir $RESULT_ROOT/mirage_raptor_index \
  --lm-endpoint $LM_ENDPOINT \
  --lm-model $LM_MODEL \
  --work-dir $RESULT_ROOT/mirage_raptor_run \
  --new

# 5. Simple GraphRAG
echo "Running Simple GraphRAG..."
python scripts/mirage/run_simple_graphrag.py \
  --dataset-path $DATASET \
  --lm-endpoint $LM_ENDPOINT \
  --lm-model $LM_MODEL \
  --work-dir $RESULT_ROOT/mirage_graphrag_run \
  --new

echo "All baselines finished."
