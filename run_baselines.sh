#!/bin/bash
set -e

# Configuration
DATASET="data/mirage_sample/dataset.json"
DOC_POOL="data/mirage_sample/doc_pool.json"
RESULT_ROOT="result"

echo "Using Mock LLM and CPU embeddings."

# 1. Vanilla RAG
echo "Running Vanilla RAG..."
python scripts/mirage/run_vanilla_rag.py \
  --dataset-path $DATASET \
  --lmstudio-endpoint mock \
  --lmstudio-model mock-model \
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
  --lmstudio-endpoint mock \
  --lmstudio-model mock-model \
  --work-dir $RESULT_ROOT/mirage_fid_rag \
  --new

# 3. Simple Self-RAG
echo "Running Simple Self-RAG..."
python scripts/mirage/run_simple_selfrag.py \
  --dataset-path $DATASET \
  --lmstudio-endpoint mock \
  --lmstudio-model mock-model \
  --new

# 4. Simple Raptor
echo "Running Simple Raptor..."
# Build index
python scripts/mirage/build_simple_raptor_index.py \
  --doc-pool $DOC_POOL \
  --out-dir $RESULT_ROOT/mirage_raptor_index \
  --lmstudio-endpoint mock \
  --lmstudio-model mock-model

# Run QA
python scripts/mirage/run_simple_raptor.py \
  --dataset-path $DATASET \
  --index-dir $RESULT_ROOT/mirage_raptor_index \
  --lmstudio-endpoint mock \
  --lmstudio-model mock-model \
  --work-dir $RESULT_ROOT/mirage_raptor_run \
  --new

# 5. Simple GraphRAG
echo "Running Simple GraphRAG..."
python scripts/mirage/run_simple_graphrag.py \
  --dataset-path $DATASET \
  --lmstudio-endpoint mock \
  --lmstudio-model mock-model \
  --work-dir $RESULT_ROOT/mirage_graphrag_run \
  --new

echo "All baselines finished."
