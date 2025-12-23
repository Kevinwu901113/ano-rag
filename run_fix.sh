#!/bin/bash
set -e

# Configuration
DATASET="data/mirage_sample/dataset.json"
DOC_POOL="data/mirage_sample/doc_pool.json"
RESULT_ROOT="result"

echo "Re-running Simple Raptor and Simple GraphRAG with fixes..."

# 4. Simple Raptor
echo "Running Simple Raptor (FIXED)..."
# Build index
python scripts/mirage/build_simple_raptor_index.py   --doc-pool $DOC_POOL   --out-dir $RESULT_ROOT/mirage_raptor_index   --lm-endpoint http://127.0.0.1:8000/v1   --lm-model qwen3-30b-a3b

# Run QA
python scripts/mirage/run_simple_raptor.py   --dataset-path $DATASET   --index-dir $RESULT_ROOT/mirage_raptor_index   --lm-endpoint http://127.0.0.1:8000/v1   --lm-model qwen3-30b-a3b   --work-dir $RESULT_ROOT/mirage_raptor_run   --new

# 5. Simple GraphRAG
echo "Running Simple GraphRAG (FIXED)..."
python scripts/mirage/run_simple_graphrag.py   --dataset-path $DATASET   --lm-endpoint http://127.0.0.1:8000/v1   --lm-model qwen3-30b-a3b   --work-dir $RESULT_ROOT/mirage_graphrag_run   --new

echo "Fix run finished."
