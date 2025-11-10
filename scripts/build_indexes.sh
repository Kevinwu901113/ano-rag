#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[build_indexes] building embedding index..."
python -m indexer.embedding_index

echo "[build_indexes] building bm25 index..."
python -m indexer.bm25_index
