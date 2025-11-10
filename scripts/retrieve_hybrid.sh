#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 'question text'"
  exit 1
fi

QUESTION="$1"
export QUESTION

python - <<'PYCODE'
import os
from config import config
from retriever.note_store import NoteStore
from retriever.operators import Indexes
from retriever.pipeline import retrieve_answer

cfg = config.load_config()
indexes = Indexes(cfg.get("notes.indexes_dir", "indexes"))
store = NoteStore(cfg.get("notes.out_path", "notes/notes.jsonl"))

question = os.environ["QUESTION"]
result = retrieve_answer(question, indexes, store)
print(result)
PYCODE
