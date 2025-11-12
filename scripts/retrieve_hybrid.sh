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
notes_path = cfg.get("notes.out_path", "notes/notes.jsonl")
weak_new = os.path.join(os.path.dirname(notes_path), "weak", "weak_notes.jsonl")
weak_legacy = os.path.join(os.path.dirname(notes_path), "weak_notes.jsonl")
if os.path.exists(weak_new):
    weak_path = weak_new
elif os.path.exists(weak_legacy):
    weak_path = weak_legacy
else:
    weak_path = None
store = NoteStore(notes_path, weak_path)

question = os.environ["QUESTION"]
result = retrieve_answer(question, indexes, store)
print(result)
PYCODE
