from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from config import config
from generator.answerer import call_lmstudio
from retriever.pipeline import retrieve_answer


class QueryProcessor:
    """面向结构化索引的最小查询管线。"""

    def __init__(
        self,
        *,
        indexes_dir: Optional[str] = None,
        notes_path: Optional[str] = None,
        lmstudio_endpoint: Optional[str] = None,
        lmstudio_model: Optional[str] = None,
    ) -> None:
        cfg = config.load_config()
        self.indexes_dir = indexes_dir or cfg.get("notes.indexes_dir", "indexes")
        self.notes_path = notes_path or cfg.get("notes.out_path", "notes/notes.jsonl")

        self.lmstudio_endpoint = lmstudio_endpoint or cfg.get("lmstudio.endpoint")
        self.lmstudio_model = lmstudio_model or cfg.get("lmstudio.model")

        if not self.indexes_dir:
            raise ValueError("indexes_dir must be provided")
        if not Path(self.indexes_dir).exists():
            raise FileNotFoundError(
                f"Indexes directory '{self.indexes_dir}' not found. Run 'main.py process' first."
            )

        required = [
            "entity_to_notes.json",
            "predicate_to_notes.json",
            "type_edge_index.json",
            "graph_edges.jsonl",
            "inverse_edges.jsonl",
        ]
        missing = [name for name in required if not Path(self.indexes_dir, name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing index files: {missing}. Rebuild notes with 'main.py process'."
            )

        if not self.notes_path:
            raise ValueError("notes_path must be provided")
        if not Path(self.notes_path).exists():
            raise FileNotFoundError(
                f"Notes file '{self.notes_path}' not found. Run 'main.py process' first."
            )

    def process(self, question: str) -> Dict[str, Any]:
        logger.info("Running structured retrieval for question: {}", question)
        structured = retrieve_answer(question, self.indexes_dir, self.notes_path)

        final_answer = "Insufficient evidence"
        if structured.get("answer"):
            if not self.lmstudio_endpoint or not self.lmstudio_model:
                raise ValueError("LM Studio endpoint/model must be configured")
            final_answer = call_lmstudio(
                self.lmstudio_endpoint,
                self.lmstudio_model,
                question,
                structured.get("evidence", []),
            )

        return {"structured": structured, "answer": final_answer}
