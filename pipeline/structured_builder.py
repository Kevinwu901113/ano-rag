import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from doc.chunker import DocumentChunker
from generator.note_generator import NoteGenerator
from indexer.index_builder import IndexBuilder
from utils import FileUtils


def _slugify(value: str) -> str:
    value = value.strip()
    if not value:
        return "doc"
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^\w\-]+", "_", value)
    value = value.strip("_")
    return value or "doc"


class StructuredBuilder:
    """串联分块、笔记生成与索引构建的最小流水线。"""

    def __init__(
        self,
        vllm_endpoint: str,
        vllm_model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        if not vllm_endpoint or not vllm_model:
            raise ValueError("vLLM endpoint/model must be provided")
        self.chunker = DocumentChunker()
        self.note_generator = NoteGenerator(
            vllm_endpoint, vllm_model, temperature=temperature, max_tokens=max_tokens
        )

    def build(
        self,
        data_dir: str,
        notes_out: str,
        indexes_dir: str,
        *,
        chunks_out: Optional[str] = None,
    ) -> Dict[str, int]:
        files = FileUtils.list_files(data_dir, [".json", ".jsonl", ".docx"])
        if not files:
            logger.warning(f"No documents found in {data_dir}")
            return {"chunks": 0, "notes": 0}

        doc_counters = defaultdict(int)
        chunk_records: List[Dict[str, str]] = []

        for file_path in files:
            source_info = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_hash": FileUtils.get_file_hash(file_path),
            }
            raw_chunks = self.chunker.chunk_document(file_path, source_info)

            doc_id = _slugify(Path(file_path).stem)
            for chunk in raw_chunks:
                text = (chunk.get("text") or "").strip()
                if not text:
                    continue
                idx = doc_counters[doc_id]
                doc_counters[doc_id] += 1
                chunk_id = f"{doc_id}_{idx:04d}"
                chunk_records.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": text,
                        "source": {
                            "file_path": source_info["file_path"],
                            "file_name": source_info["file_name"],
                            "chunk_index": chunk.get("chunk_index", idx),
                        },
                    }
                )

        if not chunk_records:
            logger.warning("Chunking produced no usable text blocks.")
            return {"chunks": 0, "notes": 0}

        notes_path = Path(notes_out)
        notes_path.parent.mkdir(parents=True, exist_ok=True)

        if chunks_out is None:
            chunks_out = notes_path.parent / "chunks.jsonl"
        chunks_path = Path(chunks_out)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        with open(chunks_path, "w", encoding="utf-8") as handle:
            for chunk in chunk_records:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(chunk_records)} chunks to {chunks_path}")

        notes_written = 0
        with open(notes_path, "w", encoding="utf-8") as handle:
            for chunk in chunk_records:
                notes = self.note_generator.generate_for_chunk(chunk)
                for note in notes:
                    handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                    notes_written += 1

        logger.info(f"Wrote {notes_written} notes to {notes_path}")

        if notes_written:
            index_builder = IndexBuilder()
            index_builder.build_from_jsonl(str(notes_path))
            index_builder.dump(indexes_dir)
            logger.info(f"Indexes dumped to {indexes_dir}")
        else:
            logger.warning("No notes generated; skipping index build.")

        return {"chunks": len(chunk_records), "notes": notes_written}
