import json
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from config.config_loader import config as global_config

from doc import make_chunks
from generator.note_generator import NoteGenerator
from indexer.index_builder import IndexBuilder
from utils import FileUtils


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class StructuredBuilder:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("vLLM endpoint/model must be provided")
        self.generator = NoteGenerator(endpoint, model, temperature, max_tokens)

    def _collect_chunks(self, data_dir: Path) -> List[Dict]:
        files = FileUtils.list_files(str(data_dir), [".txt", ".md", ".jsonl", ".json"])
        if not files:
            logger.warning("No documents found in {}", data_dir)
            return []

        chunks: List[Dict] = []
        for file_path in files:
            path = Path(file_path)
            doc_id = path.stem
            if path.suffix.lower() == ".jsonl":
                for idx, row in enumerate(FileUtils.read_jsonl(str(path))):
                    text = row.get("text") or row.get("content") or ""
                    if not isinstance(text, str):
                        continue
                    chunks.extend(
                        make_chunks(f"{doc_id}_{idx:04d}", text, chunk_id_prefix="c")
                    )
            else:
                text = _read_text(path)
                chunks.extend(make_chunks(doc_id, text, chunk_id_prefix="c"))
        return chunks

    def build(
        self,
        data_dir: str,
        notes_out: str,
        indexes_dir: str,
        chunks_out: Optional[str] = None,
    ) -> Dict[str, int]:
        chunk_records = self._collect_chunks(Path(data_dir))
        if not chunk_records:
            return {"chunks": 0, "notes": 0}

        notes_path = Path(notes_out)
        notes_path.parent.mkdir(parents=True, exist_ok=True)

        if chunks_out is None:
            chunks_out = str(notes_path.parent / "chunks.jsonl")
        FileUtils.write_jsonl(chunks_out, chunk_records)
        logger.info("Wrote {} chunks to {}", len(chunk_records), chunks_out)

        # Concurrency settings
        vllm_cfg = global_config.get("vllm", {}) if 'global_config' in globals() else {}
        ccfg = (vllm_cfg or {}).get("concurrency", {})
        max_workers = int(ccfg.get("max_workers", 8))
        batch_size = int(ccfg.get("batch_size", 1))

        def _process_one(chunk):
            try:
                return self.generator.generate_for_chunk(chunk)
            except Exception as exc:
                logger.warning("Chunk generation failed doc={} chunk={} err={}", chunk.get("doc_id"), chunk.get("chunk_id"), exc)
                return []

        notes_written = 0
        with open(notes_path, "w", encoding="utf-8") as handle:
            if max_workers <= 1:
                for chunk in chunk_records:
                    for note in _process_one(chunk):
                        handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                        notes_written += 1
            else:
                # Thread pool for parallel chunk processing, keep pool saturated
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    inflight = set()
                    idx = 0
                    n_total = len(chunk_records)
                    # Prime the pool
                    while idx < n_total and len(inflight) < max_workers:
                        fut = executor.submit(_process_one, chunk_records[idx])
                        inflight.add(fut)
                        idx += 1
                    # As each future completes, submit next to maintain saturation
                    while inflight:
                        done, inflight = concurrent.futures.wait(inflight, return_when=concurrent.futures.FIRST_COMPLETED)
                        for fut in done:
                            notes = fut.result()
                            for note in notes:
                                handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                                notes_written += 1
                        # Refill up to max_workers
                        while idx < n_total and len(inflight) < max_workers:
                            fut = executor.submit(_process_one, chunk_records[idx])
                            inflight.add(fut)
                            idx += 1

        logger.info("Wrote {} notes to {}", notes_written, notes_path)

        if notes_written:
            builder = IndexBuilder()
            builder.build_from_jsonl(str(notes_path))
            builder.dump(indexes_dir)
            logger.info("Indexes dumped to {}", indexes_dir)
        else:
            logger.warning("No notes generated; skipping index build.")

        return {"chunks": len(chunk_records), "notes": notes_written}
