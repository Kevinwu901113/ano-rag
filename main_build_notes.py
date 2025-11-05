import argparse
import json
import concurrent.futures
import os
from pathlib import Path

from loguru import logger

from adapters import get_adapter
from generator.note_generator import NoteGenerator
from indexer.index_builder import IndexBuilder


def build_notes(
    dataset: str,
    data_dir: str,
    notes_out: str,
    indexes_dir: str,
    vllm_endpoint: str,
    vllm_model: str,
    shard_idx: int = 0,
    shard_cnt: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> dict:
    adapter = get_adapter(dataset)
    generator = NoteGenerator(vllm_endpoint, vllm_model, temperature=temperature, max_tokens=max_tokens)

    if shard_cnt < 1:
        raise ValueError(f"shard_cnt must be >= 1 (got {shard_cnt})")
    if not 0 <= shard_idx < shard_cnt:
        raise ValueError(f"shard_idx must be in [0, {shard_cnt - 1}] (got {shard_idx})")

    notes_path = Path(notes_out)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    # Concurrency controls (per shard)
    from config.config_loader import config as global_config
    ccfg = (global_config.get("vllm", {}) or {}).get("concurrency", {})
    max_workers = int(ccfg.get("max_workers", 8))
    batch_size = int(ccfg.get("batch_size", 1))

    def _process_one(chunk):
        try:
            return generator.generate_for_chunk(chunk)
        except Exception as exc:
            logger.warning("Shard {} chunk failed doc={} chunk={} err={}", shard_idx, chunk.get("doc_id"), chunk.get("chunk_id"), exc)
            return []

    written = 0
    with open(notes_path, "w", encoding="utf-8") as handle:
        if max_workers <= 1:
            for idx, (_doc, chunk) in enumerate(adapter(data_dir)):
                if idx % shard_cnt != shard_idx:
                    continue
                for note in _process_one(chunk):
                    handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                    written += 1
        else:
            # Collect shard-specific chunks first to avoid scheduling overhead
            shard_chunks = []
            for idx, (_doc, chunk) in enumerate(adapter(data_dir)):
                if idx % shard_cnt == shard_idx:
                    shard_chunks.append(chunk)

            # Saturate pool with continuous submission, not batch-gated
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                inflight = set()
                i = 0
                n = len(shard_chunks)
                # Prime
                while i < n and len(inflight) < max_workers:
                    inflight.add(executor.submit(_process_one, shard_chunks[i]))
                    i += 1
                # Maintain saturation
                while inflight:
                    done, inflight = concurrent.futures.wait(inflight, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        notes = fut.result()
                        for note in notes:
                            handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                            written += 1
                    while i < n and len(inflight) < max_workers:
                        inflight.add(executor.submit(_process_one, shard_chunks[i]))
                        i += 1

    logger.info("Notes written to {} (shard {}/{}; {} notes)", notes_out, shard_idx, shard_cnt, written)

    if shard_cnt == 1:
        builder = IndexBuilder()
        builder.build_from_jsonl(str(notes_path))
        builder.dump(indexes_dir)
        logger.info("Indexes dumped to {}", indexes_dir)
    else:
        logger.info("Skipping index build for shard mode (shard_cnt={})", shard_cnt)

    return {"notes_written": written, "shard_idx": shard_idx, "shard_cnt": shard_cnt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build notes and indexes for structured pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name (e.g., mirage)")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--out", required=True, help="Output notes JSONL path")
    parser.add_argument("--indexes_dir", required=True, help="Output indexes directory")
    parser.add_argument("--vllm_endpoint", required=True)
    parser.add_argument("--vllm_model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=700)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--shard-cnt", type=int, default=1)
    args = parser.parse_args()

    Path(args.indexes_dir).mkdir(parents=True, exist_ok=True)
    build_notes(
        dataset=args.dataset,
        data_dir=args.data_dir,
        notes_out=args.out,
        indexes_dir=args.indexes_dir,
        vllm_endpoint=args.vllm_endpoint,
        vllm_model=args.vllm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        shard_idx=args.shard_idx,
        shard_cnt=args.shard_cnt,
    )


if __name__ == "__main__":
    main()
