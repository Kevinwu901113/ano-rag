import argparse
import json
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

    written = 0
    with open(notes_path, "w", encoding="utf-8") as handle:
        for idx, (_doc, chunk) in enumerate(adapter(data_dir)):
            if idx % shard_cnt != shard_idx:
                continue
            notes = generator.generate_for_chunk(chunk)
            for note in notes:
                handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                written += 1

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
