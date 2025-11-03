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
    temperature: float,
    max_tokens: int,
) -> None:
    adapter = get_adapter(dataset)
    generator = NoteGenerator(vllm_endpoint, vllm_model, temperature=temperature, max_tokens=max_tokens)

    notes_path = Path(notes_out)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    with open(notes_path, "w", encoding="utf-8") as handle:
        for _doc, chunk in adapter(data_dir):
            notes = generator.generate_for_chunk(chunk)
            for note in notes:
                handle.write(json.dumps(note, ensure_ascii=False) + "\n")

    logger.info("Notes written to {}", notes_out)

    builder = IndexBuilder()
    builder.build_from_jsonl(str(notes_path))
    builder.dump(indexes_dir)
    logger.info("Indexes dumped to {}", indexes_dir)


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
