import argparse
import os
from pathlib import Path

from loguru import logger

from config import config
from pipeline.structured_builder import StructuredBuilder
from query.query_processor import QueryProcessor
from utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured triple-note pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("process", help="Chunk documents, generate notes, build indexes")
    build.add_argument("--data-dir", required=True)
    build.add_argument("--notes-out", default=None)
    build.add_argument("--indexes-dir", default=None)
    build.add_argument("--vllm-endpoint", default=None)
    build.add_argument("--vllm-model", default=None)
    build.add_argument("--temperature", type=float, default=0.0)
    build.add_argument("--max-tokens", type=int, default=8000)

    query = sub.add_parser("query", help="Query the structured indexes")
    query.add_argument("question")
    query.add_argument("--indexes-dir", default=None)
    query.add_argument("--notes-path", default=None)
    query.add_argument("--lmstudio-endpoint", default=None)
    query.add_argument("--lmstudio-model", default=None)

    args = parser.parse_args()
    cfg = config.load_config()

    if args.cmd == "process":
        notes_out = args.notes_out or cfg.get("notes.out_path", "notes/notes.jsonl")
        indexes_dir = args.indexes_dir or cfg.get("notes.indexes_dir", "indexes")
        vllm_endpoint = args.vllm_endpoint or cfg.get("vllm.endpoint")
        vllm_model = args.vllm_model or cfg.get("vllm.model")

        if not vllm_endpoint or not vllm_model:
            raise ValueError("vLLM endpoint/model must be specified via CLI or config")

        log_path = Path(notes_out).parent / "ano-rag-build.log"
        setup_logging(str(log_path))

        Path(indexes_dir).mkdir(parents=True, exist_ok=True)

        builder = StructuredBuilder(
            endpoint=vllm_endpoint,
            model=vllm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        metrics = builder.build(
            data_dir=args.data_dir,
            notes_out=notes_out,
            indexes_dir=indexes_dir,
        )
        logger.info("Build metrics: {}", metrics)

    elif args.cmd == "query":
        setup_logging()
        processor = QueryProcessor(
            indexes_dir=args.indexes_dir,
            notes_path=args.notes_path,
            lmstudio_endpoint=args.lmstudio_endpoint,
            lmstudio_model=args.lmstudio_model,
        )
        result = processor.process(args.question)
        print(result)


if __name__ == "__main__":
    main()
