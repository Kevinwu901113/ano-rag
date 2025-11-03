import argparse
import os
from pathlib import Path

from loguru import logger

from config import config
from pipeline.structured_builder import StructuredBuilder
from query import QueryProcessor
from utils import setup_logging


def _resolve_path(base: str, default: str) -> str:
    return base if base else default


def process_docs(args) -> None:
    cfg = config.load_config()

    data_dir = _resolve_path(
        args.data_dir, cfg.get("data_dir", cfg.get("storage.source_docs_dir", "data"))
    )
    notes_out = _resolve_path(args.notes_out, cfg.get("notes.out_path", "notes/notes.jsonl"))
    indexes_dir = _resolve_path(
        args.indexes_dir, cfg.get("notes.indexes_dir", "indexes")
    )
    vllm_endpoint = args.vllm_endpoint or cfg.get("vllm.endpoint")
    vllm_model = args.vllm_model or cfg.get("vllm.model")

    setup_logging(os.path.join(Path(notes_out).parent, "ano-rag-build.log"))

    builder = StructuredBuilder(
        vllm_endpoint=vllm_endpoint,
        vllm_model=vllm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    metrics = builder.build(
        data_dir=data_dir,
        notes_out=notes_out,
        indexes_dir=indexes_dir,
        chunks_out=args.chunks_out,
    )
    logger.info("Build complete: {}", metrics)


def query_mode(args) -> None:
    cfg = config.load_config()

    indexes_dir = _resolve_path(args.indexes_dir, cfg.get("notes.indexes_dir", "indexes"))
    notes_path = _resolve_path(args.notes, cfg.get("notes.out_path", "notes/notes.jsonl"))

    processor = QueryProcessor(
        indexes_dir=indexes_dir,
        notes_path=notes_path,
        lmstudio_endpoint=args.lmstudio_endpoint or cfg.get("lmstudio.endpoint"),
        lmstudio_model=args.lmstudio_model or cfg.get("lmstudio.model"),
    )

    result = processor.process(args.question)
    print("STRUCTURED:", result["structured"])
    print("FINAL ANSWER:", result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="ANO-RAG structured pipeline")
    subparsers = parser.add_subparsers(dest="cmd")

    proc = subparsers.add_parser("process", help="Chunk documents, generate notes, and build indexes")
    proc.add_argument("--data-dir", default=None, help="Directory containing source documents")
    proc.add_argument("--notes-out", default=None, help="Output JSONL path for structured notes")
    proc.add_argument("--indexes-dir", default=None, help="Directory to store indexes")
    proc.add_argument("--chunks-out", default=None, help="Optional path for serialized chunks.jsonl")
    proc.add_argument("--vllm-endpoint", default=None, help="vLLM endpoint URL")
    proc.add_argument("--vllm-model", default=None, help="vLLM model name")
    proc.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    proc.add_argument("--max-tokens", type=int, default=700, help="Max tokens for vLLM generation")

    query_cmd = subparsers.add_parser("query", help="Query the structured indexes")
    query_cmd.add_argument("question", help="Question to answer")
    query_cmd.add_argument("--indexes-dir", default=None, help="Directory containing indexes")
    query_cmd.add_argument("--notes", default=None, help="Structured notes JSONL path")
    query_cmd.add_argument("--lmstudio-endpoint", default=None, help="LM Studio endpoint URL")
    query_cmd.add_argument("--lmstudio-model", default=None, help="LM Studio model name")

    args = parser.parse_args()
    if args.cmd == "process":
        process_docs(args)
    elif args.cmd == "query":
        query_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
