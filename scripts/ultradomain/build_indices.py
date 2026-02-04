from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from relrag.api import build_index
from relrag.config.config_loader import config as config_loader
from relrag.indexer.embedding_index import EmbeddingIndexBuilder
from scripts.ultradomain.common import (
    CHUNKS_DIR,
    DOMAIN_LABELS,
    INDEX_DIR,
    RUN_META_DIR,
    chunk_note_id,
    ensure_dirs,
    read_jsonl,
    write_jsonl,
)


def _write_chunk_notes(domain: str, out_dir: Path) -> Path:
    chunks_path = CHUNKS_DIR / f"chunks_{domain}.jsonl"
    notes_path = out_dir / "notes.jsonl"
    notes: List[Dict[str, Any]] = []
    for chunk in read_jsonl(chunks_path):
        doc_id = str(chunk.get("doc_id") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        text = chunk.get("text") or ""
        if not doc_id or not chunk_id or not text:
            continue
        note_id = chunk_note_id(doc_id, chunk_id)
        notes.append(
            {
                "note_id": note_id,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "dataset": domain,
                "text": text,
                "evidence": text,
                "subj": doc_id,
                "pred": "chunk",
                "obj": "",
                "fields": {"ctx": text},
            }
        )
    write_jsonl(notes_path, notes)
    return notes_path


def _build_chunk_faiss(notes_path: Path, index_dir: Path, base_cfg: Dict[str, Any], embed_overrides: Dict[str, Any]) -> None:
    cfg = {
        "notes": {"out_path": str(notes_path)},
        "system": base_cfg.get("system") or {},
        "retriever": {"embedding": embed_overrides},
    }
    builder = EmbeddingIndexBuilder(cfg)
    builder.build()


def _prepare_embed_cfg(base_cfg: Dict[str, Any], index_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    embed_cfg = (base_cfg.get("retriever") or {}).get("embedding") or {}
    embed_cfg = dict(embed_cfg)
    embed_cfg.update(overrides)
    embed_cfg["enabled"] = True
    embed_cfg["offline_index_path"] = str(index_dir / "notes.faiss")
    embed_cfg["meta_path"] = str(index_dir / "notes.meta.parquet")
    return embed_cfg


def _build_relrag_index(domain: str, args: argparse.Namespace, base_cfg: Dict[str, Any]) -> None:
    docs_path = RUN_META_DIR / f"docs_{domain}.jsonl"
    out_dir = INDEX_DIR / f"relrag_{domain}"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_endpoint = args.llm_endpoint or (base_cfg.get("vllm") or {}).get("endpoint")
    llm_model = args.llm_model or (base_cfg.get("vllm") or {}).get("model")
    llm_provider = args.llm_provider
    build_index(
        docs_input=str(docs_path),
        output_dir=str(out_dir),
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        llm_provider=llm_provider,
        llm_api_key=args.llm_api_key,
    )


def _build_relrag_faiss(domain: str, base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    out_dir = INDEX_DIR / f"relrag_{domain}"
    notes_path = out_dir / "notes.jsonl"
    index_dir = out_dir / "faiss"
    index_dir.mkdir(parents=True, exist_ok=True)
    embed_cfg = _prepare_embed_cfg(base_cfg, index_dir, overrides)
    cfg = {
        "notes": {"out_path": str(notes_path)},
        "system": base_cfg.get("system") or {},
        "retriever": {"embedding": embed_cfg},
    }
    builder = EmbeddingIndexBuilder(cfg)
    builder.build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build UltraDomain indices for Mix/Legal.")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--skip_relrag", action="store_true")
    parser.add_argument("--skip_chunk", action="store_true")
    parser.add_argument("--llm_provider", default="vllm", choices=["vllm", "openai"])
    parser.add_argument("--llm_endpoint", default=None)
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--llm_api_key", default=None)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_max_tokens", type=int, default=None)
    parser.add_argument("--embed_provider", default=None)
    parser.add_argument("--embed_model", default=None)
    parser.add_argument("--embed_endpoint", default=None)
    parser.add_argument("--embed_api_key", default=None)
    args = parser.parse_args()

    ensure_dirs()
    base_cfg = config_loader.load_config()
    domains = list(DOMAIN_LABELS.keys()) if args.domain == "all" else [args.domain]

    embed_overrides: Dict[str, Any] = {}
    if args.embed_provider:
        embed_overrides["provider"] = args.embed_provider
    if args.embed_model:
        embed_overrides["model"] = args.embed_model
    if args.embed_endpoint:
        embed_overrides["endpoint"] = args.embed_endpoint
    if args.embed_api_key:
        embed_overrides["api_key"] = args.embed_api_key

    for domain in domains:
        if domain not in DOMAIN_LABELS:
            continue
        if not args.skip_chunk:
            bm25_dir = INDEX_DIR / f"chunk_bm25_{domain}"
            faiss_dir = INDEX_DIR / f"chunk_faiss_{domain}"
            bm25_dir.mkdir(parents=True, exist_ok=True)
            faiss_dir.mkdir(parents=True, exist_ok=True)
            bm25_notes = _write_chunk_notes(domain, bm25_dir)
            faiss_notes = _write_chunk_notes(domain, faiss_dir)
            embed_cfg = _prepare_embed_cfg(base_cfg, faiss_dir, embed_overrides)
            _build_chunk_faiss(faiss_notes, faiss_dir, base_cfg, embed_cfg)
            print(f"Chunk indices ready for {domain}: {bm25_notes} / {faiss_dir}")
        if not args.skip_relrag:
            _build_relrag_index(domain, args, base_cfg)
            _build_relrag_faiss(domain, base_cfg, embed_overrides)
            print(f"RelRAG indices ready for {domain}")


if __name__ == "__main__":
    main()
