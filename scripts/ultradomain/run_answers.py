from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from relrag.config.config_loader import config as config_loader
from relrag.retriever.bm25_client import BM25Client
from relrag.retriever.embedding_client import EmbeddingClient
from relrag.retriever.fusion import fuse_rankings
from relrag.retriever.operators import Indexes
from relrag.retriever.note_store import NoteStore
from relrag.retriever.pipeline import retrieve_answer
from relrag.utils.context_budget import budget_answer_prompt
from relrag.utils.output_eval import extract_final_answer, has_final_tag
from relrag.utils.openai_client import chat_completion

from scripts.ultradomain.common import (
    ANSWERS_DIR,
    CHUNKS_DIR,
    DOMAIN_LABELS,
    INDEX_DIR,
    QUESTIONS_DIR,
    TOKENIZER_ID,
    assemble_budgeted_items,
    chunk_note_id,
    count_tokens,
    ensure_dirs,
    get_api_key,
    now_iso,
    read_json,
    read_jsonl,
    write_jsonl,
)

SYSTEMS = ["RelRAG-full", "BM25-only", "Dense-only", "Hybrid-only"]


def _load_questions(domain: str) -> List[Dict[str, Any]]:
    data = read_json(QUESTIONS_DIR / f"questions_{domain}.json")
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list):
        raise ValueError("questions file missing questions list")
    return questions


def _load_chunk_map(domain: str) -> Dict[str, Dict[str, Any]]:
    path = CHUNKS_DIR / f"chunks_{domain}.jsonl"
    mapping: Dict[str, Dict[str, Any]] = {}
    for chunk in read_jsonl(path):
        doc_id = str(chunk.get("doc_id") or "")
        chunk_id = str(chunk.get("chunk_id") or "")
        if not doc_id or not chunk_id:
            continue
        note_id = chunk_note_id(doc_id, chunk_id)
        mapping[note_id] = chunk
    return mapping


def _build_bm25_client(domain: str, base_cfg: Dict[str, Any]) -> BM25Client:
    cfg = deepcopy((base_cfg.get("retriever") or {}).get("bm25") or {})
    cfg["enabled"] = True
    cfg["store_path"] = str(INDEX_DIR / f"chunk_bm25_{domain}")
    cfg["topn"] = 40
    return BM25Client(cfg)


def _build_dense_client(domain: str, base_cfg: Dict[str, Any]) -> EmbeddingClient:
    cfg = deepcopy((base_cfg.get("retriever") or {}).get("embedding") or {})
    cfg["enabled"] = True
    cfg["offline_index_path"] = str(INDEX_DIR / f"chunk_faiss_{domain}" / "notes.faiss")
    cfg["meta_path"] = str(INDEX_DIR / f"chunk_faiss_{domain}" / "notes.meta.parquet")
    cfg["topn"] = 40
    return EmbeddingClient(cfg)


def _retrieve_chunk_system(
    question: str,
    system: str,
    chunk_map: Dict[str, Dict[str, Any]],
    bm25: BM25Client,
    dense: EmbeddingClient,
) -> List[Dict[str, Any]]:
    if system == "BM25-only":
        results = bm25.search(question, topn=40)
    elif system == "Dense-only":
        results = dense.search(question, topn=40)
    elif system == "Hybrid-only":
        bm25_results = bm25.search(question, topn=40)
        dense_results = dense.search(question, topn=40)
        fused = fuse_rankings(
            {"bm25": bm25_results, "emb": dense_results},
            weights={"bm25": 1.0, "emb": 1.0},
            rrf_k=60,
        )
        results = fused
    else:
        raise ValueError(f"unsupported system: {system}")

    evidences: List[Dict[str, Any]] = []
    for item in results:
        note_id = item.get("note_id")
        if not note_id:
            continue
        chunk = chunk_map.get(note_id)
        if not chunk:
            continue
        text = chunk.get("text") or ""
        if not text:
            continue
        evidences.append(
            {
                "note_id": note_id,
                "evidence": text,
                "canonical": text,
                "score": item.get("score"),
                "token_count": chunk.get("token_count") or count_tokens(text),
            }
        )
    return evidences


def _build_relrag_cfg(domain: str, base_cfg: Dict[str, Any], base_url: str, api_key_env: str, model: str) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    retr = cfg.setdefault("retriever", {})
    retr.setdefault("structured", {})
    retr.setdefault("embedding", {})
    retr.setdefault("bm25", {})
    retr.setdefault("hybrid", {})
    retr.setdefault("fusion", {})
    retr["structured"]["enabled"] = True
    retr["structured"]["top_k"] = 40
    retr["embedding"]["enabled"] = True
    retr["embedding"]["offline_index_path"] = str(INDEX_DIR / f"relrag_{domain}" / "faiss" / "notes.faiss")
    retr["embedding"]["meta_path"] = str(INDEX_DIR / f"relrag_{domain}" / "faiss" / "notes.meta.parquet")
    retr["embedding"]["topn"] = 40
    retr["bm25"]["enabled"] = True
    retr["bm25"]["store_path"] = str(INDEX_DIR / f"relrag_{domain}")
    retr["bm25"]["topn"] = 40
    retr["hybrid"]["enabled"] = True

    reranker = cfg.setdefault("reranker", {})
    reranker["enabled"] = True
    reranker["provider"] = "openai"
    reranker["openai"] = {
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "timeout_sec": 60.0,
        "max_retries": 2,
        "retry_backoff_sec": 1.0,
        "retry_backoff_max_sec": 20.0,
    }
    return cfg


def _build_answer_budget_cfg() -> Dict[str, Any]:
    return {
        "llm": {"max_context_len": 128000, "safety_margin_tokens": 256},
        "answer": {"max_evidence_items": 10000, "max_evidence_tokens": None},
    }


def _call_answer_llm(messages: List[Dict[str, str]], *, model: str, base_url: str, api_key: str, temperature: float, top_p: float, max_tokens: int) -> str:
    return chat_completion(
        messages,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"top_p": top_p},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UltraDomain answers for Mix/Legal.")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--system", default="all")
    parser.add_argument("--base_url", default="https://api.deepseek.com/v1")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--budget_tokens", type=int, default=12000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    api_key = get_api_key(args.api_key_env)
    base_cfg = config_loader.load_config()
    domains = list(DOMAIN_LABELS.keys()) if args.domain == "all" else [args.domain]
    systems = SYSTEMS if args.system == "all" else [args.system]

    for domain in domains:
        if domain not in DOMAIN_LABELS:
            continue
        questions = _load_questions(domain)
        chunk_map = _load_chunk_map(domain)
        bm25 = _build_bm25_client(domain, base_cfg)
        dense = _build_dense_client(domain, base_cfg)

        relrag_indexes = None
        relrag_notes = None
        relrag_cfg = None
        if "RelRAG-full" in systems:
            relrag_indexes = Indexes(str(INDEX_DIR / f"relrag_{domain}" / "indexes"))
            relrag_notes = NoteStore(str(INDEX_DIR / f"relrag_{domain}" / "notes.jsonl"))
            relrag_cfg = _build_relrag_cfg(domain, base_cfg, args.base_url, args.api_key_env, args.model)

        for system in systems:
            out_dir = ANSWERS_DIR / domain
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{system}.jsonl"
            existing = set()
            if args.resume and out_path.exists():
                for row in read_jsonl(out_path):
                    qid = row.get("question_id")
                    if qid:
                        existing.add(qid)

            rows: List[Dict[str, Any]] = []
            for q in questions:
                qid = q.get("question_id")
                question = q.get("question")
                if not qid or not question:
                    continue
                if qid in existing:
                    continue

                if system == "RelRAG-full":
                    if relrag_indexes is None or relrag_notes is None or relrag_cfg is None:
                        raise RuntimeError("RelRAG indexes not initialized")
                    result = retrieve_answer(
                        question=question,
                        indexes=relrag_indexes,
                        note_store=relrag_notes,
                        cfg=relrag_cfg,
                    )
                    evidences = result.get("evidence") or []
                else:
                    evidences = _retrieve_chunk_system(question, system, chunk_map, bm25, dense)

                selected, used_tokens = assemble_budgeted_items(evidences, args.budget_tokens)
                budget_cfg = _build_answer_budget_cfg()
                budgeted = budget_answer_prompt(
                    question,
                    selected,
                    prompt_name="answerer_openai.txt",
                    label_instruction="",
                    system_prompt=None,
                    cfg=budget_cfg,
                    llm_cfg={"max_context_len": 128000},
                    requested_max_tokens=args.max_output_tokens,
                    max_items_override=10000,
                    max_item_tokens_override=None,
                    include_raw_evidence=False,
                )
                answer_raw = _call_answer_llm(
                    budgeted.messages,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=api_key,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_output_tokens,
                )
                answer_final = extract_final_answer(answer_raw)
                rows.append(
                    {
                        "question_id": qid,
                        "question": question,
                        "answer_raw": answer_raw,
                        "answer_final": answer_final,
                        "has_final": has_final_tag(answer_raw),
                        "retrieved_context_ids": [item.get("note_id") for item in selected if item.get("note_id")],
                        "token_budget_used": used_tokens,
                        "run_meta": {
                            "system": system,
                            "domain": domain,
                            "model": args.model,
                            "base_url": args.base_url,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_output_tokens": args.max_output_tokens,
                            "budget_tokens": args.budget_tokens,
                            "prefilter_top_k": 40,
                            "generated_at": now_iso(),
                        },
                    }
                )

            if rows:
                mode = "a" if args.resume and out_path.exists() else "w"
                with out_path.open(mode, encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")
            print(f"Wrote answers for {domain} {system} -> {out_path}")


if __name__ == "__main__":
    main()
