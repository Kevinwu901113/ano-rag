from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from baselines.fid_rag.retriever import NaiveIndex
from config import config as config_loader
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from baselines.common.model_clients import get_default_llm_client


PROMPT_TEMPLATE = """Answer the question based on the context.
Keep the answer concise.
{final_instruction}

{context}

Question: {question}
Answer:"""


class FiDRAGRunner:
    """FiD-style RAG baseline with NaiveIndex retriever and LM Studio backend."""

    def __init__(
        self,
        index_path: str,
        chunks_path: str,
        *,
        topk: int = 5,
        context_budget: Optional[int] = None,
        lm_endpoint: Optional[str] = None,
        lm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = config or config_loader.load_config()
        self.topk = max(1, int(topk))
        self.context_budget = int(context_budget or 0)
        lm_cfg = self.cfg.get("lmstudio", {}) or {}
        endpoint = lm_endpoint or lm_cfg.get("endpoint")
        model = lm_model or lm_cfg.get("model")
        
        # Ensure endpoint and model are strings
        endpoint = str(endpoint) if endpoint else None
        model = str(model) if model else None
        
        temp = temperature if temperature is not None else lm_cfg.get("temperature", 0.0)
        max_new_tokens = max_tokens if max_tokens is not None else lm_cfg.get("max_tokens", 128)
        self.temperature = float(temp or 0.0)
        self.max_new_tokens = int(max_new_tokens or 128)
        self.retriever = NaiveIndex(index_path, chunks_path, config=self.cfg)
        
        # Override config if args provided
        if endpoint or model:
            # We need to ensure get_default_llm_client uses these overrides.
            # But get_default_llm_client reads from config dict.
            # So we create a temporary config dict with overrides.
            # Or just update self.cfg['lmstudio']?
            # Safer to just rely on global config updates if passed via CLI, 
            # BUT arguments here are passed explicitly.
            # Let's just update self.cfg['lmstudio'] locally.
            if "lmstudio" not in self.cfg:
                self.cfg["lmstudio"] = {}
            if endpoint:
                self.cfg["lmstudio"]["endpoint"] = endpoint
            if model:
                self.cfg["lmstudio"]["model"] = model

        self.lm = get_default_llm_client(self.cfg)

    def run_dataset(
        self,
        dataset: Iterable[Dict[str, Any]],
        *,
        work_dir: str,
        limit: Optional[int] = None,
        debug: bool = True,
        dataset_name: str = "mirage",
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        items = list(dataset)
        if limit:
            items = items[:limit]
        work_path = Path(work_dir)
        preds_dir = work_path / "preds"
        artifacts_dir = work_path / "artifacts"
        preds_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        resolved_run_name = run_name or work_path.name
        answers: List[Dict[str, Any]] = []
        qa_rows: List[str] = []
        qa_rows_no_header: List[str] = []
        qa_with_q: List[str] = []
        debug_records: List[str] = []
        pred_raw_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            question = item.get("query") or item.get("question") or ""
            qid = item.get("query_id") or str(idx)
            hits = self.retriever.search(question, self.topk)
            annotated_hits = []
            for rank, hit in enumerate(hits, start=1):
                annotated_hits.append({**hit, "text": f"[{rank}] {hit.get('text', '')}"})
            context_str, contexts_used, context_tokens = pack_contexts(
                annotated_hits, self.context_budget
            )
            prompt = PROMPT_TEMPLATE.format(
                context=context_str,
                question=question,
                final_instruction=build_final_instruction(),
            )
            raw_output = (
                self.lm.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                if prompt.strip()
                else ""
            )
            answers.append(
                {
                    "query_id": qid,
                    "question": question,
                    "answer": raw_output,
                    "hits": hits,
                }
            )
            qa_rows.append(f"{question}\t{raw_output}")
            qa_rows_no_header.append(f"{question}\t{raw_output}")
            qa_with_q.append(f"{question}\t{raw_output}")
            try:
                log_retrieval(
                    sample_id=qid,
                    dataset=dataset_name,
                    run_name=resolved_run_name,
                    retrieved=[
                        {**hit, "rank": idx + 1} for idx, hit in enumerate(hits)
                    ],
                    topk=len(hits),
                    final_context=contexts_used,
                    final_context_tokens=context_tokens,
                    context_budget_tokens=self.context_budget or None,
                    log_dir=artifacts_dir,
                )
            except Exception as log_exc:
                logger.error("retrieval logging failed for {}: {}", qid, log_exc)
            if debug:
                debug_records.append(
                    json.dumps(
                        {
                            "query_id": qid,
                            "question": question,
                            "answer": raw_output,
                            "hits": hits,
                        },
                        ensure_ascii=False,
                    )
                )
            pred_raw_records.append(
                {
                    "id": str(qid),
                    "question": question,
                    "pred_raw": raw_output,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": self.context_budget or None,
                }
            )

        answers_path = preds_dir / "answers.json"
        qa_path = preds_dir / "qa.tsv"
        qa_no_header_path = preds_dir / "qa.no_header.tsv"
        qa_q_path = preds_dir / "qa_with_question.tsv"
        debug_dir = artifacts_dir / "debug"
        answers_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        qa_path.write_text("\n".join(qa_rows), encoding="utf-8")
        qa_no_header_path.write_text("\n".join(qa_rows_no_header), encoding="utf-8")
        qa_q_path.write_text("\n".join(qa_with_q), encoding="utf-8")
        write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
        if debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "retrieval.jsonl").write_text("\n".join(debug_records), encoding="utf-8")
        logger.info("FiD RAG run complete: {} examples -> {}", len(answers), qa_path)
        return {
            "answers": str(answers_path),
            "qa": str(qa_path),
            "qa_no_header": str(qa_no_header_path),
            "qa_with_question": str(qa_q_path),
            "debug": str(debug_dir) if debug else None,
        }
