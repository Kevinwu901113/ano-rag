import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structrag.llm_client import LLMChatClient
from scripts.hotpotqa.baselines.baseline_utils import (
    build_passage_entries,
    detect_device,
    get_embedding_model,
    truncate_text,
)
from scripts.musique.baselines.musique_utils import load_dataset, save_musique_results_and_qa
from utils.context_budget import pack_contexts
from utils.jsonl_utils import write_jsonl
from utils.logging_utils import setup_logging
from utils.output_protocol import build_final_instruction
from utils.retrieval_logger import log_retrieval
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.run_metadata import build_basic_config, write_config_resolved


class MiniRaptor:
    def __init__(self, encoder: Callable[[List[str]], np.ndarray], llm: LLMChatClient):
        self.encoder = encoder
        self.llm = llm
        self.tree_nodes: List[Dict[str, Any]] = []
        self.leaf_entries: List[Dict[str, Any]] = []
        self.last_hits: List[Dict[str, Any]] = []

    def build_tree(self, passages: List[Dict[str, Any]]):
        self.leaf_entries = passages
        leaf_texts = [p["text"] for p in passages]
        if not leaf_texts:
            self.tree_nodes = []
            return
        if len(leaf_texts) < 3:
            self.tree_nodes = [
                {**p, "layer": 0, "children": [], "node_id": f"leaf_{i}"}
                for i, p in enumerate(passages)
            ]
            return

        vectors = self.encoder(leaf_texts)
        k = min(3, len(leaf_texts))
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(vectors)

        cluster_summaries: List[Dict[str, Any]] = []
        for c in range(k):
            idxs = [i for i, lab in enumerate(labels) if lab == c]
            cluster_passages = [leaf_texts[i] for i in idxs]
            cluster_text = "\n\n".join(cluster_passages)
            prompt = f"""Summarize the following passages into a concise note:

{cluster_text}

Summary:"""
            try:
                resp = self.llm.chat([{"role": "user", "content": prompt}])
                summary = truncate_text(resp.content, 800)
            except Exception:
                summary = truncate_text(cluster_text, 800)
            cluster_summaries.append(
                {
                    "text": summary,
                    "doc_id": f"cluster_{c}",
                    "sent_ids": [],
                    "passage_id": f"cluster_{c}",
                    "layer": 1,
                    "children": idxs,
                    "node_id": f"cluster_{c}",
                }
            )

        self.tree_nodes = [
            {**p, "layer": 0, "children": [], "node_id": f"leaf_{i}"}
            for i, p in enumerate(passages)
        ] + cluster_summaries

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.tree_nodes:
            return []
        texts = [n["text"] for n in self.tree_nodes]
        vectors = self.encoder(texts)
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norm + 1e-10)

        q_vec = self.encoder([query])
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(vectors, q_vec.T).flatten()
        indices = np.argsort(scores)[::-1][:k]
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices):
            node = self.tree_nodes[idx]
            hits.append(
                {
                    "text": node["text"],
                    "score": float(scores[idx]),
                    "doc_id": node.get("doc_id"),
                    "sent_ids": node.get("sent_ids"),
                    "passage_id": node.get("passage_id"),
                    "rank": rank + 1,
                }
            )
        self.last_hits = hits
        return hits


def process_example(
    item: Dict[str, Any],
    llm: LLMChatClient,
    encoder: Callable[[List[str]], np.ndarray],
    args,
    *,
    run_name: str,
    dataset_name: str,
    log_dir: Path,
) -> Tuple[str, str, str, List[str], List[Dict[str, Any]], int]:
    qid = str(item.get("id") or item.get("_id") or item.get("query_id") or "")
    question = str(item.get("question") or item.get("query") or "").strip()
    paragraphs = item.get("paragraphs") or item.get("contexts") or item.get("passages") or []
    passages = build_passage_entries(paragraphs, max_passages=args.max_context)

    raptor = MiniRaptor(encoder, llm)
    raptor.build_tree(passages)
    hits = raptor.retrieve(question, k=args.topk)

    context_str, contexts_used, context_tokens = pack_contexts(
        hits, int(getattr(args, "context_budget", 0) or 0)
    )
    prompt = f"""Answer the question using the provided context.
Keep the answer concise.
{build_final_instruction()}

{context_str}

Question: {question}
Answer:"""
    try:
        resp = llm.chat([{"role": "user", "content": prompt}])
        ans = resp.content
        try:
            log_retrieval(
                sample_id=qid,
                dataset=dataset_name,
                run_name=run_name,
                retrieved=hits,
                topk=len(hits),
                final_context=contexts_used,
                final_context_tokens=context_tokens,
                context_budget_tokens=int(getattr(args, "context_budget", 0) or 0) or None,
                log_dir=log_dir,
            )
        except Exception as log_exc:
            logger.error(f"retrieval logging failed for {qid}: {log_exc}")

        pred_evidence = [str(h.get("passage_id")) for h in hits if h.get("passage_id") and str(h.get("passage_id")).startswith("p")]
        seen = set()
        pred_evidence = [p for p in pred_evidence if not (p in seen or seen.add(p))]
        return qid, question, ans, pred_evidence, contexts_used, context_tokens
    except Exception as e:
        logger.error(f"Error processing {qid}: {e}")
        return qid, question, "error", [], contexts_used, context_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Raptor baseline on MuSiQue")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--qa-path", default=None)
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--lm-endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--lm-model", default="model-identifier")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for LLM decoding")
    parser.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--emb-device", default=None)
    parser.add_argument("--emb-dtype", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--context-budget", type=int, default=0, help="Max context tokens (0 disables)")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    logger.info(f"Loaded {len(data)} examples from {args.dataset}")

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="musique")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    run_name = work_dir.name
    dataset_name = "musique"
    output_path = Path(args.output) if args.output else preds_dir / "musique_results.jsonl"
    qa_path = Path(args.qa_path) if args.qa_path else preds_dir / "qa.tsv"
    logger.info(f"Writing outputs to workspace {work_dir}")
    setup_logging(str(work_dir / "run.log"))
    write_config_resolved(
        work_dir,
        build_basic_config(
            dataset="musique",
            model=args.lm_model,
            endpoint=args.lm_endpoint,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            context_budget=args.context_budget or None,
            topk=args.topk,
            extra={"max_context": args.max_context, "emb_model": args.emb_model},
        ),
    )

    llm = LLMChatClient(
        endpoint=args.lm_endpoint,
        model=args.lm_model,
        temperature=0.0,
        max_tokens=args.max_new_tokens if args.max_new_tokens is not None else 8192,
    )
    device = args.emb_device or detect_device()
    encoder = get_embedding_model(args.emb_model, device, torch_dtype=args.emb_dtype)
    logger.info(f"Embedding model {args.emb_model} on {device} (dtype={args.emb_dtype or 'auto'})")

    results: List[Dict[str, Any]] = []
    qa_rows: List[Tuple[str, str]] = []
    pred_raw_records: List[Dict[str, Any]] = []

    num_workers = max(1, args.num_workers)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                process_example,
                item,
                llm,
                encoder,
                args,
                run_name=run_name,
                dataset_name=dataset_name,
                log_dir=artifacts_dir,
            )
            for item in data
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            qid, question, ans, pred_evidence, contexts_used, context_tokens = fut.result()
            if not qid:
                continue
            results.append(
                {"id": qid, "predicted_answer": ans, "predicted_evidence": pred_evidence}
            )
            qa_rows.append((question, ans))
            pred_raw_records.append(
                {
                    "id": str(qid),
                    "question": question,
                    "pred_raw": ans,
                    "contexts_used": contexts_used,
                    "context_tokens_used": context_tokens,
                    "context_budget_tokens": int(args.context_budget or 0) or None,
                }
            )

    out_path, qa_file = save_musique_results_and_qa(
        work_dir,
        results,
        qa_rows,
        output_path=output_path,
        qa_path=qa_path,
    )
    write_jsonl(preds_dir / "pred_raw.jsonl", pred_raw_records)
    logger.info(f"Saved results to {out_path}")
    logger.info(f"Saved QA log to {qa_file}")


if __name__ == "__main__":
    main()
