#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hotpotqa.baselines.baseline_utils import TransformerEmbedder, build_passage_entries


@dataclass(frozen=True)
class Example:
    qid: str
    question: str
    gold_titles: Set[str]
    passage_titles: List[str]
    passage_texts: List[str]

from pathlib import Path
from config.config_loader import DEFAULT_EMBED_MODEL

def _default_embed_model() -> str:
    local = Path("/home/wjk/models/qwen3-emb")
    if local.exists():
        return str(local)
    return DEFAULT_EMBED_MODEL


def load_dataset(path: Path, *, max_context: int) -> List[Example]:
    data = json.loads(path.read_text(encoding="utf-8"))
    examples: List[Example] = []
    for item in data:
        qid = str(item.get("_id") or item.get("id") or "")
        if not qid:
            continue
        question = str(item.get("question") or "")
        supp = item.get("supporting_facts") or {}
        gold_titles = {str(t).strip() for t in (supp.get("title") or []) if str(t).strip()}
        entries = build_passage_entries(item.get("context"), max_passages=max_context)
        passage_titles = [str(e.get("doc_id") or "").strip() for e in entries]
        passage_texts = [str(e.get("text") or "") for e in entries]
        examples.append(
            Example(
                qid=qid,
                question=question,
                gold_titles=gold_titles,
                passage_titles=passage_titles,
                passage_texts=passage_texts,
            )
        )
    return examples


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / (norms + 1e-10)


def encode_in_chunks(
    embedder: TransformerEmbedder,
    texts: List[str],
    *,
    device: str,
    chunk_size: int,
    desc: str,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    chunks: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), chunk_size), desc=desc):
        batch = texts[i : i + chunk_size]
        emb = embedder.encode(batch, device=device)
        if emb is None or getattr(emb, "size", 0) == 0:
            continue
        chunks.append(np.asarray(emb))
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def rank_vanilla(passage_vecs: np.ndarray, query_vec: np.ndarray, *, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    vecs = l2_normalize(passage_vecs)
    q = l2_normalize(query_vec.reshape(1, -1)).reshape(-1)
    scores = vecs @ q
    idx = np.argsort(scores)[::-1][: max(1, int(topk))]
    return idx, scores[idx]


def rank_selfrag_retrieval_only(
    passage_vecs: np.ndarray, query_vec: np.ndarray, *, topk: int
) -> Tuple[np.ndarray, np.ndarray]:
    vecs = l2_normalize(passage_vecs)
    q = l2_normalize(query_vec.reshape(1, -1)).reshape(-1)
    scores = vecs @ q
    limit = max(1, int(topk))
    candidate_k = min(limit * 2, scores.shape[0])
    idx = np.argsort(scores)[::-1][:candidate_k][:limit]
    return idx, scores[idx]


def rank_raptor_retrieval_only(passage_vecs: np.ndarray, query_vec: np.ndarray, *, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    # MiniRAPTOR in retrieval-only mode builds no summaries and scores leaves by dot-product.
    # The embedder may already normalize; to match other baselines and keep cosine behavior stable,
    # we L2-normalize here (idempotent if already normalized).
    return rank_vanilla(passage_vecs, query_vec, topk=topk)


def rank_relrag(
    passage_vecs: np.ndarray,
    query_vec: np.ndarray,
    *,
    topk: int,
    threshold: float = 0.7,
    alpha: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    vecs = l2_normalize(passage_vecs)
    q = l2_normalize(query_vec.reshape(1, -1)).reshape(-1)

    sim = vecs @ vecs.T
    adj = (sim > float(threshold)).astype(np.float32)

    initial_scores = vecs @ q
    denom = (np.sum(adj, axis=1) + 1e-10).astype(np.float32)
    neighbor_scores = (adj @ initial_scores) / denom
    final_scores = float(alpha) * initial_scores + (1.0 - float(alpha)) * neighbor_scores

    idx = np.argsort(final_scores)[::-1][: max(1, int(topk))]
    return idx, final_scores[idx]


def evaluate_titles(
    examples: List[Example],
    query_vecs: np.ndarray,
    passage_vecs: np.ndarray,
    passage_offsets: List[Tuple[int, int]],
    *,
    ks: Sequence[int],
    topk: int,
) -> Dict[str, Dict[str, float]]:
    runners = {
        "hotpot_vanilla_rag_embedonly": rank_vanilla,
        "hotpot_selfrag_embedonly": rank_selfrag_retrieval_only,
        "hotpot_raptor_embedonly": rank_raptor_retrieval_only,
        "hotpot_relrag_embedonly": rank_relrag,
    }

    acc: Dict[str, Dict[str, float]] = {}
    for name in runners:
        metrics = {f"TitleRecall@{k}": 0.0 for k in ks}
        metrics.update({f"TitlePrec@{k}": 0.0 for k in ks})
        metrics.update({f"Hit@{k}": 0.0 for k in ks})
        acc[name] = metrics

    n = 0
    for row, ex in enumerate(tqdm(examples, desc="Scoring")):
        q = query_vecs[row]
        start, length = passage_offsets[row]
        p = passage_vecs[start : start + length]
        titles = ex.passage_titles
        if not titles or p.size == 0:
            continue
        gold = ex.gold_titles
        n += 1

        for run_name, rank_fn in runners.items():
            idx, _scores = rank_fn(p, q, topk=topk)
            ranked_titles = [titles[int(i)] for i in idx if 0 <= int(i) < len(titles)]

            for k in ks:
                k = int(k)
                retrieved_set = {t for t in ranked_titles[:k] if t}
                inter = len(retrieved_set & gold)
                if gold:
                    recall = inter / len(gold)
                else:
                    recall = 1.0 if not retrieved_set else 0.0
                if retrieved_set:
                    prec = inter / len(retrieved_set)
                else:
                    prec = 1.0 if not gold else 0.0
                hit = 1.0 if inter > 0 else (1.0 if not gold and not retrieved_set else 0.0)
                acc[run_name][f"TitleRecall@{k}"] += recall
                acc[run_name][f"TitlePrec@{k}"] += prec
                acc[run_name][f"Hit@{k}"] += hit

    if n == 0:
        return {}

    for run_name in acc:
        acc[run_name]["count"] = float(n)
        for key in list(acc[run_name].keys()):
            if key == "count":
                continue
            acc[run_name][key] = acc[run_name][key] / n
    return acc


def render_table(metrics: Dict[str, Dict[str, float]], *, ks: Sequence[int]) -> str:
    ks = [int(k) for k in ks]
    if set(ks) != {5, 10}:
        cols = [f"TitleRecall@{k}" for k in ks] + [f"TitlePrec@{k}" for k in ks] + [f"Hit@{k}" for k in ks]
        header = "| run | " + " | ".join(cols) + " | count |\n"
        header += "| --- | " + " | ".join(["---"] * (len(cols) + 1)) + " |\n"
        rows = []
        for run in sorted(metrics.keys()):
            m = metrics[run]
            rows.append(
                "| "
                + run
                + " | "
                + " | ".join(f"{m.get(c, 0.0):.3f}" for c in cols)
                + f" | {int(m.get('count', 0))} |"
            )
        return header + "\n".join(rows)

    header = (
        "| run | TitleRecall@5 | TitleRecall@10 | TitlePrec@5 | TitlePrec@10 | Hit@5 | Hit@10 | count |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for run in sorted(metrics.keys()):
        m = metrics[run]
        rows.append(
            f"| {run} | {m.get('TitleRecall@5', 0):.3f} | {m.get('TitleRecall@10', 0):.3f} | "
            f"{m.get('TitlePrec@5', 0):.3f} | {m.get('TitlePrec@10', 0):.3f} | "
            f"{m.get('Hit@5', 0):.3f} | {m.get('Hit@10', 0):.3f} | {int(m.get('count', 0))} |"
        )
    return header + "\n" + "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedding-only HotpotQA baselines on distractor setting.")
    parser.add_argument("--dataset", default="data/hotpotqa/dataset_distractor_200.json", help="HotpotQA dataset json")
    parser.add_argument("--max-context", type=int, default=10, help="Max paragraphs per question")
    parser.add_argument("--topk", type=int, default=10, help="Top-k passages to rank (>= max ks)")
    parser.add_argument("--ks", default="5,10", help="Comma-separated ks for recall/prec/hit (e.g., 1,3,5,10)")

    parser.add_argument("--embed-model", default=_default_embed_model(), help="Embedding model name or local path")
    parser.add_argument("--embed-device", choices=["cpu", "cuda", "auto"], default="cpu", help="Embedding device")
    parser.add_argument("--embed-batch-size", type=int, default=2, help="Embedding batch size")
    parser.add_argument("--embed-max-length", type=int, default=256, help="Embedding max sequence length")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--embed-normalize", dest="embed_normalize", action="store_true", help="L2-normalize embeddings")
    norm.add_argument("--no-embed-normalize", dest="embed_normalize", action="store_false", help="Disable L2-normalization")
    parser.set_defaults(embed_normalize=True)
    parser.add_argument("--emb-dtype", default=None, help="Embedding torch dtype (float16, bfloat16, float32)")
    parser.add_argument("--encode-chunk-size", type=int, default=200, help="Encode texts in chunks for progress")

    parser.add_argument("--output-json", default=None, help="Write metrics json to this path")
    parser.add_argument("--output-md", default=None, help="Write markdown table to this path")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ks = [int(x) for x in str(args.ks).split(",") if str(x).strip()]
    ks = sorted({k for k in ks if k > 0})
    if not ks:
        raise ValueError("--ks must contain positive integers, e.g. 5,10")
    topk = int(args.topk)
    if topk < max(ks):
        raise ValueError(f"--topk must be >= max(--ks). Got topk={topk}, ks={ks}")

    device = str(args.embed_device)
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    examples = load_dataset(dataset_path, max_context=int(args.max_context))
    if not examples:
        raise RuntimeError("No examples loaded from dataset")

    embedder = TransformerEmbedder(
        args.embed_model,
        torch_dtype=args.emb_dtype,
        batch_size=int(args.embed_batch_size),
        max_length=int(args.embed_max_length),
        normalize=bool(args.embed_normalize),
    )

    passage_texts: List[str] = []
    passage_offsets: List[Tuple[int, int]] = []
    for ex in examples:
        start = len(passage_texts)
        passage_texts.extend(ex.passage_texts)
        passage_offsets.append((start, len(ex.passage_texts)))

    query_texts = [ex.question for ex in examples]

    passage_vecs = encode_in_chunks(
        embedder,
        passage_texts,
        device=device,
        chunk_size=int(args.encode_chunk_size),
        desc="Encoding passages",
    )
    query_vecs = encode_in_chunks(
        embedder,
        query_texts,
        device=device,
        chunk_size=int(args.encode_chunk_size),
        desc="Encoding questions",
    )

    if query_vecs.shape[0] != len(examples):
        raise RuntimeError(f"Expected {len(examples)} query vectors, got {query_vecs.shape[0]}")

    metrics = evaluate_titles(
        examples,
        query_vecs,
        passage_vecs,
        passage_offsets,
        ks=ks,
        topk=topk,
    )

    payload = {
        "dataset": str(dataset_path),
        "count": len(examples),
        "embed_model": str(args.embed_model),
        "embed_device": device,
        "embed_batch_size": int(args.embed_batch_size),
        "embed_max_length": int(args.embed_max_length),
        "embed_normalize": bool(args.embed_normalize),
        "ks": ks,
        "topk": topk,
        "runs": metrics,
    }

    table = render_table(metrics, ks=ks)
    print(table)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Avoid tokenizers warning noise in CLI runs.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

