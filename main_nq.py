#!/usr/bin/env python3
"""Natural Questions (NQ) 批量预测入口（逐行独立 + 并行 + 稳健兜底）。"""

from __future__ import annotations
import argparse, json, logging, os, re
from typing import Dict, Any, List

from retrieval.quick_hybrid_test import HybridRetriever
from answer.efsa_answer import EfsaAnswer
from answer.verify_shell import Verifier
from answer.nq_answer import decide_nq_answer
from utils.nq_normalize import normalize_text
from parallel.parallel_interface import ParallelProcessor  # 若不想用并行，可注释并改为串行

logger = logging.getLogger("main_nq")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------- 小工具：兜底排序/短答案 ----------
def _normalize_space(s:str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _fallback_rank_by_overlap(question: str, paragraphs: list, topk: int = 20):
    # 与 retriever 的实现一致，确保即使 retriever 异常也能给出排序
    from retrieval.quick_hybrid_test import _tok
    qset = set(_tok(question))
    ranked = []
    for p in paragraphs:
        idx = p.get("idx")
        text = p.get("paragraph_text") or p.get("text") or ""
        tset = set(_tok(text))
        score = len(qset & tset) / (len(qset | tset) or 1)
        if idx is not None:
            ranked.append({"idx": int(idx), "text": text, "score": float(score)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:topk]

def _fallback_short_answer(question: str, support_texts: List[str]) -> str:
    # 极简兜底：从支持段落里截一个小窗口，避免全空
    tokens = re.findall(r"\w+", support_texts[0] if support_texts else "")
    return normalize_text(" ".join(tokens[:6])) if tokens else ""

# ---------- 核心处理：单条样本 ----------
def process_item(item: Dict[str, Any],
                 retriever: HybridRetriever,
                 efsa: EfsaAnswer,
                 verifier: Verifier,
                 topk: int = 20,
                 topk_support: int = 3) -> Dict[str, Any]:
    qid = str(item["id"])
    question = item["question"]
    paragraphs = item.get("paragraphs", [])

    # 1) 检索（先调用 retriever；若空/异常则兜底）
    try:
        ranked = retriever.rank(question, paragraphs) or []
        # 校验结构
        ok = ranked and all(isinstance(r, dict) and ("idx" in r) and ("text" in r) for r in ranked)
        if not ok:
            ranked = _fallback_rank_by_overlap(question, paragraphs, topk=topk)
    except Exception:
        ranked = _fallback_rank_by_overlap(question, paragraphs, topk=topk)

    retrieved_doc_ids = [f"{qid}_{int(r['idx'])}" for r in ranked[:topk]]
    support = ranked[: min(topk_support, len(ranked))]
    predicted_support_idxs = [int(x["idx"]) for x in support]
    support_texts = [_normalize_space(x["text"]) for x in support]

    # 2) EFSA/span 候选（失败或空则兜底一个短片段）
    try:
        efsa_candidate, conf_scores, support_sents = efsa.produce(
            question=question,
            contexts=support_texts
        )
    except Exception:
        efsa_candidate, conf_scores, support_sents = (
            "",
            {"answer_conf": 0.0, "support_conf": 0.0, "coverage": 0.0, "entailment": 0.0},
            []
        )

    if not efsa_candidate:
        efsa_candidate = _fallback_short_answer(question, support_texts)

    # 3) NQ 答案决策（yes/no / 无答案 / 文本）
    predicted_answer, predicted_answerable = decide_nq_answer(
        example=item,
        efsa_candidate=efsa_candidate,
        support_sents=support_sents or support_texts,
        conf_scores=conf_scores,
        verifier=verifier
    )

    # 再兜底：确保不是全空（仅在确实拿不到答案时触发）
    if (not predicted_answer) and efsa_candidate:
        predicted_answer = normalize_text(efsa_candidate)
        predicted_answerable = True

    return {
        "id": qid,
        "predicted_answer": predicted_answer or "",
        "predicted_answerable": bool(predicted_answerable),
        "predicted_support_idxs": predicted_support_idxs,
        "retrieved_doc_ids": retrieved_doc_ids
    }

# ---------- I/O ----------
def _load_dataset(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="data/nq_dev/musique_style_dev.jsonl")
    ap.add_argument("output_file", help="result/nq_dev/nq_results.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=20, help="检索候选写入 retrieved_doc_ids 的上限")
    ap.add_argument("--topk_support", type=int, default=3, help="用于生成/验证的支持段落数")
    args = ap.parse_args()

    logger.info("[INFO] loading dataset from %s", args.input_file)
    data = _load_dataset(args.input_file)
    logger.info("[INFO] loaded %d examples", len(data))
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 初始化组件（可替换成你真实实现）
    retriever = HybridRetriever()
    efsa = EfsaAnswer()
    verifier = Verifier()

    # 逐行独立 + 并行
    proc = ParallelProcessor(max_workers=args.workers)
    n_total = len(data)
    n_have_retrieval = 0
    n_have_support = 0
    n_nonempty_answer = 0

    def _task(item):
        res = process_item(item, retriever, efsa, verifier, topk=args.topk, topk_support=args.topk_support)
        return res

    results: List[Dict[str, Any]] = []
    for res in proc.map(_task, data):
        # 统计
        if res.get("retrieved_doc_ids"):
            n_have_retrieval += 1
        if res.get("predicted_support_idxs"):
            n_have_support += 1
        if res.get("predicted_answer"):
            n_nonempty_answer += 1
        results.append(res)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("[STAT] total=%d have_retrieval=%d have_support=%d nonempty_answer=%d",
                n_total, n_have_retrieval, n_have_support, n_nonempty_answer)
    logger.info("[OK] wrote %s ; total=%d", args.output_file, len(results))

if __name__ == "__main__":
    main()
