#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuSiQue end2end (v2) runner
- 输入：MuSiQue 测试集 jsonl （每行一个任务）
- 对每个任务： 原子笔记/构图（仅 paragraphs）→ 检索（仅 question）→ 生成答案（RAG）
- 输出目录（工作目录）包含四个文件：
  1) musique_result.jsonl     # 官方提交格式：{id, answer, (可选)support_idxs}
  2) log                      # JSONL：每条任务的运行日志（用时、候选、后端等）
  3) error_log                # JSONL：失败任务的错误信息
  4) musique_recall.jsonl     # JSONL：自查检索结果（召回列表与最终选择）
- 断点重续：默认使用 result/ 下最新 run 目录，跳过已完成 id；加 --new 则新建 run 目录
"""

import argparse, json, os, sys, time, glob, math, yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- v2 构图/检索 ---
from atomic_v2.dbtes import dbtes_select_edges
from retrieval_v2.candidate_gate import build_candidates_for_note
from retrieval_v2.mmr import mmr_select
from graph_v2.struct_prior import build_weak_graph

# --- LLM 路由（LM Studio + Ollama）---
from llm.router import build_router_from_config
from llm.lmstudio_client import LMStudioClient
try:
    from llm.ollama_client import OllamaClient
except Exception:
    OllamaClient = None

# ========== 工具 ==========
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def append_jsonl(path: Path, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_done_ids(result_file: Path) -> set:
    done = set()
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done

def latest_run_dir(base: Path) -> Path:
    runs = sorted([p for p in base.iterdir() if p.is_dir()],
                  key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None

def ensure_run_dir(base: Path, new_flag: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if new_flag:
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        out = base / ts
        out.mkdir()
        return out
    # 无 --new：使用最新；若不存在则新建
    last = latest_run_dir(base)
    if last is None:
        ts = time.strftime("run_%Y%m%d_%H%M%S")
        out = base / ts
        out.mkdir()
        return out
    return last

# ========== MuSiQue 结构处理 ==========
def notes_from_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 paragraphs 转为 notes（仅用于构图阶段）"""
    out = []
    for p in task.get("paragraphs", []):
        out.append({
            "id": f"{task['id']}__{p['idx']}",
            "title": p.get("title", ""),
            "text": p.get("paragraph_text", "")
        })
    return out

def build_graph_for_task(notes: List[Dict[str, Any]],
                         cfg: Dict[str, Any],
                         router) -> List[Dict[str, Any]]:
    """Gate → DBTES → (MMR) 生成边；仅使用 paragraphs 内容"""
    # 弱图（结构先验）
    paras_like = {
        n["id"]: {"idx": n["id"], "title": n.get("title",""), "paragraph_text": n.get("text","")}
        for n in notes
    }
    weak_graph = build_weak_graph(paras_like)

    edges = []
    gate_cfg  = cfg["retrieval_v2"]["gate"]
    dbtes_cfg = cfg["retrieval_v2"]["dbtes"]
    mmr_cfg   = cfg.get("mmr", {})

    id2title = {n["id"]: n.get("title","") for n in notes}

    for n in notes:
        # 候选（结构/局部文本相似）
        cands = build_candidates_for_note(n["id"], notes, weak_graph, gate_cfg)
        slim  = [{"id": c["id"], "score": c["score"]} for c in cands]

        # DBTES 选择（需要完整 notes 提供 title/text）
        sel = dbtes_select_edges(center_note=n, candidates=slim,
                                 llm_call=router.call, config=dbtes_cfg, notes=notes)

        # MMR 去冗（无向量 → 用标题字符集 Jaccard 兜底；没有标题则跳过）
        keep_k = mmr_cfg.get("keep_k", dbtes_cfg.get("keep_k", 3))
        if keep_k and keep_k < len(sel) and id2title:
            def _tj(a: str, b: str) -> float:
                A, B = set((a or "").lower()), set((b or "").lower())
                return (len(A & B) / max(1, len(A | B)))
            # 构造 items 以便 mmr_select 使用 sim(x,y)
            items = [{"id": s["id"], "score": s.get("score", 0.9), "title": id2title.get(s["id"], "")} for s in sel]
            chosen = mmr_select(items=items, k=keep_k,
                                sim=lambda x,y: _tj(x.get("title",""), y.get("title","")))
            sel_ids = {s["id"] for s in sel}
            keep = [s for s in sel if s["id"] in {c["id"] for c in chosen}]
        else:
            keep = sel[:keep_k]

        cand_map = {c["id"]: c["score"] for c in slim}
        for e in keep:
            edges.append({
                "src": n["id"],
                "dst": e["id"],
                "weight": float(cand_map.get(e["id"], e.get("score", 0.9))),
                "evidence": {"reason": e.get("reason", "dbtes")}
            })
    return edges

# ========== 检索（仅 question，不透题） ==========
def _idf(corpus: List[List[str]]) -> Dict[str, float]:
    df = {}
    N  = len(corpus)
    for doc in corpus:
        for tok in set(doc):
            df[tok] = df.get(tok, 0) + 1
    import math
    return {t: math.log((N + 1) / (df_t + 0.5)) + 1.0 for t, df_t in df.items()}

def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]

def retrieve_for_question(question: str,
                          notes: List[Dict[str, Any]],
                          edges: List[Dict[str, Any]],
                          topk: int = 5) -> Tuple[List[Tuple[str, float]], List[str]]:
    """
    返回：(recall_list, support_note_ids)
      - recall_list: [(note_id, score), ...]  根据 question 与 note.text 的 BM25-like 打分
      - support_note_ids: 最终用于生成的 note id（取前 topk）
    """
    # 构建索引（当前任务的 notes）
    texts = [n["text"] for n in notes]
    toks  = [_tokenize(t) for t in texts]
    idf   = _idf(toks)
    qtok  = _tokenize(question)

    def bm25(q: List[str], d: List[str], k1=1.5, b=0.75) -> float:
        if not d: return 0.0
        import collections
        tf = collections.Counter(d)
        dl = len(d)
        avgdl = sum(len(x) for x in toks) / max(1, len(toks))
        score = 0.0
        for w in set(q):
            if w not in idf: continue
            tfw = tf.get(w, 0)
            denom = tfw + k1 * (1 - b + b * dl / max(1e-9, avgdl))
            score += idf[w] * (tfw * (k1 + 1) / max(1e-9, denom))
        return score

    scored = []
    for n in notes:
        s = bm25(qtok, _tokenize(n["text"]))
        scored.append((n["id"], s))
    scored.sort(key=lambda x: -x[1])

    support_note_ids = [nid for nid, _ in scored[:topk]]
    return scored, support_note_ids

# ========== 生成答案（RAG） ==========
def answer_by_rag(question: str,
                  support_note_ids: List[str],
                  id2note: Dict[str, Dict[str, Any]],
                  router) -> str:
    ctx = " ".join([id2note[nid]["text"] for nid in support_note_ids if nid in id2note])
    prompt = f"Question: {question}\nContext:\n{ctx}\nAnswer (concise):"
    resp = router.call(prompt)
    return (resp or "").strip()

# ========== 单条任务处理 ==========
def process_one_task(task: Dict[str, Any],
                     cfg: Dict[str, Any],
                     router,
                     recall_topk: int = 5) -> Dict[str, Any]:
    t0 = time.time()
    # 仅 paragraphs → 构图
    notes = notes_from_task(task)
    edges = build_graph_for_task(notes, cfg, router)

    # 仅 question → 检索
    recall_list, support_ids = retrieve_for_question(task["question"], notes, edges, topk=recall_topk)

    # 生成答案
    id2note = {n["id"]: n for n in notes}
    answer  = answer_by_rag(task["question"], support_ids, id2note, router)

    # 结果与日志
    used_idx = []
    for nid in support_ids:
        # 从 note_id 还原 paragraphs 的 idx
        # 形如 "<task_id>__<idx>"
        try:
            used_idx.append(int(nid.split("__")[-1]))
        except Exception:
            pass

    log = {
        "id": task["id"],
        "elapsed_ms": int(1000 * (time.time() - t0)),
        "n_notes": len(notes),
        "n_edges": len(edges),
        "recall_top": recall_list[:min(20, len(recall_list))],
        "support_note_ids": support_ids
    }
    result_record = {
        "id": task["id"],
        "answer": answer,
        "support_idxs": used_idx  # 官方可选字段；仅 {id, answer} 也可提交
    }
    recall_record = {"id": task["id"], "recall": recall_list, "chosen": support_ids}
    return result_record, log, recall_record

# ========== 主入口 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="MuSiQue jsonl (test)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--result", default="result", help="结果根目录")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--retrieval-topk", type=int, default=5)
    ap.add_argument("--new", action="store_true", help="新建工作目录；否则用最新目录并断点续跑")
    args = ap.parse_args()

    # 配置与 LLM 路由
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    lm = LMStudioClient()
    def lm_call(p: str) -> str:
        return (lm.chat([{"role": "user", "content": p}]) or "").strip()
    oll = None
    if OllamaClient:
        try:
            oll = OllamaClient()
        except Exception:
            oll = None
    router = build_router_from_config(cfg, lm_call, (oll.generate if oll else None))

    # 输出目录（工作目录）
    base = Path(args.result)
    run_dir = ensure_run_dir(base, args.new)
    res_file   = run_dir / "musique_result.jsonl"
    log_file   = run_dir / "log"
    err_file   = run_dir / "error_log"
    recall_file= run_dir / "musique_recall.jsonl"

    # 断点续跑：已完成集合
    done = load_done_ids(res_file)

    # 读输入
    samples = read_jsonl(Path(args.input))
    todo = [s for s in samples if s.get("id") not in done]

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] total={len(samples)}, done={len(done)}, todo={len(todo)}")

    # 并行执行（跑多少写多少）
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process_one_task, t, cfg, router, args.retrieval_topk): t for t in todo}
        for fu in as_completed(futs):
            task = futs[fu]
            try:
                result_record, log_record, recall_record = fu.result()
                append_jsonl(res_file,   result_record)
                append_jsonl(log_file,   log_record)
                append_jsonl(recall_file,recall_record)
            except Exception as e:
                append_jsonl(err_file, {"id": task.get("id"), "error": str(e)})

    print(f"[DONE] wrote -> {res_file} , {log_file} , {err_file} , {recall_file}")

if __name__ == "__main__":
    main()
