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

import argparse, json, os, sys, time, glob, math, yaml, hashlib, re
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence
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

from query.evaluator import RetrievalEvaluator
from utils.logging import StructuredLogger
from vector.encoder import encode_notes, encode_query
from vector.indexer import VectorIndexer

# ========== 工具 ==========
def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)

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

def validate_input_path(input_path: str) -> Path:
    """验证输入路径是否在 ./data 目录下，防止路径越界"""
    path = Path(input_path).resolve()
    data_dir = Path("./data").resolve()
    
    try:
        path.relative_to(data_dir)
    except ValueError:
        raise ValueError(f"输入路径必须位于 ./data 目录下，当前路径: {input_path}")
    
    return path

def collect_jsonl_files(input_path: Path) -> List[Path]:
    """收集 .jsonl 文件，支持单文件或目录递归"""
    if input_path.is_file():
        if input_path.suffix != '.jsonl':
            raise ValueError(f"文件必须是 .jsonl 格式: {input_path}")
        return [input_path]
    elif input_path.is_dir():
        jsonl_files = list(input_path.rglob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"目录中未找到 .jsonl 文件: {input_path}")
        return sorted(jsonl_files)
    else:
        raise ValueError(f"输入路径不存在: {input_path}")

def load_and_merge_samples(jsonl_files: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """加载并合并多个 .jsonl 文件，返回样本列表和清单信息"""
    all_samples = []
    manifest = {
        "files": [],
        "total_samples": 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for file_path in jsonl_files:
        samples = read_jsonl(file_path)
        all_samples.extend(samples)
        
        manifest["files"].append({
            "path": str(file_path),
            "sample_count": len(samples)
        })
    
    manifest["total_samples"] = len(all_samples)
    return all_samples, manifest

def resolve_input_path(args_input: str, config: Dict[str, Any]) -> str:
    """解析输入路径，优先级：--input → config.yaml"""
    if args_input:
        return args_input
    
    config_path = config.get("input", {}).get("path")
    if not config_path:
        raise ValueError("未指定输入路径，请使用 --input 参数或在 config.yaml 中配置 input.path")
    
    return config_path

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

# ========== 向量检索辅助 ==========
def _safe_index_name(identifier: Any) -> str:
    text = str(identifier or "default")
    safe = re.sub(r"[^0-9a-zA-Z._-]", "_", text)
    return safe or "default"


def _resolve_index_path(base_path: Optional[str], task_id: Any) -> Optional[Path]:
    if not base_path:
        return None
    base = Path(base_path)
    if base.suffix:
        return base
    return base / f"{_safe_index_name(task_id)}.faiss"


def _notes_fingerprint(notes: Sequence[Dict[str, Any]]) -> str:
    hasher = hashlib.sha1()
    for note in notes:
        hasher.update(str(note.get("id", "")).encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update((note.get("title", "") or "").encode("utf-8"))
        hasher.update(b"\x1f")
        hasher.update((note.get("text", "") or "").encode("utf-8"))
        hasher.update(b"\x1e")
    return hasher.hexdigest()


def _latency_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _summarize_latencies(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    values = [float(v) for v in latencies if v is not None]
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    mean_ms = sum(values) / len(values)
    return {
        "count": len(values),
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(_latency_percentile(values, 50), 3),
        "p95_ms": round(_latency_percentile(values, 95), 3),
    }

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

def _bm25_retrieve(question: str,
                   notes: List[Dict[str, Any]],
                   topk: int) -> Tuple[List[Tuple[str, float]], List[str]]:
    texts = [n.get("text", "") for n in notes]
    toks = [_tokenize(t) for t in texts]
    idf = _idf(toks)
    qtok = _tokenize(question)

    def bm25(q: List[str], d: List[str], k1=1.5, b=0.75) -> float:
        if not d:
            return 0.0
        import collections

        tf = collections.Counter(d)
        dl = len(d)
        avgdl = sum(len(x) for x in toks) / max(1, len(toks))
        score = 0.0
        for w in set(q):
            if w not in idf:
                continue
            tfw = tf.get(w, 0)
            denom = tfw + k1 * (1 - b + b * dl / max(1e-9, avgdl))
            score += idf[w] * (tfw * (k1 + 1) / max(1e-9, denom))
        return score

    scored: List[Tuple[str, float]] = []
    for n in notes:
        note_id = n.get("id")
        if note_id is None:
            continue
        score = bm25(qtok, _tokenize(n.get("text", "")))
        scored.append((note_id, score))
    scored.sort(key=lambda x: -x[1])
    support_note_ids = [nid for nid, _ in scored[:topk]]
    return scored, support_note_ids


def retrieve_for_question(question: str,
                          notes: List[Dict[str, Any]],
                          edges: List[Dict[str, Any]],
                          topk: int,
                          config: Dict[str, Any],
                          task_id: Any,
                          logger: Optional[StructuredLogger] = None
                          ) -> Tuple[List[Tuple[str, float]], List[str], str]:
    ann_cfg = (config.get("ann") or {})
    embedding_cfg = (config.get("embedding") or {})
    use_vector = bool(embedding_cfg) and _as_bool(ann_cfg.get("enabled", False), False)
    vector_results: Optional[List[Tuple[str, float]]] = None
    vector_support: List[str] = []

    if use_vector:
        model_name = embedding_cfg.get("model")
        dimension = embedding_cfg.get("dimension")
        if not model_name or not dimension:
            if logger:
                logger.warning(
                    "vector_config_incomplete",
                    model=model_name,
                    dimension=dimension,
                )
        else:
            normalize = _as_bool(embedding_cfg.get("normalize", True), True)
            batch_size = int(embedding_cfg.get("batch_size", 32) or 32)
            note_instruction = embedding_cfg.get("note_instruction") or embedding_cfg.get("instruction")
            query_instruction = embedding_cfg.get("query_instruction") or embedding_cfg.get("instruction")
            index_path = _resolve_index_path(ann_cfg.get("index_path"), task_id)
            ann_params = ann_cfg.get("params") or {}
            topk_raw = int(ann_cfg.get("topk_raw", max(topk, 50)) or max(topk, 50))
            topk_raw = max(topk_raw, topk)
            fingerprint = _notes_fingerprint(notes)
            indexer = VectorIndexer(
                int(dimension),
                metric=ann_cfg.get("metric", "ip"),
                index_path=index_path,
                ann_params=ann_params,
            )

            loaded = False
            if index_path is not None:
                try:
                    loaded = indexer.load(expected_version=fingerprint)
                except Exception as exc:  # pragma: no cover - defensive logging
                    loaded = False
                    if logger:
                        logger.warning(
                            "vector_index_load_failed",
                            error=str(exc),
                            index_path=str(index_path),
                        )
            if not loaded:
                try:
                    note_vectors = encode_notes(
                        notes,
                        model_name=model_name,
                        normalize=normalize,
                        batch_size=batch_size,
                        instruction=note_instruction,
                    )
                except Exception as exc:  # pragma: no cover - encoding failure
                    if logger:
                        logger.error("vector_note_encode_failed", error=str(exc))
                else:
                    if note_vectors.shape[1] != int(dimension):
                        if logger:
                            logger.error(
                                "vector_dimension_mismatch",
                                expected=int(dimension),
                                actual=int(note_vectors.shape[1]),
                            )
                    else:
                        try:
                            ids = []
                            for idx, note in enumerate(notes):
                                note_id = note.get("id")
                                ids.append(str(note_id) if note_id is not None else f"note_{idx}")
                            indexer.build(
                                note_vectors,
                                ids,
                                note_version=fingerprint,
                            )
                            if index_path is not None:
                                try:
                                    indexer.persist()
                                except Exception as exc:  # pragma: no cover
                                    if logger:
                                        logger.warning(
                                            "vector_index_persist_failed",
                                            error=str(exc),
                                            index_path=str(index_path),
                                        )
                            if logger:
                                logger.debug(
                                    "vector_index_built",
                                    count=len(notes),
                                    index_path=str(index_path) if index_path else None,
                                )
                        except Exception as exc:  # pragma: no cover - build failure
                            if logger:
                                logger.error("vector_index_build_failed", error=str(exc))
            else:
                if logger:
                    logger.debug(
                        "vector_index_loaded",
                        index_path=str(index_path),
                        count=len(notes),
                    )

            if indexer.index is not None and len(indexer) > 0:
                try:
                    query_vec = encode_query(
                        question,
                        model_name=model_name,
                        normalize=normalize,
                        instruction=query_instruction,
                    )
                    vector_results = indexer.query(query_vec, topk_raw)
                    vector_support = [nid for nid, _ in vector_results[:topk]]
                except Exception as exc:  # pragma: no cover - query failure
                    vector_results = None
                    vector_support = []
                    if logger:
                        logger.error("vector_query_failed", error=str(exc))
            elif logger and use_vector:
                logger.warning("vector_index_unavailable", index_path=str(index_path))

    if vector_results:
        return vector_results, vector_support, "vector"

    # fallback to BM25
    bm25_results, bm25_support = _bm25_retrieve(question, notes, topk)
    return bm25_results, bm25_support, "bm25"

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
                     recall_topk: int = 5,
                     structured_logger: Optional[StructuredLogger] = None,
                     candidate_log_limit: int = 50,
                     retrieval_latency_collector: Optional[List[float]] = None
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    # 仅 paragraphs → 构图
    notes = notes_from_task(task)
    edges = build_graph_for_task(notes, cfg, router)

    # 仅 question → 检索
    retrieval_cm = structured_logger.time_stage(
        "retrieval", query_id=task.get("id"), query=task.get("question", "")
    ) if structured_logger else nullcontext({"elapsed_ms": None})
    with retrieval_cm as retrieval_ctx:
        recall_list, retrieved_support_ids, retrieval_mode = retrieve_for_question(
            task["question"],
            notes,
            edges,
            topk=recall_topk,
            config=cfg,
            task_id=task.get("id"),
            logger=structured_logger,
        )
    retrieval_latency = retrieval_ctx.get("elapsed_ms") if isinstance(retrieval_ctx, dict) else None
    if retrieval_latency is not None and retrieval_latency_collector is not None:
        retrieval_latency_collector.append(float(retrieval_latency))
    if structured_logger:
        structured_logger.info(
            "retrieval_mode",
            query_id=task.get("id"),
            mode=retrieval_mode,
        )
        structured_logger.log_candidates(
            stage="retrieval_after",
            query_id=task.get("id"),
            candidates=recall_list,
            latency_ms=retrieval_latency,
            limit=candidate_log_limit,
            query=task.get("question", ""),
        )

    rerank_cm = structured_logger.time_stage(
        "rerank", query_id=task.get("id"), query=task.get("question", "")
    ) if structured_logger else nullcontext({"elapsed_ms": None})
    with rerank_cm as rerank_ctx:
        support_ids = (retrieved_support_ids or [])[:recall_topk]
    rerank_latency = rerank_ctx.get("elapsed_ms") if isinstance(rerank_ctx, dict) else None
    if structured_logger:
        score_map = {nid: score for nid, score in recall_list}
        rerank_candidates = [
            {"id": nid, "score": score_map.get(nid)} for nid in support_ids
        ]
        structured_logger.log_candidates(
            stage="rerank_after",
            query_id=task.get("id"),
            candidates=rerank_candidates,
            latency_ms=rerank_latency,
            limit=len(rerank_candidates) or recall_topk,
            query=task.get("question", ""),
        )

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
        "support_note_ids": support_ids,
        "retrieval_latency_ms": retrieval_latency,
        "rerank_latency_ms": rerank_latency,
        "retrieval_mode": retrieval_mode,
    }
    result_record = {
        "id": task["id"],
        "answer": answer,
        "support_idxs": used_idx  # 官方可选字段；仅 {id, answer} 也可提交
    }
    recall_record = {
        "id": task["id"],
        "recall": recall_list,
        "chosen": support_ids,
        "mode": retrieval_mode,
    }
    evaluation_payload = {
        "query_id": task.get("id"),
        "question": task.get("question", ""),
        "retrieval_candidates": recall_list,
        "predicted_answer": answer,
        "predicted_support_idxs": used_idx,
        "reference": task,
    }
    return result_record, log, recall_record, evaluation_payload

# ========== 主入口 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="MuSiQue jsonl (test) 或目录路径")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--result", default="result", help="结果根目录")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--retrieval-topk", type=int, default=5)
    ap.add_argument("--new", action="store_true", help="新建工作目录；否则用最新目录并断点续跑")
    args = ap.parse_args()

    # 配置与 LLM 路由
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    logging_cfg = cfg.get("logging", {}) or {}
    log_level = str(logging_cfg.get("level", "INFO"))
    log_json = _as_bool(logging_cfg.get("json", True), True)
    logging_enabled = log_level.upper() != "OFF"
    structured_logger = StructuredLogger(
        level=log_level,
        json_lines=log_json,
        enabled=logging_enabled,
    )
    metrics_cfg = cfg.get("metrics", {}) or {}
    raw_topk = metrics_cfg.get("topk_eval", 50)
    try:
        metrics_topk = int(raw_topk)
    except (TypeError, ValueError):
        metrics_topk = 50
    metrics_topk = max(1, metrics_topk)
    metrics_enabled = _as_bool(metrics_cfg.get("enable", False), False)
    evaluator = RetrievalEvaluator(topk_eval=metrics_topk) if metrics_enabled else None
    candidate_log_limit = max(metrics_topk, args.retrieval_topk)
    structured_logger.info(
        "config_loaded",
        log_level=log_level,
        logging_json=log_json,
        logging_enabled=logging_enabled,
        metrics_enabled=metrics_enabled,
        metrics_topk=metrics_topk,
    )
    retrieval_latencies: List[float] = []
    
    # 解析输入路径（优先级：--input → config.yaml）
    input_path_str = resolve_input_path(args.input, cfg)
    input_path = validate_input_path(input_path_str)
    
    # 收集 .jsonl 文件
    jsonl_files = collect_jsonl_files(input_path)
    
    # 加载并合并样本
    samples, manifest = load_and_merge_samples(jsonl_files)
    
    # 如果是多文件，写出清单
    if len(jsonl_files) > 1:
        manifest_path = Path("input_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 多文件输入，清单已写入: {manifest_path}")
    
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
    todo = [s for s in samples if s.get("id") not in done]

    print(f"[INFO] input_path={input_path}")
    print(f"[INFO] jsonl_files={len(jsonl_files)}")
    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] total={len(samples)}, done={len(done)}, todo={len(todo)}")
    structured_logger.info(
        "runner_start",
        input_path=str(input_path),
        total=len(samples),
        done=len(done),
        todo=len(todo),
        run_dir=str(run_dir),
    )

    # 并行执行（跑多少写多少）
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {
            ex.submit(
                process_one_task,
                t,
                cfg,
                router,
                args.retrieval_topk,
                structured_logger,
                candidate_log_limit,
                retrieval_latencies,
            ): t for t in todo
        }
        for fu in as_completed(futs):
            task = futs[fu]
            try:
                result_record, log_record, recall_record, eval_payload = fu.result()
                append_jsonl(res_file,   result_record)
                append_jsonl(log_file,   log_record)
                append_jsonl(recall_file,recall_record)
                if evaluator is not None and eval_payload:
                    evaluator.add_record(**eval_payload)
            except Exception as e:
                append_jsonl(err_file, {"id": task.get("id"), "error": str(e)})
                structured_logger.error(
                    "task_failed",
                    query_id=task.get("id"),
                    error=str(e),
                )

    if retrieval_latencies:
        stats = _summarize_latencies(retrieval_latencies)
        structured_logger.info("retrieval_latency_stats", **stats)
        print(
            "[STATS] retrieval latency ms "
            f"mean={stats['mean_ms']}, p50={stats['p50_ms']}, "
            f"p95={stats['p95_ms']} (n={stats['count']})"
        )

    print(f"[DONE] wrote -> {res_file} , {log_file} , {err_file} , {recall_file}")
    structured_logger.info(
        "runner_finished",
        result_file=str(res_file),
        log_file=str(log_file),
        error_file=str(err_file),
        recall_file=str(recall_file),
    )
    if evaluator is not None:
        report_path, report_payload = evaluator.finalize()
        if report_path:
            print(f"[METRICS] wrote -> {report_path}")
            structured_logger.info(
                "metrics_written",
                path=str(report_path),
                metrics=report_payload.get("metrics", {}),
            )

if __name__ == "__main__":
    main()
