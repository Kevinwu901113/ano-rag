import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Dict, List

from loguru import logger

from adapters import get_adapter
from config.config_loader import config as global_config
from generator.note_generator import NoteGenerator
from generator.pronoun_resolver import resolve_pronouns_for_doc
from indexer.index_builder import IndexBuilder


def build_notes(
    dataset: str,
    data_dir: str,
    notes_out: str,
    indexes_dir: str,
    vllm_endpoint: str,
    vllm_model: str,
    shard_idx: int = 0,
    shard_cnt: int = 1,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    progress_path: str | None = None,
) -> dict:
    adapter = get_adapter(dataset)
    vllm_cfg = (global_config.get("vllm", {}) or {})
    default_max_tokens = (
        vllm_cfg.get("max_new_tokens") or vllm_cfg.get("max_tokens") or 700
    )
    try:
        resolved_max_tokens = int(max_tokens if max_tokens is not None else default_max_tokens)
    except (TypeError, ValueError):
        resolved_max_tokens = int(default_max_tokens)
    if resolved_max_tokens <= 0:
        resolved_max_tokens = 1
    generator = NoteGenerator(
        vllm_endpoint,
        vllm_model,
        temperature=temperature,
        max_tokens=resolved_max_tokens,
        # Use dynamic backend pool rather than pinning to a strict endpoint
        strict_endpoint=False,
    )

    if shard_cnt < 1:
        raise ValueError(f"shard_cnt must be >= 1 (got {shard_cnt})")
    if not 0 <= shard_idx < shard_cnt:
        raise ValueError(f"shard_idx must be in [0, {shard_cnt - 1}] (got {shard_idx})")

    notes_path = Path(notes_out)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    # Concurrency controls (per shard)
    ccfg = vllm_cfg.get("concurrency", {})
    # 从配置读取并发参数；若关闭自适应，上下限一致
    max_workers = int(ccfg.get("max_workers", 8))
    acfg = (vllm_cfg.get("adaptive", {}) or {})
    adaptive_enabled = bool(acfg.get("enabled", False))
    upper_workers = int(acfg.get("max_workers", max_workers)) if adaptive_enabled else max_workers

    def _process_one(chunk):
        try:
            return generator.generate_for_chunk(chunk)
        except Exception as exc:
            logger.warning("Shard {} chunk failed doc={} chunk={} err={}", shard_idx, chunk.get("doc_id"), chunk.get("chunk_id"), exc)
            return []

    # Progress state
    written = 0
    processed_chunks = 0
    start_ts = time.time()
    doc_notes: Dict[str, List[Dict]] = {}
    doc_order: List[str] = []
    doc_lock = threading.Lock()

    def _stash_notes(doc_id: str, notes: List[Dict]) -> None:
        nonlocal written
        if not notes:
            return
        key = doc_id or "__default__"
        with doc_lock:
            if key not in doc_notes:
                doc_notes[key] = []
                doc_order.append(key)
            doc_notes[key].extend(notes)
        written += len(notes)

    def _emit_progress(total: int, completed: bool = False, current_workers: int | None = None) -> None:
        if not progress_path:
            return
        payload = {
            "total_chunks": int(total),
            "processed_chunks": int(processed_chunks),
            "start_time": float(start_ts),
            "notes_written": int(written),
            "completed": bool(completed),
        }
        if current_workers is not None:
            payload["current_workers"] = int(current_workers)
        try:
            with open(progress_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to write progress path={} err={}", progress_path, exc)

    # Collect ALL chunks into a shared task list (no static sharding)
    chunk_records = []
    for _idx, (_doc, chunk) in enumerate(adapter(data_dir)):
        chunk_records.append(chunk)

    total = len(chunk_records)
    _emit_progress(total, completed=False)

    if max_workers <= 1:
        for chunk in chunk_records:
            notes = _process_one(chunk)
            _stash_notes(chunk.get("doc_id") or "__default__", notes)
            processed_chunks += 1
            _emit_progress(total, completed=False, current_workers=1)
    else:
        # Saturate pool with adaptive target over the shared task list
        with concurrent.futures.ThreadPoolExecutor(max_workers=upper_workers) as executor:
            inflight = set()
            i = 0
            n = total
            target = max_workers
            last_check = time.time()
            future_doc: Dict[concurrent.futures.Future, str] = {}
            # 根据自适应开关决定是否周期性检查
            check_interval = float(acfg.get("cool_down_sec", 5.0)) if adaptive_enabled else 1e9

            def _maybe_update_target():
                nonlocal target, last_check
                now = time.time()
                if (now - last_check) >= max(1.0, check_interval):
                    if adaptive_enabled:
                        try:
                            suggested = generator.suggest_concurrency()
                            target = max(1, min(suggested, upper_workers))
                        except Exception:
                            pass
                        finally:
                            last_check = now
                    else:
                        # 关闭自适应：保持固定目标
                        target = upper_workers
                        last_check = now

            def _submit(idx: int) -> None:
                fut = executor.submit(_process_one, chunk_records[idx])
                inflight.add(fut)
                future_doc[fut] = chunk_records[idx].get("doc_id") or "__default__"

            # Prime
            while i < n and len(inflight) < target:
                _submit(i)
                i += 1
            # Maintain saturation
            while inflight:
                done, inflight = concurrent.futures.wait(inflight, return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    doc_id = future_doc.pop(fut, "__default__")
                    notes = fut.result()
                    _stash_notes(doc_id, notes)
                    processed_chunks += 1
                    _emit_progress(total, completed=False, current_workers=len(inflight))
                _maybe_update_target()
                while i < n and len(inflight) < target:
                    _submit(i)
                    i += 1

    # Persist after doc-level pronoun resolution
    actual_written = 0
    with open(notes_path, "w", encoding="utf-8") as handle:
        for doc_id in doc_order:
            resolved_notes = resolve_pronouns_for_doc(doc_id, doc_notes.get(doc_id, []))
            if not resolved_notes:
                continue
            for note in resolved_notes:
                handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                actual_written += 1
    written = actual_written

    _emit_progress(total, completed=True)

    logger.info("Notes written to {} ({} notes)", notes_out, written)

    # Always build indexes in dynamic task mode
    builder = IndexBuilder()
    builder.build_from_jsonl(str(notes_path))
    builder.dump(indexes_dir)
    logger.info("Indexes dumped to {}", indexes_dir)

    return {"notes_written": written, "shard_idx": shard_idx, "shard_cnt": shard_cnt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build notes and indexes for structured pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset adapter name (e.g., mirage)")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--out", required=True, help="Output notes JSONL path")
    parser.add_argument("--indexes_dir", required=True, help="Output indexes directory")
    parser.add_argument("--vllm_endpoint", required=True)
    parser.add_argument("--vllm_model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--shard-cnt", type=int, default=1)
    parser.add_argument("--progress-path", default=None)
    args = parser.parse_args()

    Path(args.indexes_dir).mkdir(parents=True, exist_ok=True)
    build_notes(
        dataset=args.dataset,
        data_dir=args.data_dir,
        notes_out=args.out,
        indexes_dir=args.indexes_dir,
        vllm_endpoint=args.vllm_endpoint,
        vllm_model=args.vllm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        shard_idx=args.shard_idx,
        shard_cnt=args.shard_cnt,
        progress_path=args.progress_path,
    )


if __name__ == "__main__":
    main()
