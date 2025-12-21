#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

from loguru import logger
from tqdm import tqdm

## Ensure repo root on sys.path for absolute imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config as global_config
from generator.note_generator import NoteGenerator
from generator.answerer import call_lmstudio
from utils.logging_utils import setup_logging
from utils.notes_cache import NotesCache
from utils.run_layout import ensure_workdir_layout, resolve_workdir
from utils.vllm_server_manager import VLLMServerManager


# -------------------------------
# Helper structures
# -------------------------------
@dataclass
class ManifestItem:
    qid: str
    allowed_hashes: List[str]
    pid_map: Dict[str, str]


def _sha1(text: str) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    if dataset_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    with open(dataset_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        # allow dict with key 'data'
        data = data.get("data") or data.get("examples") or data.get("questions") or []
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list or a dict containing a list under 'data/examples/questions'")
    return data


def _build_manifests(dataset: List[Dict[str, Any]]) -> List[ManifestItem]:
    manifests: List[ManifestItem] = []
    for item in dataset:
        qid = str(item.get("id") or item.get("query_id") or item.get("qid") or "")
        if not qid:
            continue
        # Try multiple schema variations
        paragraphs: List[Tuple[str, str]] = []  # (pid, text)
        if isinstance(item.get("paragraphs"), list):
            for idx, p in enumerate(item["paragraphs"]):
                if isinstance(p, dict):
                    pid = str(p.get("pid") or p.get("id") or p.get("para_id") or f"p{idx:04d}")
                    raw_text = p.get("text") or p.get("content") or p.get("paragraph") or p.get("para") or p.get("paragraph_text")
                    if not raw_text and isinstance(p.get("sentences"), list):
                        try:
                            raw_text = " ".join([str(s) for s in p.get("sentences") if str(s)])
                        except Exception:
                            raw_text = ""
                    title = p.get("title") or ""
                    text = f"{title}. {str(raw_text)}" if (title and raw_text) else str(raw_text or "")
                else:
                    pid = f"p{idx:04d}"
                    text = str(p)
                if pid and text:
                    paragraphs.append((pid, text))
        elif isinstance(item.get("contexts"), list):
            for idx, ctx in enumerate(item["contexts"]):
                if isinstance(ctx, dict):
                    pid = str(ctx.get("pid") or ctx.get("id") or f"p{idx:04d}")
                    raw_text = ctx.get("text") or ctx.get("content") or ctx.get("paragraph") or ctx.get("para") or ctx.get("paragraph_text")
                    if not raw_text and isinstance(ctx.get("sentences"), list):
                        try:
                            raw_text = " ".join([str(s) for s in ctx.get("sentences") if str(s)])
                        except Exception:
                            raw_text = ""
                    title = ctx.get("title") or ""
                    text = f"{title}. {str(raw_text)}" if (title and raw_text) else str(raw_text or "")
                else:
                    pid = f"p{idx:04d}"
                    text = str(ctx)
                if pid and text:
                    paragraphs.append((pid, text))
        elif isinstance(item.get("passages"), list):
            for idx, ctx in enumerate(item["passages"]):
                pid = str(ctx.get("pid") or ctx.get("id") or f"p{idx:04d}")
                raw_text = ctx.get("text") or ctx.get("content") or ctx.get("paragraph") or ctx.get("para") or ctx.get("paragraph_text")
                if not raw_text and isinstance(ctx.get("sentences"), list):
                    try:
                        raw_text = " ".join([str(s) for s in ctx.get("sentences") if str(s)])
                    except Exception:
                        raw_text = ""
                title = ctx.get("title") or ""
                text = f"{title}. {str(raw_text)}" if (title and raw_text) else str(raw_text or "")
                if pid and text:
                    paragraphs.append((pid, text))
        else:
            # Fallback: single context
            text = str(item.get("context") or item.get("passage") or "")
            if text:
                paragraphs.append(("p0000", text))

        pid_map: Dict[str, str] = {}
        allowed_hashes: List[str] = []
        for pid, text in paragraphs:
            h = _sha1(text)
            pid_map[h] = pid
            allowed_hashes.append(h)
        manifests.append(ManifestItem(qid=qid, allowed_hashes=allowed_hashes, pid_map=pid_map))
    return manifests


def _select_workspace(result_root: Path, new: bool, tag: Optional[str]) -> Path:
    result_root.mkdir(parents=True, exist_ok=True)
    if new:
        if tag:
            name = f"musique_{tag}"
        else:
            name = f"musique_{time.strftime('%Y-%m-%dT%H-%M-%S')}"
        work = result_root / name
        work.mkdir(parents=True, exist_ok=True)
        return work
    # pick latest existing musique_* dir
    candidates = [p for p in result_root.iterdir() if p.is_dir() and p.name.startswith("musique_")]
    if not candidates:
        # create new if none
        return _select_workspace(result_root, True, tag)
    # prefer by ctime then lexicographic
    candidates.sort(key=lambda p: (p.stat().st_ctime, p.name), reverse=True)
    return candidates[0]


def _write_run_info(work_dir: Path, meta: Dict[str, Any]) -> None:
    path = work_dir / "RUN.info"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def _acquire_lock(work_dir: Path) -> None:
    lock_path = work_dir / "RUN.lock"
    if lock_path.exists():
        try:
            with open(lock_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            pid = int(data.get("pid") or 0)
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    raise RuntimeError(f"Existing run detected (pid={pid}); refuse to start.")
                except ProcessLookupError:
                    pass
        except Exception:
            # non-json or unreadable: continue
            pass
    with open(lock_path, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_at": int(time.time())}, handle)


def _release_lock(work_dir: Path) -> None:
    try:
        (work_dir / "RUN.lock").unlink(missing_ok=True)
    except Exception:
        pass


def _warm_up_vllm(endpoint: str, model: str, log_path: Path) -> None:
    try:
        import requests
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "ping"}],
        }
        resp = requests.post(f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} vLLM ready\n")
    except Exception as exc:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%H:%M:%S')} vLLM warm-up failed: {exc}\n")


def _warm_up_lmstudio(endpoint: str, model: str, log_path: Path, max_wait_sec: int = 120) -> None:
    """Warm up LM Studio by triggering model load and verifying chat readiness.

    Conditions to finish:
    - `/models` lists the target model id, and
    - a test `POST /chat/completions` returns 200 with choices.
    """
    start_ts = time.time()
    try:
        import requests
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "ping"}],
        }
        # Initial ping to trigger lazy load (ignore errors during spin-up)
        try:
            requests.post(f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=10)
        except Exception:
            pass

        models_ready = False
        chat_ready = False
        deadline = start_ts + max_wait_sec
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            # Check /models
            try:
                r2 = requests.get(f"{endpoint.rstrip('/')}/models", timeout=5)
                if r2.ok:
                    data = r2.json()
                    items = data.get("data") or []
                    models_ready = any(str(it.get("id") or "") == str(model) for it in items)
                else:
                    models_ready = False
            except Exception:
                models_ready = False

            # If models look ready, verify chat/completions success
            if models_ready:
                try:
                    resp = requests.post(
                        f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=10
                    )
                    if resp.ok:
                        data = resp.json()
                        if isinstance(data, dict) and (data.get("choices") or []):
                            chat_ready = True
                            break
                except Exception:
                    chat_ready = False

            # Re-trigger loader occasionally
            if (attempts % 5) == 0:
                try:
                    requests.post(
                        f"{endpoint.rstrip('/')}/chat/completions", json=payload, timeout=8
                    )
                except Exception:
                    pass
            time.sleep(2)

        with open(log_path, "a", encoding="utf-8") as handle:
            waited = int(time.time() - start_ts)
            status = "ready" if (models_ready and chat_ready) else "timeout"
            handle.write(
                f"{time.strftime('%H:%M:%S')} LM Studio warm-up {status} (waited={waited}s, attempts={attempts}, models_ready={models_ready}, chat_ready={chat_ready})\n"
            )
    except Exception as exc:
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} LM Studio warm-up failed: {exc}\n")
        except Exception:
            pass


# -------------------------------
# Producer/Consumer pipeline
# -------------------------------
def _load_or_build_manifests(work_dir: Path, dataset_path: Path) -> List[ManifestItem]:
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifests_path = artifacts_dir / "manifests.jsonl"
    items: List[ManifestItem] = []
    if manifests_path.exists():
        for row in ManifestReader.read_jsonl(str(manifests_path)):
            qid = str(row.get("id") or row.get("qid") or row.get("query_id"))
            allowed = list(row.get("allowed_hashes") or [])
            pid_map = dict(row.get("pid_map") or {})
            if qid:
                items.append(ManifestItem(qid=qid, allowed_hashes=allowed, pid_map=pid_map))
        # If existing manifests have empty allowed_hashes, rebuild from dataset for correctness
        if any(len(m.allowed_hashes) == 0 for m in items):
            logger.warning("Existing manifests.jsonl contains items with empty allowed_hashes; rebuilding manifests from dataset")
            dataset = _load_dataset(dataset_path)
            built = _build_manifests(dataset)
            with open(manifests_path, "w", encoding="utf-8") as handle:
                for m in built:
                    handle.write(
                        json.dumps({"id": m.qid, "allowed_hashes": m.allowed_hashes, "pid_map": m.pid_map}, ensure_ascii=False)
                        + "\n"
                    )
            return built
        return items
    dataset = _load_dataset(dataset_path)
    built = _build_manifests(dataset)
    with open(manifests_path, "w", encoding="utf-8") as handle:
        for m in built:
            handle.write(
                json.dumps({"id": m.qid, "allowed_hashes": m.allowed_hashes, "pid_map": m.pid_map}, ensure_ascii=False)
                + "\n"
            )
    return built


class ManifestReader:
    @staticmethod
    def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _append_notes_index(work_dir: Path, pairs: List[Tuple[str, str]]) -> None:
    # pairs: (hash, pid)
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    idx_path = artifacts_dir / "notes_index.jsonl"
    # avoid duplicates by in-memory set when file exists
    seen: set[Tuple[str, str]] = set()
    if idx_path.exists():
        try:
            for row in ManifestReader.read_jsonl(str(idx_path)):
                h = str(row.get("hash") or "")
                p = str(row.get("pid") or "")
                if h and p:
                    seen.add((h, p))
        except Exception:
            pass
    with open(idx_path, "a", encoding="utf-8") as handle:
        for h, p in pairs:
            if (h, p) in seen:
                continue
            handle.write(json.dumps({"hash": h, "pid": p}, ensure_ascii=False) + "\n")


def run_pipeline(
    *,
    work_dir: Path,
    dataset_path: Path,
    vllm_endpoint: str,
    vllm_model: str,
    lmstudio_endpoint: Optional[str],
    lmstudio_model: Optional[str],
    max_producer_workers: int = 8,
    consumer_concurrency: int = 2,
) -> None:
    # Directories
    artifacts_dir = work_dir / "artifacts"
    preds_dir = work_dir / "preds"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = artifacts_dir / "logs"
    pending_dir = artifacts_dir / "pending"
    answers_dir = artifacts_dir / "answers"
    notes_dir = artifacts_dir / "notes"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    runner_log_path = logs_dir / "runner.log"
    setup_logging(str(runner_log_path))

    # Build or load manifests
    manifests = _load_or_build_manifests(work_dir, dataset_path)
    logger.info("Loaded {} manifest items", len(manifests))

    # Load dataset into pid->text maps for generation (robust to sentences/title)
    dataset_rows = _load_dataset(dataset_path)
    qid_to_pid_text: Dict[str, Dict[str, str]] = {}
    for item in dataset_rows:
        qid = str(item.get("id") or item.get("query_id") or item.get("qid") or "")
        if not qid:
            continue
        mapping: Dict[str, str] = {}
        if isinstance(item.get("paragraphs"), list):
            for idx, p in enumerate(item["paragraphs"]):
                if isinstance(p, dict):
                    pid = str(p.get("pid") or p.get("id") or p.get("para_id") or f"p{idx:04d}")
                    raw_text = p.get("text") or p.get("content") or p.get("paragraph") or p.get("para") or p.get("paragraph_text")
                    if not raw_text and isinstance(p.get("sentences"), list):
                        try:
                            raw_text = " ".join([str(s) for s in p.get("sentences") if str(s)])
                        except Exception:
                            raw_text = ""
                    title = p.get("title") or ""
                    text = f"{title}. {str(raw_text)}" if (title and raw_text) else str(raw_text or "")
                else:
                    pid = f"p{idx:04d}"
                    text = str(p)
                if pid and text:
                    mapping[pid] = text
        elif isinstance(item.get("contexts"), list):
            for idx, ctx in enumerate(item["contexts"]):
                if isinstance(ctx, dict):
                    pid = str(ctx.get("pid") or ctx.get("id") or f"p{idx:04d}")
                    raw_text = ctx.get("text") or ctx.get("content") or ctx.get("paragraph") or ctx.get("para") or ctx.get("paragraph_text")
                    if not raw_text and isinstance(ctx.get("sentences"), list):
                        try:
                            raw_text = " ".join([str(s) for s in ctx.get("sentences") if str(s)])
                        except Exception:
                            raw_text = ""
                    title = ctx.get("title") or ""
                    text = f"{title}. {str(raw_text)}" if (title and raw_text) else str(raw_text or "")
                else:
                    pid = f"p{idx:04d}"
                    text = str(ctx)
                if pid and text:
                    mapping[pid] = text
        elif isinstance(item.get("passages"), list):
            for idx, ctx in enumerate(item["passages"]):
                pid = str(ctx.get("pid") or ctx.get("id") or f"p{idx:04d}")
                raw_text = ctx.get("text") or ctx.get("content") or ctx.get("paragraph") or ctx.get("para") or ctx.get("paragraph_text")
                if not raw_text and isinstance(ctx.get("sentences"), list):
                    try:
                        raw_text = " ".join([str(s) for s in ctx.get("sentences") if str(s)])
                    except Exception:
                        raw_text = ""
                title = ctx.get("title") or ""
                text = f"{title}. {str(raw_text)}" if (title and raw_text) else str(raw_text or "")
                if pid and text:
                    mapping[pid] = text
        else:
            text = str(item.get("context") or item.get("passage") or "")
            if text:
                mapping["p0000"] = text
        qid_to_pid_text[qid] = mapping

    # Init cache
    cache = NotesCache(str(artifacts_dir / "notes_cache.parquet"))

    # Pending files
    pending_notes_path = pending_dir / "pending_notes.txt"
    failed_notes_path = pending_dir / "failed_notes.jsonl"
    # Answers files
    pred_jsonl_path = answers_dir / "pred.jsonl"
    pred_ckpt_path = answers_dir / "pred.ckpt"
    # Notes out (mirage-style naming)
    notes_out_path = notes_dir / "notes.musique.jsonl"
    # Official MuSiQue evaluation output (JSONL)
    musique_results_path = preds_dir / "musique_results.jsonl"

    # Recover checkpoints
    completed: set[str] = set()
    if pred_ckpt_path.exists():
        try:
            with open(pred_ckpt_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        completed.add(line)
        except Exception:
            pass
    # Load initial pending notes
    resume_hashes: List[str] = []
    if pending_notes_path.exists():
        try:
            with open(pending_notes_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    h = line.strip()
                    if h:
                        resume_hashes.append(h)
        except Exception:
            pass

    # Initialize NoteGenerator
    gen = NoteGenerator(endpoint=vllm_endpoint, model=vllm_model)

    # Warm up vLLM
    _warm_up_vllm(vllm_endpoint, vllm_model, logs_dir / "vllm.log")
    # Warm up LM Studio (if configured): single ping then brief readiness wait
    if lmstudio_endpoint and lmstudio_model:
        _warm_up_lmstudio(lmstudio_endpoint, lmstudio_model, logs_dir / "lmstudio.log")

    # Producer: fill cache for missing hashes (global de-dup)
    to_process: List[Tuple[str, str, str]] = []  # (hash, pid, text)
    # Build global pid->text map for quick lookup
    pid_to_text: Dict[str, str] = {}
    for qmap in qid_to_pid_text.values():
        for pid, text in qmap.items():
            pid_to_text[pid] = text
    # First, schedule resume hashes if provided
    for m in manifests:
        qmap = qid_to_pid_text.get(m.qid) or {}
        for h in m.allowed_hashes:
            pid = m.pid_map.get(h) or ""
            if pid and h in resume_hashes:
                text = pid_to_text.get(pid) or qmap.get(pid) or ""
                if text:
                    to_process.append((h, pid, text))
    # Then schedule missing hashes
    for m in manifests:
        qmap = qid_to_pid_text.get(m.qid) or {}
        for h in m.allowed_hashes:
            if cache.contains(h):
                continue
            pid = m.pid_map.get(h) or ""
            text = pid_to_text.get(pid) or qmap.get(pid) or ""
            if not text:
                # cannot produce without text; record failure
                with open(failed_notes_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"hash": h, "pid": pid, "reason": "missing_text"}, ensure_ascii=False) + "\n")
                continue
            to_process.append((h, pid, text))

    # Persist pending list for resume (idempotent)
    with open(pending_notes_path, "w", encoding="utf-8") as handle:
        for h, _, _ in to_process:
            handle.write(h + "\n")

    # Process notes in batches respecting max_producer_workers saturation
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import threading
    write_lock = threading.Lock()

    def _process_one(hash_pid_text: Tuple[str, str, str]) -> Tuple[str, str, bool]:
        h, pid, text = hash_pid_text
        try:
            # Single-paragraph chunk
            chunk = {"doc_id": pid, "chunk_id": "c0000", "text": text}
            notes = gen.generate_for_chunk(chunk)
            # Ensure note_id->hash mapping can be reconstructed downstream by including meta.source_hash (optional)
            for note in notes:
                meta = (note.get("meta") or {})
                meta["source_hash"] = h
                note["meta"] = meta
            cache.put(h, notes)
            _append_notes_index(work_dir, [(h, pid)])
            # Append to notes.musique.jsonl with thread-safe writes
            with write_lock:
                with open(notes_out_path, "a", encoding="utf-8") as nf:
                    for note in notes:
                        nf.write(json.dumps(note, ensure_ascii=False) + "\n")
            return h, pid, True
        except Exception as exc:
            with open(failed_notes_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"hash": h, "pid": pid, "reason": str(exc)}, ensure_ascii=False) + "\n")
            return h, pid, False

    logger.info("Producer: {} hashes to process", len(to_process))
    if to_process:
        ok_count = 0
        fail_count = 0
        with tqdm(total=len(to_process), desc="Producer", unit="hash", dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=max_producer_workers) as ex:
                futures = {ex.submit(_process_one, item): item for item in to_process}
                for fut in as_completed(futures):
                    h, pid, ok = fut.result()
                    if ok:
                        ok_count += 1
                    else:
                        fail_count += 1
                    pbar.update(1)
                    # Occasionally update postfix for visibility
                    if ((ok_count + fail_count) % 10) == 0:
                        pbar.set_postfix({"ok": ok_count, "fail": fail_count})
        # Rewrite pending to only include missing ones
        missing = [h for h, _, _ in to_process if not cache.contains(h)]
        with open(pending_notes_path, "w", encoding="utf-8") as handle:
            for h in missing:
                handle.write(h + "\n")

    # Retry failed notes at tail (one pass)
    retry_items: List[Tuple[str, str, str]] = []
    if failed_notes_path.exists():
        try:
            with open(failed_notes_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    h = str(row.get("hash") or "")
                    pid = str(row.get("pid") or "")
                    if h and pid and not cache.contains(h):
                        text = pid_to_text.get(pid) or ""
                        if text:
                            retry_items.append((h, pid, text))
        except Exception:
            pass

    if retry_items:
        logger.info("Tail retry: {} failed notes to reattempt", len(retry_items))
        from concurrent.futures import ThreadPoolExecutor as _TE, as_completed as _ac
        ok2 = 0
        fail2 = 0
        with tqdm(total=len(retry_items), desc="Retry", unit="hash", dynamic_ncols=True) as pbar2:
            with _TE(max_workers=max_producer_workers) as ex:
                futs = {ex.submit(_process_one, item): item for item in retry_items}
                for fut in _ac(futs):
                    h, pid, ok = fut.result()
                    if ok:
                        ok2 += 1
                    else:
                        fail2 += 1
                    pbar2.update(1)
                    if ((ok2 + fail2) % 10) == 0:
                        pbar2.set_postfix({"ok": ok2, "fail": fail2})

    # Consumer: answer questions using restricted evidence set
    if not lmstudio_endpoint or not lmstudio_model:
        logger.warning("LM Studio config not provided; skipping answer generation")
        return

    # Build queue of qids where all allowed_hashes are present in cache
    ready_qids: List[ManifestItem] = []
    empty_manifest_count = 0
    for m in manifests:
        if m.qid in completed:
            continue
        if not m.allowed_hashes:
            empty_manifest_count += 1
            logger.warning("Manifest for {} has empty allowed_hashes; skipping answer until notes exist", m.qid)
            continue
        if all(cache.contains(h) for h in m.allowed_hashes):
            ready_qids.append(m)
    if empty_manifest_count:
        logger.warning("Detected {} manifests with empty allowed_hashes (dataset parsing issue?)", empty_manifest_count)
    logger.info("Consumer: {} qids ready for answering", len(ready_qids))

    def _restricted_retrieve(question: str, mitem: ManifestItem) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Gather notes within allowed hashes
        pool_notes: List[Dict[str, Any]] = []
        note_to_pid: Dict[str, str] = {}
        for h in mitem.allowed_hashes:
            notes = cache.get(h) or []
            for note in notes:
                pool_notes.append(note)
                nid = note.get("note_id") or ""
                if nid:
                    # Map via source_hash meta if present; else fallback to pid_map by hash
                    src_h = (note.get("meta") or {}).get("source_hash") or h
                    pid = mitem.pid_map.get(src_h) or ""
                    if pid:
                        note_to_pid[nid] = pid
        # Structural-only evidence selection: rank by meta.final_conf/meta.confidence and evidence length
        def _conf(note: Dict[str, Any]) -> float:
            meta = (note.get("meta") or {})
            val = meta.get("final_conf")
            if val is None:
                val = meta.get("confidence")
            try:
                return float(val) if isinstance(val, (int, float, str)) else 0.0
            except Exception:
                return 0.0

        def _ev_len(note: Dict[str, Any]) -> int:
            ev = note.get("evidence") or ""
            return len(ev) if isinstance(ev, str) else 0

        filtered = [n for n in pool_notes if _ev_len(n) >= 4]
        # Sort: confidence desc, then evidence length desc
        filtered.sort(key=lambda n: (_conf(n), _ev_len(n)), reverse=True)

        # Enforce pid_map boundary and collect top-k note_ids
        top_ids: List[str] = []
        for note in filtered:
            nid = note.get("note_id") or ""
            if not nid:
                continue
            if nid in note_to_pid:
                top_ids.append(nid)
            if len(top_ids) >= 10:
                break

        evidences: List[Dict[str, Any]] = []
        for nid in top_ids:
            note = next((n for n in filtered if (n.get("note_id") or "") == nid), None)
            if note:
                evidences.append(note)

        trace = {
            "strategy": "struct_conf_restricted",
            "pool_size": len(pool_notes),
            "selected_ids": top_ids,
            "allowed_pid_count": len(mitem.pid_map),
        }
        return evidences, trace

    from concurrent.futures import ThreadPoolExecutor
    import threading as _th
    answers_write_lock = _th.Lock()
    def _answer_one(mitem: ManifestItem) -> Tuple[str, bool]:
        try:
            question = next((row.get("question") or row.get("query") for row in dataset_rows if str(row.get("id") or row.get("query_id") or row.get("qid") or "") == mitem.qid), None)
            question = str(question or "")
            evidences, trace = _restricted_retrieve(question, mitem)
            # Format evidences for LM Studio
            lm_evs = []
            evidence_ids = []
            for note in evidences:
                evidence_ids.append(note.get("note_id") or "")
                lm_evs.append({
                    "note_id": note.get("note_id"),
                    "canonical": (note.get("meta") or {}).get("evidence_canonical") or (note.get("evidence") or ""),
                    "evidence": note.get("evidence") or "",
                })
            # Do not retry: LM Studio is pre-warmed; one-shot call avoids repeated access
            answer = call_lmstudio(
                lmstudio_endpoint,
                lmstudio_model,
                question,
                lm_evs,
                temperature=0.2,
                max_tokens=128,
                retries=0,
            )
            # Audit: evidence_note_ids must map back to pid_map
            unmapped = []
            mapped_pids = []
            for nid in evidence_ids:
                # we can recover pid via meta.source_hash or via pid_map using source hash recorded earlier
                note = next((n for n in evidences if (n.get("note_id") or "") == nid), None)
                src_h = (note.get("meta") or {}).get("source_hash") if note else None
                pid = mitem.pid_map.get(str(src_h or "")) if src_h else None
                if pid:
                    mapped_pids.append(pid)
                else:
                    unmapped.append(nid)
            if unmapped:
                logger.warning("越界错误：evidence_note_ids 无法映射到 pid_map: {}", unmapped)
            # Prepare official MuSiQue result line
            pred_evidence = []
            for pid in mapped_pids:
                if pid not in pred_evidence:
                    pred_evidence.append(pid)

            # Append results with thread-safe writes
            with answers_write_lock:
                # pred.jsonl (internal rich trace)
                with open(pred_jsonl_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "id": mitem.qid,
                        "predicted_answer": answer,
                        "evidence_note_ids": evidence_ids,
                        "trace": {**trace, "mapped_pids": mapped_pids, "unmapped": unmapped},
                    }, ensure_ascii=False) + "\n")
                # checkpoint
                with open(pred_ckpt_path, "a", encoding="utf-8") as handle:
                    handle.write(mitem.qid + "\n")
                # official musique_results.jsonl
                with open(musique_results_path, "a", encoding="utf-8") as mrf:
                    mrf.write(json.dumps({
                        "id": mitem.qid,
                        "predicted_answer": answer,
                        "predicted_evidence": pred_evidence,
                    }, ensure_ascii=False) + "\n")
            return mitem.qid, True
        except Exception as exc:
            logger.error("Answer generation failed for {}: {}", mitem.qid, exc)
            return mitem.qid, False

    if ready_qids:
        from concurrent.futures import as_completed as _ac2
        ok_ans = 0
        fail_ans = 0
        with tqdm(total=len(ready_qids), desc="Consumer", unit="qid", dynamic_ncols=True) as pbar_ans:
            with ThreadPoolExecutor(max_workers=max(1, consumer_concurrency)) as ex:
                futures = [ex.submit(_answer_one, m) for m in ready_qids]
                for fut in _ac2(futures):
                    qid, ok = fut.result()
                    if ok:
                        ok_ans += 1
                    else:
                        fail_ans += 1
                    pbar_ans.update(1)
                    if ((ok_ans + fail_ans) % 5) == 0:
                        pbar_ans.set_postfix({"ok": ok_ans, "fail": fail_ans})

    # Emit aggregated answers.json (mirage-style) if we generated answers
    if lmstudio_endpoint and lmstudio_model:
        answers_json_path = preds_dir / "answers.json"
        results: List[Dict[str, Any]] = []
        # Build quick lookup for question by id
        qid_to_question: Dict[str, str] = {}
        for item in dataset_rows:
            qid = str(item.get("id") or item.get("query_id") or item.get("qid") or "")
            question = str(item.get("question") or item.get("query") or "")
            if qid:
                qid_to_question[qid] = question
        # Read pred.jsonl and aggregate
        if pred_jsonl_path.exists():
            with open(pred_jsonl_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    qid = str(row.get("id") or "")
                    ans = row.get("predicted_answer")
                    structured = {
                        "trace": row.get("trace"),
                        "evidence_note_ids": row.get("evidence_note_ids"),
                    }
                    results.append({
                        "query_id": qid,
                        "question": qid_to_question.get(qid, ""),
                        "answer": ans,
                        "structured": structured,
                    })
        with open(answers_json_path, "w", encoding="utf-8") as out:
            json.dump(results, out, ensure_ascii=False, indent=2)
        logger.info("Wrote {} answers to {}", len(results), answers_json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Musique run orchestrator with two-stage pipeline")
    parser.add_argument("--dataset-path", default="data/musique_sample/musique.jsonl")
    parser.add_argument("--result-root", default="result_relrag")
    parser.add_argument("--workdir", "--work-dir", dest="work_dir", default=None)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--tag", default=None, help="Custom tag for <id> (e.g., dev200-run1)")
    parser.add_argument("--vllm-endpoint", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--lmstudio-endpoint", default=None)
    parser.add_argument("--lmstudio-model", default=None)
    parser.add_argument("--autostart-vllm", action="store_true")
    parser.add_argument("--gpu0", default="0", help="GPU id for vLLM")
    parser.add_argument("--producer-workers", type=int, default=8)
    parser.add_argument("--consumer-concurrency", type=int, default=2)
    args = parser.parse_args()

    cfg = global_config.load_config()
    vllm_endpoint = args.vllm_endpoint or cfg.get("vllm.endpoint")
    vllm_model = args.vllm_model or cfg.get("vllm.model")
    if not vllm_endpoint or not vllm_model:
        raise ValueError("vLLM endpoint/model must be provided via CLI or config")

    lmstudio_endpoint = args.lmstudio_endpoint or cfg.get("lmstudio.endpoint")
    lmstudio_model = args.lmstudio_model or cfg.get("lmstudio.model")

    work_dir = resolve_workdir(args.work_dir, result_root=args.result_root, dataset="musique")
    paths = ensure_workdir_layout(work_dir)
    artifacts_dir = paths["artifacts"]
    preds_dir = paths["preds"]
    (artifacts_dir / "answers").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "pending").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "logs").mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "notes").mkdir(parents=True, exist_ok=True)

    # Lock
    _acquire_lock(work_dir)
    atexit.register(lambda: _release_lock(work_dir))

    # Run info
    _write_run_info(work_dir, {
        "id": work_dir.name,
        "start_time": int(time.time()),
        "params": {
            "vllm_endpoint": vllm_endpoint,
            "vllm_model": vllm_model,
            "lmstudio_endpoint": lmstudio_endpoint,
            "lmstudio_model": lmstudio_model,
            "producer_workers": int(args.producer_workers),
            "consumer_concurrency": int(args.consumer_concurrency),
        },
        "gpu_binding": {"vllm": f"GPU{args.gpu0}"},
        "dataset_path": str(args.dataset_path),
    })

    # vLLM autostart
    manager: Optional[VLLMServerManager] = None
    try:
        if args.autostart_vllm:
            logger.info("Autostarting vLLM on GPU{}", args.gpu0)
            # robust port parse from endpoint
            def _parse_port_from_endpoint(url: str) -> int:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    if parsed.port:
                        return int(parsed.port)
                    # fallback: regex search
                    import re as _re
                    m = _re.search(r":(\d+)", url)
                    return int(m.group(1)) if m else 8001
                except Exception:
                    return 8001
            manager = VLLMServerManager({
                "enabled": True,
                "log_dir": str(artifacts_dir / "logs"),
                "servers": [
                    {
                        "name": "vllm_gpu0",
                        "port": _parse_port_from_endpoint(vllm_endpoint),
                        "model": vllm_model,
                        "host": "0.0.0.0",
                        "cuda_devices": str(args.gpu0),
                        "dtype": "float16",
                        "max_model_len": int(cfg.get("chunk.max_tokens", 8192)),
                    }
                ],
            })
            manager.start_all()
            # Log PID and GPU binding
            for meta in manager.get_process_info():
                with open(artifacts_dir / "logs" / "vllm.log", "a", encoding="utf-8") as fh:
                    fh.write(f"PID {meta.get('pid')} on {meta.get('host')}:{meta.get('port')} CUDA={meta.get('cuda_devices')}\n")

        # Signal handling for graceful shutdown
        def _sig_handler(signum, frame):
            logger.info("Received signal {}; initiating graceful shutdown", signum)
            # No new submissions; pending list already persisted by producer
            _write_run_info(work_dir, {"id": work_dir.name, "stop_time": int(time.time()), "status": "stopping"})
            sys.exit(0)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        # Run pipeline
        run_pipeline(
            work_dir=work_dir,
            dataset_path=Path(args.dataset_path),
            vllm_endpoint=vllm_endpoint,
            vllm_model=vllm_model,
            lmstudio_endpoint=lmstudio_endpoint,
            lmstudio_model=lmstudio_model,
            max_producer_workers=int(args.producer_workers),
            consumer_concurrency=int(args.consumer_concurrency),
        )

        # Final stats
        _write_run_info(work_dir, {"id": work_dir.name, "stop_time": int(time.time()), "status": "finished"})
    finally:
        if manager:
            manager.stop_all()


if __name__ == "__main__":
    main()
