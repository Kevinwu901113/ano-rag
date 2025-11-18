import concurrent.futures
import math
import threading
import time
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from config.config_loader import config as global_config

from doc import make_chunks
from generator.note_generator import NoteGenerator
from generator.pronoun_resolver import resolve_pronouns_for_doc
from indexer.index_builder import IndexBuilder
from utils import FileUtils, TextUtils
from utils.weak_notes import close_weak_note_writer, write_weak_note
from telemetry.metrics import record_pronoun_stat
from postprocess.notes_postprocess import (
    backfill_pronoun_subjects,
    stitch_pronoun_notes,
    SUBJECT_TYPE_HINTS,
    OBJECT_TYPE_HINTS,
)


MAX_TOKEN_BOUND = 9_999_999
DEFAULT_BUCKET_LAYOUT: List[Tuple[str, Tuple[int, int]]] = [
    ("0_256", (0, 256)),
    ("256_512", (256, 512)),
    ("512_1024", (512, 1024)),
    ("1024_plus", (1024, MAX_TOKEN_BOUND)),
]

SUBJECT_FALLBACK_PREDS: Set[str] = {
    "born_in",
    "born_on",
    "birth_place",
    "born",
    "served_as",
    "served_in",
    "served_with",
    "joined",
    "member_of",
    "affiliated_with",
    "married",
    "spouse",
    "spouse_of",
    "partner",
    "worked_at",
    "worked_for",
    "employment",
    "occupation",
    "title",
    "position_held",
    "role",
    "appointed",
    "elected",
    "led",
    "headed",
}


class EntityLedger:
    def __init__(self, capacity: int = 3) -> None:
        self.capacity = max(1, capacity)
        self._store: Dict[str, deque] = {
            "PERSON": deque(maxlen=self.capacity),
            "ORG": deque(maxlen=self.capacity),
            "PLACE": deque(maxlen=self.capacity),
        }

    def remember(self, entity: Optional[str], entity_type: Optional[str], confidence: float = 1.0) -> None:
        if not entity:
            return
        etype = (entity_type or TextUtils.guess_entity_type(entity) or "").upper()
        if not etype or etype not in self._store:
            return
        bucket = self._store[etype]
        for idx, (val, _) in enumerate(bucket):
            if val == entity:
                del bucket[idx]
                break
        bucket.appendleft((entity, float(confidence or 0.0)))

    def resolve(self, entity_type: Optional[str]) -> Tuple[Optional[str], float]:
        etype = (entity_type or "").upper()
        if etype not in self._store:
            return None, 0.0
        bucket = self._store[etype]
        if not bucket:
            return None, 0.0
        entity, conf = bucket[0]
        return entity, float(conf or 0.0)

    def primary_anchor(self) -> Optional[str]:
        for etype in ("PERSON", "ORG", "PLACE"):
            bucket = self._store.get(etype)
            if bucket:
                return bucket[0][0]
        return None


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class StructuredBuilder:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("vLLM endpoint/model must be provided")
        vllm_defaults = global_config.get("vllm", {}) if "global_config" in globals() else {}
        default_max = (
            (vllm_defaults or {}).get("max_new_tokens")
            or (vllm_defaults or {}).get("max_tokens")
            or 700
        )
        try:
            resolved_max = int(max_tokens if max_tokens is not None else default_max)
        except (TypeError, ValueError):
            resolved_max = int(default_max)
        if resolved_max <= 0:
            resolved_max = 1
        self.generator = NoteGenerator(endpoint, model, temperature, resolved_max)
        coref_cfg = global_config.get("coref", {}) if "global_config" in globals() else {}
        self._ledger_capacity = max(1, int(coref_cfg.get("ledger_capacity", 3) or 3))
        self._doc_ledgers: Dict[str, EntityLedger] = {}
        self._doc_locks: Dict[str, threading.Lock] = {}
        self._doc_lock_guard = threading.Lock()

    def _collect_chunks(self, data_dir: Path) -> List[Dict]:
        files = FileUtils.list_files(str(data_dir), [".txt", ".md", ".jsonl", ".json"])
        if not files:
            logger.warning("No documents found in {}", data_dir)
            return []

        chunks: List[Dict] = []
        for file_path in files:
            path = Path(file_path)
            doc_id = path.stem
            if path.suffix.lower() == ".jsonl":
                for idx, row in enumerate(FileUtils.read_jsonl(str(path))):
                    text = row.get("text") or row.get("content") or ""
                    if not isinstance(text, str):
                        continue
                    chunks.extend(
                        make_chunks(f"{doc_id}_{idx:04d}", text, chunk_id_prefix="c")
                    )
            else:
                text = _read_text(path)
                chunks.extend(make_chunks(doc_id, text, chunk_id_prefix="c"))
        return chunks

    def build(
        self,
        data_dir: str,
        notes_out: str,
        indexes_dir: str,
        chunks_out: Optional[str] = None,
    ) -> Dict[str, int]:
        chunk_records = self._collect_chunks(Path(data_dir))
        if not chunk_records:
            return {"chunks": 0, "notes": 0}

        notes_path = Path(notes_out)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        weak_out_dir = str(notes_path.parent)
        stats = {
            "pronoun_subj_before": 0,
            "pronoun_resolved_strong": 0,
            "weak_written": 0,
        }

        if chunks_out is None:
            chunks_out = str(notes_path.parent / "chunks.jsonl")
        FileUtils.write_jsonl(chunks_out, chunk_records)
        logger.info("Wrote {} chunks to {}", len(chunk_records), chunks_out)

        # Concurrency settings
        vllm_cfg = global_config.get("vllm", {}) if "global_config" in globals() else {}
        ccfg = (vllm_cfg or {}).get("concurrency", {}) or {}
        timeout_pause_threshold = float(ccfg.get("pause_on_timeout_rate", 0.3))
        pause_sec = float(ccfg.get("pause_sec", 7.0))
        refill_factor = max(1.0, float(ccfg.get("refill_factor", 1.5)))
        bucket_cfg = ccfg.get("buckets")
        bucket_cfg = bucket_cfg if isinstance(bucket_cfg, dict) else {}

        def _resolve_bucket_bounds(label: str, payload: Optional[Dict[str, Any]]) -> Tuple[int, int]:
            if payload:
                rng = payload.get("range")
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    try:
                        return int(rng[0]), int(rng[1])
                    except (TypeError, ValueError):
                        pass
            parts = label.split("_")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])
            if len(parts) == 2 and parts[0].isdigit() and parts[1].lower() in {"plus", "up"}:
                return int(parts[0]), MAX_TOKEN_BOUND
            return 0, MAX_TOKEN_BOUND

        if bucket_cfg:
            bucket_layout: List[Tuple[str, Tuple[int, int]]] = []
            for label, payload in bucket_cfg.items():
                payload_dict = payload if isinstance(payload, dict) else {}
                bucket_layout.append((label, _resolve_bucket_bounds(label, payload_dict)))
        else:
            bucket_layout = list(DEFAULT_BUCKET_LAYOUT)

        fallback_total_workers = max(1, int(ccfg.get("max_workers", 4)))
        fallback_per_bucket = max(1, fallback_total_workers // max(1, len(bucket_layout)))

        bucket_specs: List[Dict[str, Any]] = []
        for label, bounds in bucket_layout:
            payload = bucket_cfg.get(label) if isinstance(bucket_cfg, dict) else {}
            workers_val = payload.get("workers") if isinstance(payload, dict) else None
            workers = workers_val if workers_val is not None else fallback_per_bucket
            workers = max(1, int(workers))
            bucket_specs.append({"name": label, "bounds": bounds, "workers": workers})

        def _which_bucket(text: str) -> str:
            tlen = TextUtils.rough_token_len(text)
            for spec in bucket_specs:
                lower, upper = spec["bounds"]
                if lower <= tlen < upper:
                    return spec["name"]
            return bucket_specs[-1]["name"]

        bucket_records: Dict[str, List[Dict[str, Any]]] = {spec["name"]: [] for spec in bucket_specs}
        for chunk in chunk_records:
            bucket_name = _which_bucket(chunk.get("text") or "")
            bucket_records[bucket_name].append(chunk)

        active_specs = [spec for spec in bucket_specs if bucket_records.get(spec["name"])]
        if not active_specs:
            active_specs = bucket_specs[-1:]

        total_workers = sum(spec["workers"] for spec in active_specs)

        def _maybe_pause_submission() -> None:
            threshold = max(0.0, min(1.0, timeout_pause_threshold))
            if threshold <= 0.0:
                return
            try:
                timeout_rate = self.generator.recent_timeout_rate()
            except Exception:
                timeout_rate = 0.0
            if timeout_rate >= threshold:
                logger.warning(
                    "High timeout rate {:.1%} detected; pausing new submissions for {:.1f}s",
                    timeout_rate,
                    pause_sec,
                )
                time.sleep(pause_sec)

        def _canonicalize_sentence(subject: str | None, sentence: str) -> str:
            # 仅句首独立代词替换；避免宾语/物主误替换
            if not sentence:
                return sentence
            s = sentence.strip()
            if not subject:
                return s
            parts = s.split()
            if parts and TextUtils.is_pronoun(parts[0]):
                return (subject or parts[0]) + " " + " ".join(parts[1:])
            # 中文句首
            import re as _re
            m = _re.match(rf"^({'|'.join(TextUtils.ZH_PRONOUNS)})", s)
            if m:
                return (subject or s[: m.end()]).strip() + s[m.end():]
            return s

        def _process_with_ledger(chunk: Dict[str, Any], ledger: EntityLedger):
            try:
                # Generate notes for chunk
                notes = self.generator.generate_for_chunk(chunk)
                try:
                    notes = stitch_pronoun_notes(notes, chunk)
                except Exception:
                    pass
                # Perform minimal postprocess: pronoun backfill and alias map for this chunk
                try:
                    stubs, alias_map, alias_to_canonical = backfill_pronoun_subjects(chunk)
                except Exception:
                    stubs, alias_map, alias_to_canonical = [], {}, {}
                # Attach alias_map into each note's meta; mark unresolved pronoun if detected
                enriched: List[Dict] = []
                chunk_recent_entities = (chunk.get("meta") or {}).get("recent_entities") or []
                for ent in chunk_recent_entities[:2]:
                    ledger.remember(ent, None, confidence=0.7)
                for note in notes:
                    meta = (note.get("meta") or {})
                    if not isinstance(meta, dict):
                        meta = {}
                    note["meta"] = meta

                    def _bump_conf(field: str, value: float) -> None:
                        try:
                            prev = float(meta.get(field, 0.0))
                        except (TypeError, ValueError):
                            prev = 0.0
                        meta[field] = max(prev, value)
                    # Set alias map only once per chunk in meta (small duplication acceptable in JSONL)
                    meta["alias_map"] = alias_map
                    # Entities mentioned in evidence mapped to canonical via alias_to_canonical
                    ev_text = (note.get("evidence") or "")
                    entities_surface = TextUtils.extract_entity_candidates(ev_text)
                    entities_canonical = []
                    for surf in entities_surface:
                        canon = alias_to_canonical.get(surf.lower()) or surf
                        if canon not in entities_canonical:
                            entities_canonical.append(canon)
                    meta["entities"] = entities_canonical
                    # anchor_entity: 若该块存在唯一实体，记录之；用于检索时的回填
                    uniq_entities = [e for e in entities_canonical]
                    if len(uniq_entities) == 1:
                        meta["anchor_entity"] = uniq_entities[0]
                    elif not meta.get("anchor_entity"):
                        anchor_from_ledger = ledger.primary_anchor()
                        if anchor_from_ledger:
                            meta["anchor_entity"] = anchor_from_ledger

                    # lead_in_note_id: 若发生了回拉或前置拼接，记录来源 stub 的 note_id
                    lead_in_note_id = None

                    # If pronoun unresolved in stub for same sentence, propagate flag
                    # We attempt to match evidence start with sentence text
                    has_unresolved = False
                    original_subject_hint = None
                    resolved_subject = None
                    resolved_source = None
                    resolved_confidence = 0.0
                    for stub in stubs:
                        stub_meta = (stub.get("meta") or {})
                        if stub_meta.get("has_unresolved_pronoun"):
                            # Weak heuristic: if stub evidence is contained in note evidence
                            ev = (note.get("evidence") or "")
                            sev = (stub.get("evidence") or "")
                            if sev and ev and sev in ev:
                                has_unresolved = True
                                original_subject_hint = original_subject_hint or stub_meta.get("original_subject")
                                lead_in_note_id = stub.get("note_id")
                                stub_candidates = stub_meta.get("coref_candidates")
                                if stub_candidates:
                                    meta["coref_candidates"] = deepcopy(stub_candidates)
                                break
                        # Also allow positive subject backfill when stub has resolved subject and matches evidence
                        subj_stub = stub.get("subj")
                        if subj_stub:
                            ev = (note.get("evidence") or "")
                            sev = (stub.get("evidence") or "")
                            if sev and ev and sev in ev:
                                resolved_subject = subj_stub
                                resolved_source = stub_meta.get("subject_source") or "window_backfill"
                                stub_conf = stub_meta.get("coref_confidence")
                                if not isinstance(stub_conf, (int, float)):
                                    stub_conf = stub_meta.get("confidence")
                                try:
                                    stub_conf_val = float(stub_conf)
                                except (TypeError, ValueError):
                                    stub_conf_val = 0.8
                                resolved_confidence = max(resolved_confidence, stub_conf_val)
                                if not original_subject_hint:
                                    original_subject_hint = stub_meta.get("original_subject")
                                stub_candidates = stub_meta.get("coref_candidates")
                                if stub_candidates:
                                    meta["coref_candidates"] = deepcopy(stub_candidates)
                                # do not break: prefer unresolved flag detection above; but keep a candidate
                    # Fallback: direct pronoun lead detection on evidence
                    if not has_unresolved and TextUtils.is_pronoun_subject_sentence(ev_text):
                        has_unresolved = True
                        if not original_subject_hint:
                            original_subject_hint = ev_text.split(" ")[0]
                    if has_unresolved:
                        meta["has_unresolved_pronoun"] = True
                        if original_subject_hint and not meta.get("original_subject"):
                            meta["original_subject"] = original_subject_hint
                    if lead_in_note_id:
                        meta["lead_in_note_id"] = lead_in_note_id

                    subj_text = (note.get("subj") or "").strip()
                    subj_is_pronoun = bool(subj_text and TextUtils.is_pronoun(subj_text))
                    subj_was_pronoun = subj_is_pronoun
                    if subj_is_pronoun:
                        stats["pronoun_subj_before"] += 1
                    obj_text = (note.get("obj") or "").strip()
                    obj_is_pronoun = bool(obj_text and TextUtils.is_pronoun(obj_text))
                    original_pronoun = original_subject_hint or (subj_text if subj_is_pronoun else None)
                    if original_pronoun and not meta.get("original_subject"):
                        meta["original_subject"] = original_pronoun

                    canonical_subject = None
                    desired_subj_type = SUBJECT_TYPE_HINTS.get((note.get("pred") or "").lower()) or note.get("subj_type")
                    desired_obj_type = OBJECT_TYPE_HINTS.get((note.get("pred") or "").lower()) or note.get("obj_type")

                    if subj_is_pronoun and resolved_subject:
                        note["subj"] = resolved_subject
                        if not meta.get("subject_source"):
                            meta["subject_source"] = resolved_source or "window_backfill"
                        conf_hint = max(0.75, (resolved_confidence or 0.8))
                        _bump_conf("subject_confidence", conf_hint)
                        _bump_conf("coref_confidence", conf_hint)
                        prev_coref = meta.get("coref_confidence") or 0.0
                        meta["coref_confidence"] = max(prev_coref, conf_hint)
                        subj_is_pronoun = False
                        canonical_subject = resolved_subject
                    elif subj_is_pronoun and not resolved_subject:
                        anchor = meta.get("anchor_entity")
                        pred = (note.get("pred") or "").lower()
                        if anchor and pred in SUBJECT_FALLBACK_PREDS:
                            note["subj"] = anchor
                            meta["subject_source"] = "anchor_fallback"
                            _bump_conf("subject_confidence", 0.5)
                            _bump_conf("coref_confidence", 0.5)
                            subj_is_pronoun = False
                            canonical_subject = anchor

                    if subj_is_pronoun and desired_subj_type:
                        ledger_candidate, ledger_conf = ledger.resolve(desired_subj_type)
                        if ledger_candidate:
                            note["subj"] = ledger_candidate
                            meta["subject_source"] = "ledger_backfill"
                            _bump_conf("subject_confidence", ledger_conf or 0.7)
                            _bump_conf("coref_confidence", ledger_conf or 0.7)
                            subj_is_pronoun = False
                            canonical_subject = ledger_candidate

                    # Object fallback: when唯一实体可指代对象且不同于主体
                    if obj_is_pronoun:
                        obj_candidates = [e for e in entities_canonical if e and e != note.get("subj")]
                        if len(obj_candidates) == 1:
                            note["obj"] = obj_candidates[0]
                            meta["object_source"] = "intra_sentence_entity"
                            _bump_conf("coref_confidence_obj", 0.6)
                            obj_is_pronoun = False
                    if obj_is_pronoun and desired_obj_type:
                        ledger_obj, ledger_obj_conf = ledger.resolve(desired_obj_type)
                        if ledger_obj and ledger_obj != note.get("subj"):
                            note["obj"] = ledger_obj
                            meta["object_source"] = "ledger_backfill"
                            _bump_conf("coref_confidence_obj", ledger_obj_conf or 0.6)
                            obj_is_pronoun = False

                    if subj_was_pronoun and not subj_is_pronoun:
                        stats["pronoun_resolved_strong"] += 1
                        meta.pop("filter_out_strict", None)
                        violations = meta.get("violations")
                        if isinstance(violations, dict):
                            violations.pop("coref_unresolved", None)
                            if not violations:
                                meta.pop("violations")
                        canonical_subject = canonical_subject or note.get("subj")

                    # evidence canonical：句首代词在置信条件满足时替换为最近主体
                    # 触发条件：句内无第二实体名，且与上一句间隔≤1句（借助 stub 匹配）
                    canonical_ev = ev_text
                    try:
                        if canonical_subject:
                            ev_entities = TextUtils.extract_entity_candidates(ev_text)
                            if len(ev_entities) <= 1:
                                canonical_ev = _canonicalize_sentence(canonical_subject, ev_text)
                    except Exception:
                        canonical_ev = ev_text
                    meta["evidence_canonical"] = canonical_ev

                    # 若主体仍为代词，将 note 作为弱证据写入侧轨索引
                    if subj_is_pronoun:
                        try:
                            coref_snapshot = deepcopy(meta.get("coref_candidates") or [])
                            write_weak_note(weak_out_dir, note, coref_snapshot, meta.get("anchor_entity"))
                            stats["weak_written"] += 1
                        except Exception as exc:
                            logger.warning(
                                "Failed to persist weak note doc={} chunk={} err={}",
                                chunk.get("doc_id"),
                                chunk.get("chunk_id"),
                                exc,
                            )
                        continue

                    note["meta"] = meta
                    if not TextUtils.is_pronoun(note.get("subj") or ""):
                        ledger.remember(
                            note.get("subj"),
                            desired_subj_type,
                            meta.get("subject_confidence") or meta.get("coref_confidence") or 0.8,
                        )
                    if note.get("obj") and not TextUtils.is_pronoun(note.get("obj")):
                        ledger.remember(
                            note.get("obj"),
                            desired_obj_type,
                            meta.get("coref_confidence_obj") or 0.6,
                        )
                    enriched.append(note)
                return enriched
            except Exception as exc:
                logger.warning("Chunk generation failed doc={} chunk={} err={}", chunk.get("doc_id"), chunk.get("chunk_id"), exc)
                return []

        def _process_one(chunk: Dict[str, Any]):
            doc_id = chunk.get("doc_id") or "__default__"
            lock = self._lock_for_doc(doc_id)
            with lock:
                ledger = self._doc_ledgers.get(doc_id)
                if ledger is None:
                    ledger = EntityLedger(self._ledger_capacity)
                    self._doc_ledgers[doc_id] = ledger
                return _process_with_ledger(chunk, ledger)

        notes_written = 0
        doc_notes: Dict[str, List[Dict[str, Any]]] = {}
        doc_order: List[str] = []
        doc_lock = threading.Lock()

        def _stash_notes(doc_id: str, notes: List[Dict[str, Any]]) -> None:
            if not notes:
                return
            key = doc_id or "__default__"
            with doc_lock:
                if key not in doc_notes:
                    doc_notes[key] = []
                    doc_order.append(key)
                doc_notes[key].extend(notes)

        if total_workers <= 1:
            for chunk in chunk_records:
                _stash_notes(chunk.get("doc_id") or "__default__", _process_one(chunk))
        else:
            with ExitStack() as stack:
                executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
                for spec in active_specs:
                    executors[spec["name"]] = stack.enter_context(
                        concurrent.futures.ThreadPoolExecutor(
                            max_workers=spec["workers"],
                            thread_name_prefix=f"bucket-{spec['name']}",
                        )
                    )

                inflight: Dict[str, Set[concurrent.futures.Future]] = {
                    spec["name"]: set() for spec in active_specs
                }
                bucket_indices: Dict[str, int] = {spec["name"]: 0 for spec in active_specs}
                bucket_limits: Dict[str, int] = {
                    spec["name"]: max(spec["workers"], math.ceil(spec["workers"] * refill_factor))
                    for spec in active_specs
                }
                future_bucket: Dict[concurrent.futures.Future, str] = {}
                future_doc: Dict[concurrent.futures.Future, str] = {}
                all_futures: Set[concurrent.futures.Future] = set()

                def _refill() -> None:
                    for spec in active_specs:
                        name = spec["name"]
                        records = bucket_records[name]
                        if not records:
                            continue
                        inflight_set = inflight[name]
                        limit = bucket_limits[name]
                        while bucket_indices[name] < len(records) and len(inflight_set) < limit:
                            _maybe_pause_submission()
                            chunk = records[bucket_indices[name]]
                            fut = executors[name].submit(_process_one, chunk)
                            inflight_set.add(fut)
                            future_bucket[fut] = name
                            future_doc[fut] = chunk.get("doc_id") or "__default__"
                            all_futures.add(fut)
                            bucket_indices[name] += 1

                _refill()
                while all_futures:
                    done, _ = concurrent.futures.wait(all_futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        all_futures.discard(fut)
                        bucket_name = future_bucket.pop(fut, None)
                        if bucket_name:
                            inflight[bucket_name].discard(fut)
                        notes = fut.result()
                        _stash_notes(future_doc.pop(fut, "__default__"), notes)
                    _refill()

        with open(notes_path, "w", encoding="utf-8") as handle:
            for doc_id in doc_order:
                resolved_notes = resolve_pronouns_for_doc(doc_id, doc_notes.get(doc_id, []))
                if not resolved_notes:
                    continue
                notes_written += FileUtils.write_jsonl_batch(handle, resolved_notes)

        logger.info("Wrote {} notes to {}", notes_written, notes_path)
        close_weak_note_writer(weak_out_dir)
        weak_notes_path = Path(weak_out_dir) / "weak" / "weak_notes.jsonl"
        if stats["weak_written"]:
            logger.info("Wrote {} weak notes to {}", stats["weak_written"], weak_notes_path)
        elif weak_notes_path.exists():
            weak_notes_path.unlink()

        logger.info(
            "Pronoun stats: before={} resolved={} weak_notes={}",
            stats["pronoun_subj_before"],
            stats["pronoun_resolved_strong"],
            stats["weak_written"],
        )
        record_pronoun_stat("pronoun_subj_before", stats["pronoun_subj_before"])
        record_pronoun_stat("pronoun_resolved_strong", stats["pronoun_resolved_strong"])
        record_pronoun_stat("weak_written", stats["weak_written"])

        weak_notes_path_str = str(weak_notes_path) if weak_notes_path.exists() else None
        if notes_written:
            builder = IndexBuilder()
            builder.build_from_jsonl(str(notes_path))
            builder.build_weak_from_jsonl(weak_notes_path_str)
            builder.dump(indexes_dir)
            logger.info("Indexes dumped to {}", indexes_dir)
        else:
            logger.warning("No notes generated; skipping index build.")

        return {"chunks": len(chunk_records), "notes": notes_written}

    def _lock_for_doc(self, doc_id: str) -> threading.Lock:
        key = doc_id or "__default__"
        with self._doc_lock_guard:
            lock = self._doc_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._doc_locks[key] = lock
            return lock
