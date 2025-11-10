import json
import concurrent.futures
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from config.config_loader import config as global_config

from doc import make_chunks
from generator.note_generator import NoteGenerator
from indexer.index_builder import IndexBuilder
from utils import FileUtils, TextUtils
from postprocess.notes_postprocess import backfill_pronoun_subjects


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class StructuredBuilder:
    def __init__(
        self,
        endpoint: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        if not endpoint or not model:
            raise ValueError("vLLM endpoint/model must be provided")
        self.generator = NoteGenerator(endpoint, model, temperature, max_tokens)

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

        if chunks_out is None:
            chunks_out = str(notes_path.parent / "chunks.jsonl")
        FileUtils.write_jsonl(chunks_out, chunk_records)
        logger.info("Wrote {} chunks to {}", len(chunk_records), chunks_out)

        # Concurrency settings
        vllm_cfg = global_config.get("vllm", {}) if 'global_config' in globals() else {}
        ccfg = (vllm_cfg or {}).get("concurrency", {})
        # 从配置读取；自适应关闭时上下限相同
        max_workers = int(ccfg.get("max_workers", 8))
        acfg = (vllm_cfg or {}).get("adaptive", {})
        adaptive_enabled = bool(acfg.get("enabled", False))
        upper_workers = int(acfg.get("max_workers", max_workers)) if adaptive_enabled else max_workers
        # 提交退避参数：当最近超时率过高时暂停提交
        timeout_pause_threshold = float(ccfg.get("pause_on_timeout_rate", 0.3))
        pause_sec = float(ccfg.get("pause_sec", 7.0))

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

        def _process_one(chunk):
            try:
                # Generate notes for chunk
                notes = self.generator.generate_for_chunk(chunk)
                # Perform minimal postprocess: pronoun backfill and alias map for this chunk
                try:
                    stubs, alias_map, alias_to_canonical = backfill_pronoun_subjects(chunk)
                except Exception:
                    stubs, alias_map, alias_to_canonical = [], {}, {}
                # Attach alias_map into each note's meta; mark unresolved pronoun if detected
                enriched: List[Dict] = []
                for note in notes:
                    meta = (note.get("meta") or {})
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

                    # lead_in_note_id: 若发生了回拉或前置拼接，记录来源 stub 的 note_id
                    lead_in_note_id = None

                    # If pronoun unresolved in stub for same sentence, propagate flag
                    # We attempt to match evidence start with sentence text
                    has_unresolved = False
                    original_subject = None
                    resolved_subject = None
                    for stub in stubs:
                        if (stub.get("meta", {}) or {}).get("has_unresolved_pronoun"):
                            # Weak heuristic: if stub evidence is contained in note evidence
                            ev = (note.get("evidence") or "")
                            sev = (stub.get("evidence") or "")
                            if sev and ev and sev in ev:
                                has_unresolved = True
                                original_subject = (stub.get("meta", {}) or {}).get("original_subject")
                                lead_in_note_id = stub.get("note_id")
                                break
                        # Also allow positive subject backfill when stub has resolved subject and matches evidence
                        subj_stub = stub.get("subj")
                        if subj_stub:
                            ev = (note.get("evidence") or "")
                            sev = (stub.get("evidence") or "")
                            if sev and ev and sev in ev:
                                resolved_subject = subj_stub
                                # do not break: prefer unresolved flag detection above; but keep a candidate
                    # Fallback: direct pronoun lead detection on evidence
                    if not has_unresolved and TextUtils.is_pronoun_subject_sentence(ev_text):
                        has_unresolved = True
                        original_subject = ev_text.split(" ")[0]
                    if has_unresolved:
                        meta["has_unresolved_pronoun"] = True
                        if original_subject:
                            meta["original_subject"] = original_subject
                    if lead_in_note_id:
                        meta["lead_in_note_id"] = lead_in_note_id
                    # If note subject looks like a pronoun, try backfill using stub's resolved subject
                    subj_text = (note.get("subj") or "").strip()
                    if subj_text and TextUtils.is_pronoun(subj_text) and resolved_subject:
                        note["subj"] = resolved_subject
                        meta["subject_source"] = "window_backfill"
                        meta["subject_confidence"] = 0.8

                    # evidence canonical：句首代词在置信条件满足时替换为最近主体
                    # 触发条件：句内无第二实体名，且与上一句间隔≤1句（借助 stub 匹配）
                    canonical_ev = ev_text
                    try:
                        if resolved_subject:
                            # 简单检查：若该证据中的实体候选最多1个，才做替换
                            ev_entities = TextUtils.extract_entity_candidates(ev_text)
                            if len(ev_entities) <= 1:
                                canonical_ev = _canonicalize_sentence(resolved_subject, ev_text)
                    except Exception:
                        canonical_ev = ev_text
                    meta["evidence_canonical"] = canonical_ev

                    # 产出级硬约束：若 subj/obj 为代词且无法回填，直接跳过此 note
                    subj_is_pronoun = TextUtils.is_pronoun((note.get("subj") or "").strip())
                    obj_is_pronoun = TextUtils.is_pronoun((note.get("obj") or "").strip())
                    if (subj_is_pronoun or obj_is_pronoun) and not resolved_subject:
                        # 记录违规上下文，便于调表
                        meta.setdefault("violations", {})
                        meta["violations"]["coref_unresolved"] = True
                        meta["violations"]["evidence"] = ev_text
                        meta["violations"]["chunk_id"] = chunk.get("chunk_id")
                        # Skip adding this note
                        continue

                    note["meta"] = meta
                    enriched.append(note)
                return enriched
            except Exception as exc:
                logger.warning("Chunk generation failed doc={} chunk={} err={}", chunk.get("doc_id"), chunk.get("chunk_id"), exc)
                return []

        notes_written = 0
        with open(notes_path, "w", encoding="utf-8") as handle:
            if max_workers <= 1:
                for chunk in chunk_records:
                    for note in _process_one(chunk):
                        handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                        notes_written += 1
            else:
                # Thread pool for parallel chunk processing, with adaptive target saturation
                with concurrent.futures.ThreadPoolExecutor(max_workers=upper_workers) as executor:
                    inflight = set()
                    idx = 0
                    n_total = len(chunk_records)
                    target = max_workers
                    last_check = time.time()
                    # 周期性检查并发建议与提交退避（无论是否开启自适应）
                    check_interval = float((vllm_cfg or {}).get("adaptive", {}).get("cool_down_sec", 5.0))

                    def _maybe_update_target():
                        nonlocal target, last_check
                        now = time.time()
                        if (now - last_check) >= max(1.0, check_interval):
                            try:
                                suggested = self.generator.suggest_concurrency()
                                target = max(1, min(suggested, upper_workers))
                            except Exception:
                                # 若建议失败，维持当前目标
                                target = max(1, min(target, upper_workers))
                            finally:
                                last_check = now

                    def _maybe_pause_submission():
                        # 当最近超时率过高，暂停提交一段时间，避免重试风暴
                        try:
                            timeout_rate = getattr(self.generator, "recent_timeout_rate")()
                        except Exception:
                            timeout_rate = 0.0
                        if timeout_rate >= max(0.0, min(1.0, timeout_pause_threshold)):
                            logger.warning(
                                "High timeout rate {:.1%} detected; pausing new submissions for {:.1f}s",
                                timeout_rate,
                                pause_sec,
                            )
                            time.sleep(pause_sec)

                    # Prime the pool
                    while idx < n_total and len(inflight) < target:
                        _maybe_pause_submission()
                        fut = executor.submit(_process_one, chunk_records[idx])
                        inflight.add(fut)
                        idx += 1

                    # As each future completes, submit next to maintain target saturation
                    while inflight:
                        done, inflight = concurrent.futures.wait(
                            inflight, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            notes = fut.result()
                            for note in notes:
                                handle.write(json.dumps(note, ensure_ascii=False) + "\n")
                                notes_written += 1

                        _maybe_update_target()
                        # Refill up to target
                        while idx < n_total and len(inflight) < target:
                            _maybe_pause_submission()
                            fut = executor.submit(_process_one, chunk_records[idx])
                            inflight.add(fut)
                            idx += 1

        logger.info("Wrote {} notes to {}", notes_written, notes_path)

        if notes_written:
            builder = IndexBuilder()
            builder.build_from_jsonl(str(notes_path))
            builder.dump(indexes_dir)
            logger.info("Indexes dumped to {}", indexes_dir)
        else:
            logger.warning("No notes generated; skipping index build.")

        return {"chunks": len(chunk_records), "notes": notes_written}
