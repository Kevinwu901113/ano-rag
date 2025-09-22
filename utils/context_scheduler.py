from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

from loguru import logger

from config import config


class ContextScheduler:
    """Select context passages following a multi-stage pipeline."""

    _FALLBACK_TOPK = 10

    def __init__(self, scheduler_config: Optional[Dict[str, Any]] = None):
        cs_config = scheduler_config or config.get("context_scheduler", {})
        scheduler_defaults = (
            cs_config.get("scheduler", {})
            if isinstance(cs_config, Mapping)
            else {}
        )
        if not scheduler_defaults and isinstance(cs_config, Mapping):
            scheduler_defaults = cs_config

        self._raw_config: Dict[str, Any] = dict(cs_config) if isinstance(cs_config, Mapping) else {}
        self._scheduler_config: Dict[str, Any] = (
            dict(scheduler_defaults) if isinstance(scheduler_defaults, Mapping) else {}
        )

        weight_source: Mapping[str, Any]
        if isinstance(self._scheduler_config.get("weights"), Mapping):
            weight_source = self._scheduler_config["weights"]
        elif isinstance(self._raw_config, Mapping):
            weight_source = self._raw_config
        else:
            weight_source = {}

        self.t1 = float(weight_source.get("semantic_weight", 0.3))
        self.t2 = float(weight_source.get("graph_weight", 0.25))
        self.t3 = float(weight_source.get("topic_weight", 0.2))
        self.t4 = float(weight_source.get("feedback_weight", 0.15))
        self.t5 = float(weight_source.get("redundancy_penalty", 0.1))

        default_topk = self._scheduler_config.get(
            "topk",
            self._scheduler_config.get(
                "top_n_notes",
                self._scheduler_config.get(
                    "top_n",
                    self._raw_config.get("top_n_notes", self._FALLBACK_TOPK),
                ),
            ),
        )

        self._defaults: Dict[str, Any] = {
            "topk": self._normalize_int(default_topk) or self._FALLBACK_TOPK,
            "neighbor_hops": self._normalize_int(
                self._scheduler_config.get("neighbor_hops", 1)
            )
            or 1,
            "max_tokens": self._normalize_int(
                self._scheduler_config.get("max_tokens")
            ),
            "time_window_sec": self._normalize_int(
                self._scheduler_config.get("time_window_sec")
            ),
            "per_cluster_limit": self._normalize_int(
                self._scheduler_config.get("per_cluster_limit")
            ),
            "per_source_limit": self._normalize_int(
                self._scheduler_config.get("per_source_limit")
            ),
            "per_subquestion_limit": self._normalize_int(
                self._scheduler_config.get("per_subquestion_limit")
            ),
            "per_paragraph_limit": self._normalize_int(
                self._scheduler_config.get("per_paragraph_limit")
            ),
            "max_neighbor_expansion": self._normalize_int(
                self._scheduler_config.get("max_neighbor_expansion")
                or self._scheduler_config.get("neighbor_limit")
            ),
            "chars_per_token": self._normalize_int(
                self._scheduler_config.get("chars_per_token")
            )
            or 4,
            "token_overhead": self._normalize_int(
                self._scheduler_config.get("token_overhead")
            )
            or 0,
        }

        self._diversity_limits: Dict[str, Optional[int]] = {
            "cluster": self._defaults.get("per_cluster_limit"),
            "source": self._defaults.get("per_source_limit"),
            "subq": self._defaults.get("per_subquestion_limit"),
            "paragraph": self._defaults.get("per_paragraph_limit"),
        }

        self.top_n = int(self._defaults.get("topk") or self._FALLBACK_TOPK)

    def schedule(
        self,
        candidates: Sequence[Dict[str, Any]],
        per_candidate: Optional[Mapping[str, Any]] = None,
        *,
        topk: Optional[int] = None,
        neighbor_hops: int = 1,
        max_tokens: Optional[int] = None,
        query_processor=None,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            logger.warning("No candidate notes provided to scheduler")
            return []

        candidate_list = [c for c in candidates if isinstance(c, MutableMapping)]
        if not candidate_list:
            return []

        coverage_guard_enabled = (
            config.get("dispatcher", {})
            .get("scheduler", {})
            .get("coverage_guard", False)
        )
        if coverage_guard_enabled:
            candidate_list = self._apply_coverage_guard(
                candidate_list, query_processor
            )

        deduped = self._dedup(candidate_list, per_candidate)

        topk_value = self._normalize_int(topk)
        if topk_value is None:
            topk_value = self._defaults.get("topk")
        neighbor_hops = self._normalize_int(neighbor_hops)
        if neighbor_hops is None:
            neighbor_hops = self._defaults.get("neighbor_hops", 1)
        neighbor_hops = max(neighbor_hops or 0, 0)
        token_budget = (
            self._normalize_int(max_tokens)
            if max_tokens is not None
            else self._defaults.get("max_tokens")
        )

        sampled = self._diversity_sample(deduped, per_candidate, topk_value)
        expanded = self._expand_neighbors(
            sampled, deduped, per_candidate, neighbor_hops
        )
        reordered = self._reorder(expanded, per_candidate)
        limited = self._apply_token_budget(reordered, token_budget)

        if topk_value:
            limited = limited[: int(topk_value)]

        logger.info(
            "Context scheduler selected %s notes from %s candidates",
            len(limited),
            len(candidates),
        )
        return limited

    # --- Pipeline stages -------------------------------------------------

    def _dedup(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        per_candidate: Optional[Mapping[str, Any]] = None,
    ) -> List[MutableMapping[str, Any]]:
        unique: List[MutableMapping[str, Any]] = []
        seen_by_id: Dict[Any, int] = {}
        seen_by_content: Dict[str, int] = {}

        for note in candidates:
            note_id = self._candidate_id(note)
            content = note.get("content")
            score = self._candidate_score(note, per_candidate)
            replacement_index: Optional[int] = None

            if note_id in seen_by_id:
                replacement_index = seen_by_id[note_id]
            elif isinstance(content, str) and content in seen_by_content:
                replacement_index = seen_by_content[content]

            if replacement_index is not None:
                existing = unique[replacement_index]
                if score > existing.get("context_score", float("-inf")):
                    note["context_score"] = score
                    unique[replacement_index] = note
                    if isinstance(content, str):
                        seen_by_content[content] = replacement_index
                continue

            note["context_score"] = score
            index = len(unique)
            unique.append(note)
            seen_by_id[note_id] = index
            if isinstance(content, str):
                seen_by_content[content] = index

        logger.info(
            "After content deduplication: %s unique notes from %s candidates",
            len(unique),
            len(candidates),
        )
        return unique

    def _diversity_sample(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        per_candidate: Optional[Mapping[str, Any]],
        topk: Optional[int],
    ) -> List[MutableMapping[str, Any]]:
        scored: List[MutableMapping[str, Any]] = []
        for note in candidates:
            self._score_candidate(note, per_candidate)
            scored.append(note)

        scored.sort(key=lambda x: x.get("context_score", 0), reverse=True)
        limit = self._normalize_int(topk)
        selected = self._respect_diversity_quota(scored, limit)

        if not selected and scored:
            fallback = limit if limit is not None else min(3, len(scored))
            fallback = max(fallback, 1)
            selected = scored[:fallback]

        return selected

    def _expand_neighbors(
        self,
        selected: Sequence[MutableMapping[str, Any]],
        pool: Sequence[MutableMapping[str, Any]],
        per_candidate: Optional[Mapping[str, Any]],
        neighbor_hops: int,
    ) -> List[MutableMapping[str, Any]]:
        if neighbor_hops <= 0:
            return list(selected)

        index = self._build_inverted_index(pool)
        selected_ids = {
            self._candidate_id(note)
            for note in selected
            if isinstance(note, Mapping)
        }
        expanded = list(selected)
        frontier: Set[str] = set(selected_ids)

        for _ in range(neighbor_hops):
            if not frontier:
                break

            paragraph_neighbors = self._find_paragraph_neighbors(frontier, index)
            temporal_neighbors = self._find_temporal_neighbors(frontier, index)
            topic_neighbors = self._find_topic_neighbors(frontier, index)

            neighbor_ids = (
                paragraph_neighbors | temporal_neighbors | topic_neighbors
            ) - selected_ids
            if not neighbor_ids:
                frontier = set()
                continue

            additions = self._cap_neighbors(
                neighbor_ids,
                selected_ids,
                per_candidate,
                index,
                expanded,
            )
            if not additions:
                frontier = set()
                continue

            for candidate in additions:
                cid = self._candidate_id(candidate)
                if cid in selected_ids:
                    continue
                expanded.append(candidate)
                selected_ids.add(cid)

            frontier = {
                self._candidate_id(candidate)
                for candidate in additions
                if self._candidate_id(candidate) is not None
            }

        return expanded

    def _reorder(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        per_candidate: Optional[Mapping[str, Any]],
    ) -> List[MutableMapping[str, Any]]:
        ordered: List[MutableMapping[str, Any]] = []
        seen_ids: Set[str] = set()
        for candidate in candidates:
            cid = self._candidate_id(candidate)
            if cid in seen_ids:
                continue
            self._score_candidate(candidate, per_candidate)
            ordered.append(candidate)
            seen_ids.add(cid)
        ordered.sort(key=lambda x: x.get("context_score", 0), reverse=True)
        return ordered

    def _apply_token_budget(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        max_tokens: Optional[int],
    ) -> List[MutableMapping[str, Any]]:
        if max_tokens is None or max_tokens <= 0:
            return list(candidates)

        selected: List[MutableMapping[str, Any]] = []
        used = 0
        for candidate in candidates:
            tokens = self._approx_tokens(candidate)
            candidate.setdefault("approx_tokens", tokens)

            if selected and used + tokens > max_tokens:
                break

            if not selected and tokens > max_tokens:
                selected.append(candidate)
                break

            selected.append(candidate)
            used += tokens

        return selected

    # --- Helper utilities ------------------------------------------------

    def _candidate_id(self, candidate: MutableMapping[str, Any]) -> str:
        for key in ("note_id", "id", "uuid", "document_id", "doc_id"):
            value = candidate.get(key)
            if value is not None:
                return str(value)
        internal = candidate.get("_scheduler_id")
        if internal is None:
            internal = f"auto_{id(candidate)}"
            candidate["_scheduler_id"] = internal
        return str(internal)

    def _candidate_score(
        self,
        candidate: Mapping[str, Any],
        per_candidate: Optional[Mapping[str, Any]] = None,
    ) -> float:
        candidate_id = None
        if per_candidate:
            candidate_id = self._candidate_id(candidate)  # type: ignore[arg-type]
            payload = per_candidate.get(candidate_id)
            if isinstance(payload, Mapping):
                for key in ("score", "multi_hop_score", "context_score"):
                    value = payload.get(key)
                    if isinstance(value, (int, float)):
                        return float(value)
            elif isinstance(payload, (int, float)):
                return float(payload)
        return self._legacy_score(candidate)

    def _score_candidate(
        self,
        candidate: MutableMapping[str, Any],
        per_candidate: Optional[Mapping[str, Any]] = None,
    ) -> float:
        score = self._candidate_score(candidate, per_candidate)
        candidate["context_score"] = score
        return score

    def _legacy_score(self, note: Mapping[str, Any]) -> float:
        retrieval_info = note.get("retrieval_info", {})
        if not isinstance(retrieval_info, Mapping):
            retrieval_info = {}
        semantic = self._ensure_float(retrieval_info.get("similarity", 0))
        graph_score = self._ensure_float(note.get("centrality", 0))
        topic_score = 1.0 if note.get("cluster_id") is not None else 0.0
        feedback = self._ensure_float(note.get("feedback_score", 0))
        redundancy = self._ensure_float(note.get("redundancy_penalty", 0))

        score = (
            self.t1 * semantic
            + self.t2 * graph_score
            + self.t3 * topic_score
            + self.t4 * feedback
        )
        if redundancy:
            score -= self.t5 * redundancy
        return float(score)

    def _ensure_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _build_inverted_index(
        self, candidates: Sequence[MutableMapping[str, Any]]
    ) -> Dict[str, Any]:
        by_id: Dict[str, MutableMapping[str, Any]] = {}
        paragraph_index: Dict[Any, Set[str]] = defaultdict(set)
        cluster_index: Dict[Any, Set[str]] = defaultdict(set)
        topic_index: Dict[Any, Set[str]] = defaultdict(set)
        source_index: Dict[Any, Set[str]] = defaultdict(set)
        subq_index: Dict[Any, Set[str]] = defaultdict(set)
        timeline: List[tuple[float, str]] = []

        for note in candidates:
            note_id = self._candidate_id(note)
            by_id[note_id] = note

            paragraph_id = self._get_paragraph_id(note)
            if paragraph_id is not None:
                paragraph_index[paragraph_id].add(note_id)

            cluster_id = self._get_cluster_id(note)
            if cluster_id is not None:
                cluster_index[cluster_id].add(note_id)

            topic_id = self._get_topic_id(note)
            if topic_id is not None:
                topic_index[topic_id].add(note_id)

            source_id = self._get_source_id(note)
            if source_id is not None:
                source_index[source_id].add(note_id)

            subq_id = note.get("subq_id")
            if subq_id is not None:
                subq_index[subq_id].add(note_id)

            timestamp = self._get_timestamp(note)
            if timestamp is not None:
                timeline.append((timestamp, note_id))

        timeline.sort(key=lambda item: item[0])

        return {
            "by_id": by_id,
            "paragraph": paragraph_index,
            "cluster": cluster_index,
            "topic": topic_index,
            "source": source_index,
            "subq": subq_index,
            "time": timeline,
        }

    def _find_paragraph_neighbors(
        self, frontier: Iterable[str], index: Mapping[str, Any]
    ) -> Set[str]:
        neighbors: Set[str] = set()
        for nid in frontier:
            note = index["by_id"].get(nid)
            if not note:
                continue
            paragraph_id = self._get_paragraph_id(note)
            if paragraph_id is None:
                continue
            neighbors.update(index["paragraph"].get(paragraph_id, set()))
        return neighbors

    def _find_temporal_neighbors(
        self, frontier: Iterable[str], index: Mapping[str, Any]
    ) -> Set[str]:
        window = self._defaults.get("time_window_sec")
        if not window:
            return set()
        timeline = index.get("time", [])
        if not timeline:
            return set()

        neighbors: Set[str] = set()
        for nid in frontier:
            note = index["by_id"].get(nid)
            if not note:
                continue
            timestamp = self._get_timestamp(note)
            if timestamp is None:
                continue
            neighbors.update(self._select_from_time_window(timeline, timestamp, window))
        return neighbors

    def _find_topic_neighbors(
        self, frontier: Iterable[str], index: Mapping[str, Any]
    ) -> Set[str]:
        neighbors: Set[str] = set()
        for nid in frontier:
            note = index["by_id"].get(nid)
            if not note:
                continue
            cluster_id = self._get_cluster_id(note)
            if cluster_id is not None:
                neighbors.update(index["cluster"].get(cluster_id, set()))
            topic_id = self._get_topic_id(note)
            if topic_id is not None:
                neighbors.update(index["topic"].get(topic_id, set()))
        return neighbors

    def _cap_neighbors(
        self,
        neighbor_ids: Set[str],
        selected_ids: Set[str],
        per_candidate: Optional[Mapping[str, Any]],
        index: Mapping[str, Any],
        existing: Sequence[MutableMapping[str, Any]],
    ) -> List[MutableMapping[str, Any]]:
        candidates = self._select_from_ids(
            index["by_id"], neighbor_ids, per_candidate
        )
        if not candidates:
            return []

        limit = self._defaults.get("max_neighbor_expansion")
        if limit is not None and limit <= 0:
            return []

        filtered = [c for c in candidates if self._candidate_id(c) not in selected_ids]
        selected = self._respect_diversity_quota(filtered, limit, existing)
        return selected


    def _select_from_ids(
        self,
        index: Mapping[str, MutableMapping[str, Any]],
        ids: Iterable[str],
        per_candidate: Optional[Mapping[str, Any]],
    ) -> List[MutableMapping[str, Any]]:
        collected: List[MutableMapping[str, Any]] = []
        for nid in ids:
            candidate = index.get(nid)
            if not candidate:
                continue
            self._score_candidate(candidate, per_candidate)
            collected.append(candidate)
        collected.sort(key=lambda x: x.get("context_score", 0), reverse=True)
        return collected

    def _select_from_time_window(
        self,
        timeline: Sequence[tuple[float, str]],
        center: float,
        window: int,
    ) -> Set[str]:
        if window <= 0:
            return set()
        lower = center - window
        upper = center + window
        if lower > upper:
            lower, upper = upper, lower
        neighbors: Set[str] = set()
        for ts, nid in timeline:
            if ts < lower:
                continue
            if ts > upper:
                break
            neighbors.add(nid)
        return neighbors

    def _respect_diversity_quota(
        self,
        candidates: Sequence[MutableMapping[str, Any]],
        max_count: Optional[int],
        existing: Optional[Sequence[MutableMapping[str, Any]]] = None,
    ) -> List[MutableMapping[str, Any]]:
        if max_count is not None and max_count <= 0:
            return []

        counters: Dict[str, Dict[Any, int]] = {
            key: defaultdict(int) for key in self._diversity_limits
        }

        if existing:
            for note in existing:
                self._update_diversity_counters(counters, note)

        selected: List[MutableMapping[str, Any]] = []
        for candidate in candidates:
            if max_count is not None and len(selected) >= max_count:
                break
            if self._passes_quota(counters, candidate):
                selected.append(candidate)
                self._update_diversity_counters(counters, candidate)

        return selected

    def _passes_quota(
        self,
        counters: Mapping[str, Mapping[Any, int]],
        candidate: Mapping[str, Any],
    ) -> bool:
        for key, limit in self._diversity_limits.items():
            if not limit or limit <= 0:
                continue
            value = self._extract_diversity_value(candidate, key)
            if value is None:
                continue
            if counters[key][value] >= limit:
                return False
        return True

    def _update_diversity_counters(
        self,
        counters: MutableMapping[str, MutableMapping[Any, int]],
        candidate: Mapping[str, Any],
    ) -> None:
        for key in self._diversity_limits:
            value = self._extract_diversity_value(candidate, key)
            if value is None:
                continue
            counters[key][value] += 1

    def _extract_diversity_value(
        self, candidate: Mapping[str, Any], key: str
    ) -> Any:
        if key == "cluster":
            return self._get_cluster_id(candidate)
        if key == "source":
            return self._get_source_id(candidate)
        if key == "subq":
            return candidate.get("subq_id")
        if key == "paragraph":
            return self._get_paragraph_id(candidate)
        return None

    def _approx_tokens(self, candidate: Mapping[str, Any]) -> int:
        for key in ("token_count", "token_length", "tokens", "approx_tokens"):
            value = candidate.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)

        content = candidate.get("content") or candidate.get("text") or ""
        if not isinstance(content, str):
            content = ""
        chars_per_token = max(1, int(self._defaults.get("chars_per_token", 4)))
        approx = len(content) // chars_per_token if content else 0
        approx += int(self._defaults.get("token_overhead", 0) or 0)
        return max(1, approx)

    def _get_timestamp(self, candidate: Mapping[str, Any]) -> Optional[float]:
        value = (
            candidate.get("timestamp")
            or candidate.get("created_at")
            or candidate.get("published_at")
        )
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(value, fmt).timestamp()
                    except ValueError:
                        continue
        return None

    def _get_paragraph_id(self, candidate: Mapping[str, Any]) -> Any:
        if candidate.get("paragraph_id") is not None:
            return candidate.get("paragraph_id")
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("paragraph_id", "paragraph"):
                value = metadata.get(key)
                if value is not None:
                    return value
        return candidate.get("paragraph")

    def _get_cluster_id(self, candidate: Mapping[str, Any]) -> Any:
        if candidate.get("cluster_id") is not None:
            return candidate.get("cluster_id")
        return candidate.get("cluster")

    def _get_topic_id(self, candidate: Mapping[str, Any]) -> Any:
        if candidate.get("topic_id") is not None:
            return candidate.get("topic_id")
        return candidate.get("topic")

    def _get_source_id(self, candidate: Mapping[str, Any]) -> Any:
        for key in ("source_id", "document_id", "doc_id", "source", "namespace"):
            value = candidate.get(key)
            if value is not None:
                return value
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("source_id", "document_id", "doc_id", "source", "namespace"):
                value = metadata.get(key)
                if value is not None:
                    return value
        return None

    def _normalize_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # --- Existing helper -------------------------------------------------

    def _apply_coverage_guard(
        self,
        candidate_notes: List[Dict[str, Any]],
        query_processor=None,
    ) -> List[Dict[str, Any]]:
        if not candidate_notes:
            return candidate_notes

        subq_coverage: Dict[Any, List[Dict[str, Any]]] = {}
        for note in candidate_notes:
            subq_id = note.get("subq_id")
            if subq_id is not None:
                subq_coverage.setdefault(subq_id, []).append(note)

        if not subq_coverage:
            logger.warning("No subq_id found in candidates, skipping coverage guard")
            return candidate_notes

        missing_subqs = [
            subq_id for subq_id, notes in subq_coverage.items() if len(notes) == 0
        ]

        if missing_subqs:
            logger.warning(
                "Coverage guard detected missing evidence for subquestions: %s",
                missing_subqs,
            )

            if query_processor and hasattr(
                query_processor, "_fallback_retrieval_for_subquestion"
            ):
                logger.info(
                    "Attempting fallback retrieval for %s missing subquestions",
                    len(missing_subqs),
                )
                for subq_id in missing_subqs:
                    try:
                        logger.debug(
                            "Would perform fallback retrieval for subq_id: %s",
                            subq_id,
                        )
                    except Exception as exc:  # pragma: no cover - logging only
                        logger.error(
                            "Fallback retrieval failed for subq_id %s: %s",
                            subq_id,
                            exc,
                        )

            logger.error(
                "Coverage guard report - Missing subquestions: %s", missing_subqs
            )
            for subq_id in missing_subqs:
                logger.error(
                    "Missing subq_id: %s, expected entities: [to be implemented]",
                    subq_id,
                )

        final_notes: List[Dict[str, Any]] = []
        for subq_id, notes in subq_coverage.items():
            if notes:
                best_note = max(notes, key=lambda x: x.get("similarity", 0))
                final_notes.append(best_note)
                for note in notes:
                    if note is not best_note:
                        final_notes.append(note)

        for note in candidate_notes:
            if note.get("subq_id") is None:
                final_notes.append(note)

        logger.info(
            "Coverage guard processed: %s -> %s notes",
            len(candidate_notes),
            len(final_notes),
        )
        return final_notes


class MultiHopContextScheduler(ContextScheduler):
    """Scheduler with reasoning path awareness."""

    def __init__(self, scheduler_config: Optional[Dict[str, Any]] = None):
        super().__init__(scheduler_config=scheduler_config)

    def schedule_for_multi_hop(
        self,
        candidate_notes: Sequence[Dict[str, Any]],
        reasoning_paths: Sequence[Dict[str, Any]],
        *,
        topk: Optional[int] = None,
        neighbor_hops: int = 1,
        max_tokens: Optional[int] = None,
        query_processor=None,
    ) -> List[Dict[str, Any]]:
        candidate_list = [c for c in candidate_notes if isinstance(c, MutableMapping)]
        if not candidate_list:
            logger.warning(
                "No candidate notes provided to multi-hop scheduler"
            )
            return []

        path_scores = self._calculate_path_scores(candidate_list, reasoning_paths)

        per_candidate: Dict[str, Dict[str, float]] = {}
        for note in candidate_list:
            base_score = self._calculate_base_score(note)
            note_id = self._candidate_id(note)
            path_score = path_scores.get(note_id, 0.0)
            completeness = self._calculate_completeness_score(
                note, reasoning_paths
            )
            total = 0.3 * base_score + 0.4 * path_score + 0.3 * completeness
            note["multi_hop_score"] = total
            per_candidate[note_id] = {
                "score": total,
                "base_score": base_score,
                "path_score": path_score,
                "completeness": completeness,
            }

        scheduled = super().schedule(
            candidate_list,
            per_candidate=per_candidate,
            topk=topk,
            neighbor_hops=neighbor_hops,
            max_tokens=max_tokens,
            query_processor=query_processor,
        )

        if not reasoning_paths:
            return scheduled

        scored_candidates = sorted(
            candidate_list,
            key=lambda x: x.get("multi_hop_score", 0),
            reverse=True,
        )
        ensured = self._ensure_reasoning_chain_completeness(
            scored_candidates, reasoning_paths
        )

        if ensured:
            merged: Dict[str, Dict[str, Any]] = {
                self._candidate_id(note): note for note in scheduled
            }
            for note in ensured:
                merged[self._candidate_id(note)] = note
            merged_list = list(merged.values())
            merged_list = self._reorder(merged_list, per_candidate)
            merged_list = self._apply_token_budget(
                merged_list,
                self._normalize_int(max_tokens)
                if max_tokens is not None
                else self._defaults.get("max_tokens"),
            )
            top_limit = self._normalize_int(topk)
            if top_limit is None:
                top_limit = self._defaults.get("topk")
            if top_limit:
                merged_list = merged_list[: int(top_limit)]
            scheduled = merged_list
        elif not scheduled and scored_candidates:
            fallback_count = min(3, len(scored_candidates))
            scheduled = scored_candidates[:fallback_count]

        logger.info(
            "Multi-hop scheduler selected %s notes from %s candidates",
            len(scheduled),
            len(candidate_notes),
        )
        return scheduled

    def _calculate_base_score(self, note: Dict[str, Any]) -> float:
        semantic = note.get("retrieval_info", {}).get("similarity", 0)
        graph_score = note.get("centrality", 0)
        topic_score = 1.0 if note.get("cluster_id") is not None else 0.0
        feedback = note.get("feedback_score", 0)
        return (
            self.t1 * semantic
            + self.t2 * graph_score
            + self.t3 * topic_score
            + self.t4 * feedback
        )

    def _calculate_path_scores(
        self, candidate_notes: Sequence[Dict[str, Any]], reasoning_paths: Sequence[Dict[str, Any]]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for path in reasoning_paths:
            nodes = path.get("path") or path.get("nodes", [])
            score = path.get("path_score", 0.0)
            for nid in nodes:
                key = str(nid)
                scores[key] = scores.get(key, 0.0) + score
        return scores

    def _calculate_completeness_score(
        self, note: Dict[str, Any], reasoning_paths: Sequence[Dict[str, Any]]
    ) -> float:
        total = len(reasoning_paths)
        if total == 0:
            return 0.0
        nid = str(self._candidate_id(note))
        count = sum(
            1
            for path in reasoning_paths
            if nid in (path.get("path") or path.get("nodes", []))
        )
        return count / total

    def _ensure_reasoning_chain_completeness(
        self,
        notes: Sequence[Dict[str, Any]],
        paths: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        covered: Set[tuple] = set()
        sorted_notes = sorted(
            notes,
            key=lambda x: x.get("multi_hop_score", x.get("context_score", 0)),
            reverse=True,
        )
        for note in sorted_notes:
            nid = str(self._candidate_id(note))
            relevant = [
                p for p in paths if nid in (p.get("path") or p.get("nodes", []))
            ]
            if not relevant:
                continue
            new_paths = [
                tuple(p.get("path") or p.get("nodes", [])) for p in relevant
            ]
            new_paths = [p for p in new_paths if p not in covered]
            if new_paths or len(selected) < 3:
                selected.append(note)
                for path_tuple in new_paths:
                    covered.add(path_tuple)
            if len(selected) >= self.top_n:
                break
        return selected
