import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import re
import unicodedata


class Indexes:
    def __init__(self, directory: str) -> None:
        with open(os.path.join(directory, "entity_to_notes.json"), "r", encoding="utf-8") as handle:
            self.entity_to_notes = json.load(handle)

        with open(
            os.path.join(directory, "predicate_to_notes.json"), "r", encoding="utf-8"
        ) as handle:
            self.predicate_to_notes = json.load(handle)

        with open(
            os.path.join(directory, "type_edge_index.json"), "r", encoding="utf-8"
        ) as handle:
            raw = json.load(handle)
            parsed = {}
            for key, value in raw.items():
                subj_type, pred, obj_type = key.split("|", 2)
                parsed[(subj_type, pred, obj_type)] = value
            self.type_edge_index = parsed
        field_index_path = os.path.join(directory, "field_index.json")
        if os.path.exists(field_index_path):
            with open(field_index_path, "r", encoding="utf-8") as handle:
                self.field_index = json.load(handle)
        else:
            self.field_index = {}

        alias_index_path = os.path.join(directory, "entity_alias_index.json")
        if os.path.exists(alias_index_path):
            with open(alias_index_path, "r", encoding="utf-8") as handle:
                raw_alias = json.load(handle)
                self.alias_to_entities = {
                    (alias or "").lower(): values for alias, values in raw_alias.items()
                }
        else:
            self.alias_to_entities = {}

        self.graph_edges = defaultdict(list)
        graph_path = os.path.join(directory, "graph_edges.jsonl")
        with open(graph_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.graph_edges[row["subj"]].extend(row["edges"])

        self.inverse_edges = defaultdict(list)
        inverse_path = os.path.join(directory, "inverse_edges.jsonl")
        with open(inverse_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.inverse_edges[row["obj"]].extend(row["edges"])

        # Mentions edges (note -> [entity])
        self.mentions_edges = defaultdict(list)
        mentions_path = os.path.join(directory, "mentions_edges.jsonl")
        if os.path.exists(mentions_path):
            with open(mentions_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.mentions_edges[row["note_id"]].extend(row.get("mentions", []))

        # Corefers edges (note -> [entity])
        self.corefers_edges = defaultdict(list)
        corefers_path = os.path.join(directory, "corefers_edges.jsonl")
        if os.path.exists(corefers_path):
            with open(corefers_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.corefers_edges[row["note_id"]].extend(row.get("corefers_to", []))

        # Anchor index
        self.anchor_index = {}
        anchor_path = os.path.join(directory, "anchor_index.json")
        if os.path.exists(anchor_path):
            with open(anchor_path, "r", encoding="utf-8") as handle:
                try:
                    self.anchor_index = json.load(handle)
                except Exception:
                    self.anchor_index = {}
        weak_path = os.path.join(directory, "weak_entity_to_notes.json")
        if os.path.exists(weak_path):
            with open(weak_path, "r", encoding="utf-8") as handle:
                self.weak_entity_to_notes = json.load(handle)
        else:
            self.weak_entity_to_notes = {}
        weak_pred_path = os.path.join(directory, "weak_predicate_to_notes.json")
        if os.path.exists(weak_pred_path):
            with open(weak_pred_path, "r", encoding="utf-8") as handle:
                self.weak_predicate_to_notes = json.load(handle)
        else:
            self.weak_predicate_to_notes = {}


def BIND(indexes: Indexes, alias: str, type_candidates: List[str], limit: int = 50) -> List[str]:
    matches: List[str] = []
    bind_reason = None
    target = _normalize_alias_query(alias)
    if not target:
        return matches

    alias_hits = indexes.alias_to_entities.get(target, [])
    for entity in alias_hits:
        if entity not in matches:
            matches.append(entity)
            bind_reason = bind_reason or "alias_exact"
            if len(matches) >= limit:
                return matches

    # 归一化后的精确匹配层（对索引的key进行同步归一）
    for alias_key, entities in indexes.alias_to_entities.items():
        norm_key = _normalize_alias_query(alias_key)
        if norm_key == target and alias_key != target and not alias_hits:
            for entity in entities:
                if entity not in matches:
                    matches.append(entity)
                    bind_reason = bind_reason or "alias_norm_exact"
                    if len(matches) >= limit:
                        return matches

    # 仅当精确匹配未命中且别名索引非空时做包含匹配
    if not matches and indexes.alias_to_entities:
        for alias_key, entities in indexes.alias_to_entities.items():
            if alias_key == target:
                continue
            if target in alias_key or alias_key in target:
                for entity in entities:
                    if entity in matches:
                        continue
                    matches.append(entity)
                    bind_reason = bind_reason or "alias_contains"
                    if len(matches) >= limit:
                        return matches

    exact_norm: List[str] = []
    loose_norm: List[str] = []
    for entity in indexes.entity_to_notes.keys():
        norm_name = _normalize_alias_query(entity)
        if not norm_name or len(norm_name) < 3:
            continue
        if norm_name == target:
            exact_norm.append(entity)
        elif target in norm_name or norm_name in target or _loose_match(target, norm_name):
            loose_norm.append(entity)
    for bucket in (exact_norm, loose_norm):
        for entity in bucket:
            if entity in matches:
                continue
            matches.append(entity)
            bind_reason = bind_reason or "entity_norm"
            if len(matches) >= limit:
                return matches
    # 记录绑定理由（不改变返回结构，供上层日志使用）
    if bind_reason:
        # Attach to a sentinel attribute on the list for tracing (optional usage upstream)
        try:
            matches.bind_reason = bind_reason  # type: ignore[attr-defined]
        except Exception:
            pass
    return matches


def EXPAND_from(
    indexes: Indexes,
    entity: str,
    predicate: Optional[str],
    direction: str = "out",
    limit: int = 200,
) -> List[Tuple[str, str, float]]:
    output: List[Tuple[str, str, float]] = []
    if direction == "in":
        edge_source = indexes.inverse_edges.get(entity, [])
        for edge in edge_source:
            if isinstance(edge, dict):
                pred_val = edge.get("pred")
                subj = edge.get("subj")
                note_id = edge.get("note_id")
                conf = float(edge.get("conf", 0.0))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
                pred_val, subj, note_id = edge[:3]
                conf = 0.0
            else:
                continue
            if predicate and pred_val != predicate:
                continue
            output.append((subj, note_id, conf))
            if len(output) >= limit:
                break
        return output

    for edge in indexes.graph_edges.get(entity, []):
        if isinstance(edge, dict):
            pred_val = edge.get("pred")
            obj = edge.get("obj")
            note_id = edge.get("note_id")
            conf = float(edge.get("conf", 0.0))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 3:
            pred_val, obj, note_id = edge[:3]
            conf = 0.0
        else:
            continue
        if predicate and pred_val != predicate:
            continue
        output.append((obj, note_id, conf))
        if len(output) >= limit:
            break
    return output


def _loose_match(needle: str, hay: str) -> bool:
    if not needle or not hay:
        return False
    tokens = [t for t in needle.split() if len(t) > 3]
    return any(t in hay for t in tokens)


def _normalize_alias_query(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\([^)]*\)", "", value)
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[\W_]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()
