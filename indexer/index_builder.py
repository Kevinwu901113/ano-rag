import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import re
import unicodedata

from schema.note_schema_v1 import PRED2ATTR
from utils import TextUtils


class IndexBuilder:
    def __init__(self) -> None:
        self.entity_to_notes: Dict[str, List[str]] = defaultdict(list)
        self.predicate_to_notes: Dict[str, List[str]] = defaultdict(list)
        self.domain_index: Dict[str, List[str]] = defaultdict(list)
        self.graph_edges = defaultdict(list)
        self.inverse_edges = defaultdict(list)
        self.type_edge_index = defaultdict(list)
        self.field_index = defaultdict(lambda: defaultdict(list))
        self.alias_to_entities = defaultdict(list)
        self.weak_entity_to_notes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.weak_predicate_to_notes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Mentions and coreference edges (note-centric)
        self.mentions_edges = defaultdict(list)  # note_id -> [entity]
        self.corefers_edges = defaultdict(list)  # note_id -> [entity]
        # Anchor index: note_id -> anchor_entity
        self.anchor_index = {}

    def add_note(self, note: Dict) -> None:
        note_id = note["note_id"]
        subj, obj, pred = note["subj"], note["obj"], note["pred"]
        meta = (note.get("meta") or {})

        if meta.get("filter_out_strict"):
            return

        self.entity_to_notes[subj].append(note_id)
        self.entity_to_notes[obj].append(note_id)
        self.predicate_to_notes[pred].append(note_id)

        domain = meta.get("domain")
        if domain:
            self.domain_index[domain].append(note_id)

        self.graph_edges[subj].append((pred, obj, note_id))
        self.inverse_edges[obj].append((pred, subj, note_id))

        type_key = (note["subj_type"], pred, note["obj_type"])
        self.type_edge_index[type_key].append(note_id)

        attribute = meta.get("attribute") or {}
        raw_attr_name = attribute.get("name") or pred
        # 谓词归一到属性名（例如 profession/job → occupation）
        attr_name = PRED2ATTR.get((raw_attr_name or "").lower(), raw_attr_name)
        values = attribute.get("values") or []
        for value in values:
            if isinstance(value, dict):
                normalized = value.get("normalized") or value.get("value")
            else:
                normalized = value
            if not normalized:
                continue
            # occupation 的值统一归一：去括号/标点/多空格、大小写折叠、职业性别合并
            normalized_key = self._normalize_field_value(attr_name, str(normalized))
            if not normalized_key:
                continue
            # 写入 raw 与 stemmed 两种键，提升命中率
            self.field_index[attr_name][normalized_key].append(note_id)
            stemmed_key = self._stem_occupation(normalized_key) if attr_name == "occupation" else normalized_key
            if stemmed_key and stemmed_key != normalized_key:
                self.field_index[attr_name][stemmed_key].append(note_id)

        subject_profile = meta.get("subject_profile") or {}
        for alias in subject_profile.get("aliases") or []:
            alias_key = self._normalize_alias(alias)
            if not alias_key:
                continue
            if note["subj"] not in self.alias_to_entities[alias_key]:
                self.alias_to_entities[alias_key].append(note["subj"])
        # Merge alias_map from meta into alias index
        alias_map = meta.get("alias_map") or {}
        if isinstance(alias_map, dict):
            for canonical, aliases in alias_map.items():
                for alias in aliases or []:
                    alias_key = self._normalize_alias(alias)
                    if not alias_key:
                        continue
                    if canonical not in self.alias_to_entities[alias_key]:
                        self.alias_to_entities[alias_key].append(canonical)

        # Build MENTIONS edges from evidence text
        evidence = (note.get("evidence") or "").strip()
        if evidence:
            for entity in TextUtils.extract_entity_candidates(evidence):
                # 去噪：丢弃或强降权仅含单字母/破碎 token 的 mention（如 O、S）
                if not entity or len(entity.strip()) <= 1:
                    continue
                if entity not in self.mentions_edges[note_id]:
                    self.mentions_edges[note_id].append(entity)

        # Build COREFERS_TO edge: note -> subject entity
        if subj:
            if subj not in self.corefers_edges[note_id]:
                self.corefers_edges[note_id].append(subj)

        # Anchor entity from meta if present and unique
        anchor = meta.get("anchor_entity")
        if isinstance(anchor, str) and anchor.strip():
            self.anchor_index[note_id] = anchor.strip()

    def add_weak_note(self, note: Dict[str, Any]) -> None:
        note_id = note.get("note_id")
        if not note_id:
            return
        meta = (note.get("meta") or {}) or {}
        candidates = meta.get("coref_candidates") or []
        if isinstance(candidates, list):
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                entity = cand.get("entity")
                if not entity:
                    continue
                try:
                    score = float(cand.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                weight = max(0.0, min(score * 0.5, 1.0))
                self.weak_entity_to_notes[entity].append({"note_id": note_id, "weight": round(weight, 3)})
        pred = note.get("pred")
        if pred:
            try:
                base_conf = float(meta.get("coref_confidence") or 0.3)
            except (TypeError, ValueError):
                base_conf = 0.3
            weight = max(0.0, min(base_conf * 0.5, 1.0))
            self.weak_predicate_to_notes[pred].append({"note_id": note_id, "weight": round(weight, 3)})

    @staticmethod
    def _normalize_alias(text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        # 去括号内容
        value = re.sub(r"\([^)]*\)", "", value)
        # 去中划线/点号
        value = value.replace("-", " ").replace(".", " ")
        # Unicode 规范化
        value = unicodedata.normalize("NFKC", value)
        # 多空格归一
        value = re.sub(r"\s+", " ", value)
        return value.strip().lower()

    @staticmethod
    def _normalize_field_value(attr_name: str, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        # 去括号内容、标点和多空格；大小写折叠
        cleaned = re.sub(r"\([^)]*\)", "", raw)
        cleaned = re.sub(r"[\p{Punct}]", " ", cleaned) if hasattr(re, "P") else re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = unicodedata.normalize("NFKC", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
        if attr_name == "occupation":
            # 常见职业词典：actor/actress → actor 等
            if cleaned in {"actress"}:
                return "actor"
            if cleaned in {"comics artist", "comic artist", "cartoon artist"}:
                return "cartoonist"
        return cleaned

    @staticmethod
    def _stem_occupation(text: str) -> str:
        if not text:
            return ""
        # 简单词干处理：复数/性别统一
        mapping = {
            "actors": "actor",
            "actresses": "actor",
            "singers": "singer",
            "writers": "writer",
            "authors": "author",
            "cartoonists": "cartoonist",
        }
        return mapping.get(text, text)

    def build_from_jsonl(self, notes_path: str) -> None:
        with open(notes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                note = json.loads(line)
                self.add_note(note)

    def build_weak_from_jsonl(self, weak_notes_path: Optional[str]) -> None:
        if not weak_notes_path or not os.path.exists(weak_notes_path):
            return
        with open(weak_notes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    note = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.add_weak_note(note)

    def dump(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        def dump_json(obj: Dict, name: str) -> None:
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
                json.dump(obj, handle, ensure_ascii=False)

        dump_json(self.entity_to_notes, "entity_to_notes.json")
        dump_json(self.predicate_to_notes, "predicate_to_notes.json")
        dump_json(self.domain_index, "domain_index.json")
        converted = {"|".join(key): value for key, value in self.type_edge_index.items()}
        dump_json(converted, "type_edge_index.json")
        field_index_serializable = {
            attr: {val: ids for val, ids in values.items()} for attr, values in self.field_index.items()
        }
        dump_json(field_index_serializable, "field_index.json")
        dump_json(self.alias_to_entities, "entity_alias_index.json")
        dump_json(self.anchor_index, "anchor_index.json")
        dump_json(dict(self.weak_entity_to_notes), "weak_entity_to_notes.json")
        dump_json(dict(self.weak_predicate_to_notes), "weak_predicate_to_notes.json")

        graph_path = os.path.join(out_dir, "graph_edges.jsonl")
        with open(graph_path, "w", encoding="utf-8") as handle:
            for subject, edges in self.graph_edges.items():
                handle.write(
                    json.dumps({"subj": subject, "edges": edges}, ensure_ascii=False)
                    + "\n"
                )

        inverse_path = os.path.join(out_dir, "inverse_edges.jsonl")
        with open(inverse_path, "w", encoding="utf-8") as handle:
            for obj, edges in self.inverse_edges.items():
                handle.write(
                    json.dumps({"obj": obj, "edges": edges}, ensure_ascii=False) + "\n"
                )

        mentions_path = os.path.join(out_dir, "mentions_edges.jsonl")
        with open(mentions_path, "w", encoding="utf-8") as handle:
            for nid, entities in self.mentions_edges.items():
                handle.write(json.dumps({"note_id": nid, "mentions": entities}, ensure_ascii=False) + "\n")

        corefers_path = os.path.join(out_dir, "corefers_edges.jsonl")
        with open(corefers_path, "w", encoding="utf-8") as handle:
            for nid, entities in self.corefers_edges.items():
                handle.write(json.dumps({"note_id": nid, "corefers_to": entities}, ensure_ascii=False) + "\n")

        note_ids = {nid for notes in self.entity_to_notes.values() for nid in notes}

        manifest = {
            "generated_at": int(time.time()),
            "counts": {
                "entities": len(self.entity_to_notes),
                "predicates": len(self.predicate_to_notes),
                "notes": len(note_ids),
            },
            "files": {
                "entity_to_notes": "entity_to_notes.json",
                "predicate_to_notes": "predicate_to_notes.json",
                "domain_index": "domain_index.json",
                "type_edge_index": "type_edge_index.json",
                "graph_edges": "graph_edges.jsonl",
                "inverse_edges": "inverse_edges.jsonl",
                "field_index": "field_index.json",
                "entity_alias_index": "entity_alias_index.json",
                "mentions_edges": "mentions_edges.jsonl",
                "corefers_edges": "corefers_edges.jsonl",
                "anchor_index": "anchor_index.json",
                "weak_entity_to_notes": "weak_entity_to_notes.json",
                "weak_predicate_to_notes": "weak_predicate_to_notes.json",
            },
            "version": 1,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
