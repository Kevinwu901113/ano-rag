import json
import os
import time
from collections import defaultdict
from typing import Dict, List


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

    def add_note(self, note: Dict) -> None:
        note_id = note["note_id"]
        subj, obj, pred = note["subj"], note["obj"], note["pred"]

        self.entity_to_notes[subj].append(note_id)
        self.entity_to_notes[obj].append(note_id)
        self.predicate_to_notes[pred].append(note_id)

        domain = (note.get("meta", {}) or {}).get("domain")
        if domain:
            self.domain_index[domain].append(note_id)

        self.graph_edges[subj].append((pred, obj, note_id))
        self.inverse_edges[obj].append((pred, subj, note_id))

        type_key = (note["subj_type"], pred, note["obj_type"])
        self.type_edge_index[type_key].append(note_id)

        attribute = (note.get("meta") or {}).get("attribute") or {}
        attr_name = attribute.get("name") or pred
        values = attribute.get("values") or []
        for value in values:
            if isinstance(value, dict):
                normalized = value.get("normalized") or value.get("value")
            else:
                normalized = value
            if not normalized:
                continue
            normalized_key = str(normalized).strip()
            if not normalized_key:
                continue
            self.field_index[attr_name][normalized_key].append(note_id)

        subject_profile = (note.get("meta") or {}).get("subject_profile") or {}
        for alias in subject_profile.get("aliases") or []:
            alias_key = alias.strip().lower()
            if not alias_key:
                continue
            if note["subj"] not in self.alias_to_entities[alias_key]:
                self.alias_to_entities[alias_key].append(note["subj"])

    def build_from_jsonl(self, notes_path: str) -> None:
        with open(notes_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                note = json.loads(line)
                self.add_note(note)

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
            },
            "version": 1,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
