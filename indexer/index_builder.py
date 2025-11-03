import json
import os
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
