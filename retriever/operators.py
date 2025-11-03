import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple


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


def BIND(indexes: Indexes, alias: str, type_candidates: List[str]) -> List[str]:
    matches: List[str] = []
    target = alias.lower()
    for entity in indexes.entity_to_notes.keys():
        if target in entity.lower():
            matches.append(entity)
        if len(matches) >= 50:
            break
    return matches


def EXPAND_from(
    indexes: Indexes, subject: str, predicate: str
) -> List[Tuple[str, str]]:
    output: List[Tuple[str, str]] = []
    for pred, obj, note_id in indexes.graph_edges.get(subject, []):
        if pred == predicate:
            output.append((obj, note_id))
        if len(output) >= 200:
            break
    return output
