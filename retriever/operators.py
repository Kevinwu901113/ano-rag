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


def BIND(indexes: Indexes, alias: str, type_candidates: List[str], limit: int = 50) -> List[str]:
    matches: List[str] = []
    target = (alias or "").lower()
    if not target:
        return matches

    for entity in indexes.entity_to_notes.keys():
        name = entity.lower()
        if target in name or name in target:
            matches.append(entity)
        elif _loose_match(target, name):
            matches.append(entity)
        if len(matches) >= limit:
            break
    return matches


def EXPAND_from(
    indexes: Indexes,
    entity: str,
    predicate: str,
    direction: str = "out",
    limit: int = 200,
) -> List[Tuple[str, str]]:
    output: List[Tuple[str, str]] = []
    if direction == "in":
        edge_source = indexes.inverse_edges.get(entity, [])
        for pred, subj, note_id in edge_source:
            if pred == predicate:
                output.append((subj, note_id))
            if len(output) >= limit:
                break
        return output

    for pred, obj, note_id in indexes.graph_edges.get(entity, []):
        if pred == predicate:
            output.append((obj, note_id))
        if len(output) >= limit:
            break
    return output


def _loose_match(needle: str, hay: str) -> bool:
    if not needle or not hay:
        return False
    tokens = [t for t in needle.split() if len(t) > 3]
    return any(t in hay for t in tokens)
