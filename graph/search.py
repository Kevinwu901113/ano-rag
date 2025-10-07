from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import config


@dataclass
class Path:
    keys: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    rels: List[str] = field(default_factory=list)
    score: float = 0.0

    def last_key(self) -> str:
        return self.keys[-1] if self.keys else ""

    def extend(self, nb_key: str, rel: str, note_id: str, delta: float) -> "Path":
        return Path(
            keys=self.keys + [nb_key],
            notes=self.notes + [note_id],
            rels=self.rels + [rel],
            score=self.score + float(delta),
        )


def beam_search(graph, anchors: List[str], rel_chain: List[str]) -> List[Path]:
    max_hops = int(config.get("multi_hop.max_hops", 4))
    beam_size = int(config.get("multi_hop.beam_size", 8))
    branch = int(config.get("multi_hop.branch_factor", 6))

    valid_anchors = [a for a in anchors if a]
    if not valid_anchors:
        return []

    beams = [Path(keys=[anchor], notes=[], rels=[], score=0.0) for anchor in valid_anchors]
    results: List[Path] = []

    for hop in range(max_hops):
        next_candidates: List[Path] = []
        for path in beams:
            current_key = path.last_key()
            if not current_key:
                continue
            neighbors = graph.neighbors(current_key)
            for rel, nb_key, note_id, weight, _para in neighbors:
                if rel_chain and hop < len(rel_chain):
                    constraint = rel_chain[hop]
                    if constraint not in ("*", rel):
                        continue
                if nb_key in path.keys:
                    continue
                next_candidates.append(path.extend(nb_key, rel, note_id, weight))

        if not next_candidates:
            break

        next_candidates.sort(key=lambda p: p.score, reverse=True)

        pruned: List[Path] = []
        bucket: Dict[Tuple[str, str], int] = {}
        for candidate in next_candidates:
            prev_key = candidate.keys[-2] if len(candidate.keys) >= 2 else ""
            last_rel = candidate.rels[-1] if candidate.rels else ""
            bucket_key = (prev_key, last_rel)
            bucket.setdefault(bucket_key, 0)
            if bucket[bucket_key] >= max(1, branch):
                continue
            bucket[bucket_key] += 1
            pruned.append(candidate)
            if len(pruned) >= beam_size:
                break

        beams = pruned
        results.extend(pruned)
        if not beams:
            break

    results.sort(key=lambda p: p.score, reverse=True)
    return results[:beam_size]
