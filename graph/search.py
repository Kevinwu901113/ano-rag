from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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


def _constraint_matches(rel: str, constraint: str) -> bool:
    if not constraint or constraint == "*":
        return True
    options = [c.strip() for c in constraint.split("|") if c.strip()]
    return rel in options


def beam_search(
    graph,
    anchors: List[str],
    rel_chain: Optional[Sequence[str]] = None,
    max_hops: Optional[int] = None,
    beam_size: Optional[int] = None,
    branch: Optional[int] = None,
) -> List[Path]:
    config_hops = config.get("multi_hop", {}) or {}
    max_hops = int(max_hops or config_hops.get("max_hops", 4))
    beam_size = int(beam_size or config_hops.get("beam_size", 8))
    branch = int(branch or config_hops.get("branch_factor", 6))

    valid_anchors = [a for a in anchors if a]
    if not valid_anchors:
        return []

    beams = [Path(keys=[anchor], notes=[], rels=[], score=0.0) for anchor in valid_anchors]
    completed: List[Path] = []

    for _hop in range(max_hops):
        next_candidates: List[Path] = []
        for path in beams:
            current_key = path.last_key()
            if not current_key:
                continue
            neighbors = graph.neighbors(current_key)
            for rel, nb_key, note_id, weight, _para in neighbors:
                rel_index = len(path.rels)
                if rel_chain:
                    if rel_index >= len(rel_chain):
                        continue
                    constraint = rel_chain[rel_index]
                    if not _constraint_matches(rel, constraint):
                        continue
                if nb_key in path.keys:
                    continue
                new_path = path.extend(nb_key, rel, note_id, weight)
                if rel_chain and len(new_path.rels) >= len(rel_chain):
                    completed.append(new_path)
                    continue
                next_candidates.append(new_path)

        if not next_candidates and not completed:
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
        if not beams:
            break

    results = completed if completed else beams
    # Drop degenerate paths that never traversed an edge (no supporting notes).
    results = [path for path in results if path.notes]
    if not results:
        return []

    results.sort(key=lambda p: p.score, reverse=True)
    return results[:beam_size]
