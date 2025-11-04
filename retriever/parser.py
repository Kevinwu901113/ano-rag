from __future__ import annotations

import re
from typing import List, Optional

from .ir import PredicateStep, QueryIR, Seed


QUESTION_TYPE_PATTERNS = {
    "who": re.compile(r"^\s*who\b", re.I),
    "where": re.compile(r"^\s*where\b", re.I),
    "when": re.compile(r"^\s*when\b", re.I),
    "what": re.compile(r"^\s*what\b", re.I),
}


PREDICATE_LIBRARY = [
    {
        "pred": "spouse_of",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "spouse of", "entity_side": "right"},
            {"text": "husband of", "entity_side": "right"},
            {"text": "wife of", "entity_side": "right"},
            {"text": "married to", "entity_side": "right"},
        ],
    },
    {
        "pred": "performed_by",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "WORK",
        "aliases": [
            {"text": "performed by", "entity_side": "right"},
            {"text": "performer of", "entity_side": "right"},
        ],
    },
    {
        "pred": "born_in",
        "direction": "out",
        "target_type": "PLACE",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "born in", "entity_side": "right"},
            {"text": "birthplace of", "entity_side": "right"},
            {"text": "where was", "entity_side": "right"},
        ],
    },
    {
        "pred": "authored_by",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "WORK",
        "aliases": [
            {"text": "written by", "entity_side": "right"},
            {"text": "authored by", "entity_side": "right"},
        ],
    },
    {
        "pred": "wrote",
        "direction": "out",
        "target_type": "WORK",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "wrote", "entity_side": "right"},
            {"text": "author of", "entity_side": "right"},
        ],
    },
    {
        "pred": "located_in",
        "direction": "out",
        "target_type": "PLACE",
        "seed_type": None,
        "aliases": [
            {"text": "located in", "entity_side": "right"},
            {"text": "in which city", "entity_side": "right"},
            {"text": "in which country", "entity_side": "right"},
        ],
    },
    {
        "pred": "member_of",
        "direction": "out",
        "target_type": "ORG",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "member of", "entity_side": "right"},
            {"text": "belonged to", "entity_side": "right"},
        ],
    },
    {
        "pred": "founded_by",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "ORG",
        "aliases": [
            {"text": "founded by", "entity_side": "right"},
            {"text": "founder of", "entity_side": "right"},
        ],
    },
    {
        "pred": "parent_of",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "parent of", "entity_side": "right"},
            {"text": "father of", "entity_side": "right"},
            {"text": "mother of", "entity_side": "right"},
        ],
    },
    {
        "pred": "child_of",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "child of", "entity_side": "right"},
            {"text": "son of", "entity_side": "right"},
            {"text": "daughter of", "entity_side": "right"},
        ],
    },
    {
        "pred": "director_of",
        "direction": "out",
        "target_type": "WORK",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "directed", "entity_side": "right"},
            {"text": "director of", "entity_side": "right"},
        ],
    },
    {
        "pred": "starred_in",
        "direction": "out",
        "target_type": "WORK",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "starred in", "entity_side": "right"},
            {"text": "acted in", "entity_side": "right"},
        ],
    },
]


COMPOSITE_RULES = [
    {
        "pattern": re.compile(r"spouse of (?:the )?(?P<entity>.+?) performer", re.I),
        "seed_type": "WORK",
        "chain": [
            PredicateStep(pred="performed_by", direction="out", target_hint="PERSON"),
            PredicateStep(pred="spouse_of", direction="out", target_hint="PERSON"),
        ],
        "target_type": "PERSON",
    },
]


def parse_question(question: str) -> Optional[QueryIR]:
    text = (question or "").strip()
    if not text:
        return None

    question_type = _detect_question_type(text)

    composite = _match_composite(text)
    if composite:
        seeds = [Seed(text=composite["entity"], type_hint=composite.get("seed_type"))]
        return QueryIR(
            intent="relation_query",
            seeds=seeds,
            pred_chain=composite["chain"],
            target_type=composite.get("target_type"),
            question_type=question_type,
            max_hops=len(composite["chain"]),
            fanout=12,
            raw=question,
        )

    pred_match = _match_predicate(text)
    entity = _extract_entity(text, pred_match)
    if not entity:
        entity = _extract_fallback_entity(text)

    if not entity:
        return None

    seeds = [Seed(text=entity, type_hint=pred_match.get("seed_type") if pred_match else None)]
    chain: List[PredicateStep] = []
    target_type = None
    if pred_match:
        chain.append(
            PredicateStep(
                pred=pred_match["pred"],
                direction=pred_match.get("direction", "out"),
                target_hint=pred_match.get("target_type"),
            )
        )
        target_type = pred_match.get("target_type")
    intent = "relation_query" if chain else "open_entity_query"
    return QueryIR(
        intent=intent,
        seeds=seeds,
        pred_chain=chain,
        target_type=target_type,
        question_type=question_type,
        max_hops=max(1, len(chain) or 1),
        fanout=15 if not chain else 12,
        raw=question,
        fallback=not chain,
    )


def _detect_question_type(question: str) -> Optional[str]:
    for qtype, pattern in QUESTION_TYPE_PATTERNS.items():
        if pattern.search(question):
            return qtype
    return None


def _match_composite(question: str) -> Optional[dict]:
    for rule in COMPOSITE_RULES:
        match = rule["pattern"].search(question)
        if match:
            entity = match.group("entity").strip(" ?.,")
            return {
                "entity": entity,
                "chain": rule["chain"],
                "seed_type": rule.get("seed_type"),
                "target_type": rule.get("target_type"),
            }
    return None


def _match_predicate(question: str) -> Optional[dict]:
    lowered = question.lower()
    best = None
    for entry in PREDICATE_LIBRARY:
        for alias in entry["aliases"]:
            alias_text = alias["text"].lower()
            idx = lowered.find(alias_text)
            if idx == -1:
                continue
            candidate = {
                "pred": entry["pred"],
                "direction": entry.get("direction", "out"),
                "target_type": entry.get("target_type"),
                "seed_type": entry.get("seed_type"),
                "alias": alias_text,
                "start": idx,
                "end": idx + len(alias_text),
                "entity_side": alias.get("entity_side", "right"),
            }
            if not best or len(alias_text) > len(best["alias"]):
                best = candidate
    return best


def _extract_entity(question: str, pred_match: Optional[dict]) -> Optional[str]:
    if not pred_match:
        return None
    if pred_match.get("entity_side", "right") == "right":
        span = question[pred_match["end"] :]
        return _clean_entity_span(span, take_tail=False)
    span = question[: pred_match["start"]]
    return _clean_entity_span(span, take_tail=True)


def _extract_fallback_entity(question: str) -> Optional[str]:
    quoted = re.search(r'["“”\']([^"“”\']+)["“”\']', question)
    if quoted:
        return quoted.group(1).strip()
    tail_match = re.search(r"(?:of|about|for|from|regarding)\s+([^?]+)$", question, re.I)
    if tail_match:
        return tail_match.group(1).strip(" ?.,")
    capital = re.findall(r"([A-Z][A-Za-z0-9'&\-]+(?:\s+[A-Z][A-Za-z0-9'&\-]+)*)", question)
    if capital:
        return capital[-1].strip()
    return None


def _clean_entity_span(span: str, take_tail: bool) -> Optional[str]:
    if not span:
        return None
    cleaned = span.strip(" ?.,;:\n")
    if not cleaned:
        return None
    separators = [",", "?", " and ", " but ", " when ", " who ", " which ", " that "]
    for sep in separators:
        idx = cleaned.lower().find(sep)
        if idx > 0:
            cleaned = cleaned[:idx]
            break
    tokens = cleaned.split()
    if not tokens:
        return None
    if take_tail and len(tokens) > 6:
        tokens = tokens[-6:]
    trailing_stop = {"born", "born?", "located", "located?", "written", "wrote", "write", "called", "named"}
    while tokens and tokens[-1].lower().strip("?.,") in trailing_stop:
        tokens = tokens[:-1]
    return " ".join(tokens).strip(" ?.,") or None
