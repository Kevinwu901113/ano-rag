from __future__ import annotations

import re
from typing import List, Optional

from .ir import PredicateStep, QueryIR, Seed


# 问句类型：保留句首锚定，同时增加非句首的宽松匹配
QUESTION_TYPE_PATTERNS = {
    "who": [re.compile(r"^\s*who\b", re.I), re.compile(r"\bwho\b", re.I)],
    "where": [re.compile(r"^\s*where\b", re.I), re.compile(r"\bwhere\b", re.I)],
    "when": [re.compile(r"^\s*when\b", re.I), re.compile(r"\bwhen\b", re.I)],
    "what": [re.compile(r"^\s*what\b", re.I), re.compile(r"\bwhat\b", re.I)],
    "which": [re.compile(r"^\s*which\b", re.I), re.compile(r"\bwhich\b", re.I)],
    "whom": [re.compile(r"^\s*whom\b", re.I), re.compile(r"\bwhom\b", re.I)],
    "whose": [re.compile(r"^\s*whose\b", re.I), re.compile(r"\bwhose\b", re.I)],
    "how": [re.compile(r"^\s*how\b", re.I), re.compile(r"\bhow\b", re.I)],
    "how_many": [re.compile(r"^\s*how\s+many\b", re.I), re.compile(r"\bhow\s+many\b", re.I)],
    "in_which": [re.compile(r"^\s*in\s+which\b", re.I), re.compile(r"\bin\s+which\b", re.I)],
    "what_is": [re.compile(r"^\s*what\s+(?:is|was|are|’s|'s)\b", re.I), re.compile(r"\bwhat\s+(?:is|was|are|’s|'s)\b", re.I)],
    "name_of": [re.compile(r"^\s*name\s+of\b", re.I), re.compile(r"\bname\s+of\b", re.I)],
    "title_of": [re.compile(r"^\s*title\s+of\b", re.I), re.compile(r"\btitle\s+of\b", re.I)],
    "capital_of": [re.compile(r"^\s*capital\s+of\b", re.I), re.compile(r"\bcapital\s+of\b", re.I)],
}


PREDICATE_LIBRARY = [
    {
        "pred": "spouse",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "spouse of", "entity_side": "right"},
            {"text": "husband of", "entity_side": "right"},
            {"text": "wife of", "entity_side": "right"},
            {"text": "married to", "entity_side": "right"},
            {"regex": r"\b(spouse|wife|husband)\s+of\b", "entity_side": "right"},
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
            {"regex": r"\bperformed\s+by\b", "entity_side": "right"},
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
            {"text": "was born in", "entity_side": "right"},
            {"text": "born at", "entity_side": "right"},
            {"text": "born on", "entity_side": "right"},
            {"text": "native of", "entity_side": "right"},
            {"regex": r"\b(?:was\s+)?born\s+(?:in|at|on)\b", "entity_side": "right"},
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
            {"text": "author of", "entity_side": "right"},
            {"text": "wrote", "entity_side": "right"},
            {"regex": r"\b(wrote|written\s+by|author(?:ed)?\s+of)\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "authored_by",
        "direction": "in",
        "target_type": "WORK",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "wrote", "entity_side": "left"},
            {"text": "author of", "entity_side": "left"},
            {"regex": r"\b(wrote|author(?:ed)?\s+of)\b", "entity_side": "left"},
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
            {"regex": r"\blocated\s+in\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "located_in",
        "direction": "in",
        "target_type": "PLACE",
        "seed_type": "PLACE",
        "aliases": [
            {"text": "capital of", "entity_side": "right"},
            {"regex": r"\bcapital\s+of\b", "entity_side": "right"},
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
            {"regex": r"\bmember\s+of\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "part_of",
        "direction": "out",
        "target_type": "ORG",
        "seed_type": None,
        "aliases": [
            {"text": "part of", "entity_side": "right"},
            {"regex": r"\bpart\s+of\b", "entity_side": "right"},
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
            {"regex": r"\bfounded\s+by\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "parent",
        "direction": "in",
        "target_type": "PERSON",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "parent of", "entity_side": "right"},
            {"text": "father of", "entity_side": "right"},
            {"text": "mother of", "entity_side": "right"},
            {"regex": r"\b(parent|father|mother)\s+of\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "parent",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "child of", "entity_side": "right"},
            {"text": "son of", "entity_side": "right"},
            {"text": "daughter of", "entity_side": "right"},
            {"regex": r"\b(child|son|daughter)\s+of\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "acted_in",
        "direction": "out",
        "target_type": "WORK",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "acted in", "entity_side": "left"},
            {"text": "starred in", "entity_side": "left"},
            {"text": "played", "entity_side": "left"},
            {"text": "portrayed", "entity_side": "left"},
            {"regex": r"\b(acted\s+in|starred\s+in|played|portrayed)\b", "entity_side": "left"},
        ],
    },
    {
        "pred": "acted_in",
        "direction": "in",
        "target_type": "PERSON",
        "seed_type": "WORK",
        "aliases": [
            {"text": "acted in", "entity_side": "right"},
            {"text": "starred in", "entity_side": "right"},
            {"regex": r"\b(acted\s+in|starred\s+in)\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "directed_by",
        "direction": "out",
        "target_type": "PERSON",
        "seed_type": "WORK",
        "aliases": [
            {"text": "directed by", "entity_side": "right"},
            {"text": "director of", "entity_side": "right"},
            {"regex": r"\b(directed\s+by|director\s+of|who\s+directed)\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "directed_by",
        "direction": "in",
        "target_type": "WORK",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "directed", "entity_side": "left"},
            {"text": "directed by", "entity_side": "left"},
            {"regex": r"\b(directed\s+by|directed)\b", "entity_side": "left"},
        ],
    },
    {
        "pred": "occupation",
        "direction": "out",
        "target_type": "CONCEPT",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "occupation", "entity_side": "right"},
            {"text": "profession", "entity_side": "right"},
            {"text": "works as", "entity_side": "right"},
            {"text": "occupation of", "entity_side": "right"},
            {"regex": r"\b(occupation|profession|works\s+as)\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "title",
        "direction": "out",
        "target_type": "CONCEPT",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "title", "entity_side": "right"},
            {"text": "title of", "entity_side": "right"},
            {"text": "position", "entity_side": "right"},
            {"text": "role", "entity_side": "right"},
            {"text": "served as", "entity_side": "right"},
            {"text": "office of", "entity_side": "right"},
            {"regex": r"\b(title\s+of|position|role|served\s+as|office\s+of)\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "nationality",
        "direction": "out",
        "target_type": "PLACE",
        "seed_type": "PERSON",
        "aliases": [
            {"text": "nationality", "entity_side": "right"},
            {"text": "citizenship", "entity_side": "right"},
            {"text": "from which country", "entity_side": "right"},
            {"regex": r"\b(nationality|citizenship|from\s+which\s+country)\b", "entity_side": "right"},
        ],
    },
    {
        "pred": "headquartered_in",
        "direction": "out",
        "target_type": "PLACE",
        "seed_type": "ORG",
        "aliases": [
            {"text": "headquartered in", "entity_side": "right"},
            {"text": "headquarters of", "entity_side": "right"},
            {"regex": r"\b(headquartered\s+in|headquarters\s+of)\b", "entity_side": "right"},
        ],
    },
]


COMPOSITE_RULES = [
    {
        "pattern": re.compile(r"spouse of (?:the )?(?P<entity>.+?) performer", re.I),
        "seed_type": "WORK",
        "chain": [
            PredicateStep(pred="performed_by", direction="out", target_hint="PERSON"),
            PredicateStep(pred="spouse", direction="out", target_hint="PERSON"),
        ],
        "target_type": "PERSON",
    },
]


from .intent_detector import AnswerIntentDetector
from schema.note_schema_v1 import ALLOWED_PREDICATES


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

    # 通过 intent 注入可推断的单跳链（即使手写谓词未命中）
    injected_step = None
    injected_target_type = None
    intent_detector = AnswerIntentDetector()
    detected = intent_detector.detect(question)
    canonical_attr = detected.attribute

    seeds = [Seed(text=entity, type_hint=(pred_match.get("seed_type") if pred_match else detected.entity_type))]
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
    elif canonical_attr and canonical_attr in ALLOWED_PREDICATES:
        # 基于属性推断，构造单跳链，避免空链
        default_direction = "out"
        default_target = None
        if canonical_attr == "authored_by":
            default_direction = "out"
            default_target = "PERSON"
        elif canonical_attr == "acted_in":
            # 人 → 作品 为 out；若问题更像 "Who starred in X"，entity_type=WORK 时走反向
            default_direction = "in" if (detected.entity_type == "WORK") else "out"
            default_target = "WORK" if default_direction == "out" else "PERSON"
        elif canonical_attr == "spouse":
            default_direction = "out"
            default_target = "PERSON"
        elif canonical_attr == "born_in":
            default_direction = "out"
            default_target = "PLACE"
        elif canonical_attr == "located_in":
            default_direction = "out"
            default_target = "PLACE"
        elif canonical_attr == "member_of":
            default_direction = "out"
            default_target = "ORG"
        elif canonical_attr == "founded_by":
            default_direction = "out"
            default_target = "PERSON"
        elif canonical_attr == "parent":
            default_direction = "out"
            default_target = "PERSON"
        elif canonical_attr in {"occupation", "title", "nationality", "born_on", "died_on", "headquartered_in"}:
            default_direction = "out"
            default_target = None
        injected_step = PredicateStep(pred=canonical_attr, direction=default_direction, target_hint=default_target)
        chain.append(injected_step)
        target_type = default_target
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
    # 首先尝试句首锚定，若未命中再尝试宽松匹配
    for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
        if not isinstance(patterns, list):
            patterns = [patterns]
        if patterns and patterns[0].search(question):
            return qtype
    for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
        if not isinstance(patterns, list):
            patterns = [patterns]
        for pattern in patterns[1:]:
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
            # 正则别名优先匹配，其次字面包含；均采用最长匹配优先
            if "regex" in alias and alias["regex"]:
                try:
                    pattern = re.compile(alias["regex"], re.I)
                except Exception:
                    pattern = None
                if pattern:
                    m = pattern.search(question)
                    if m:
                        start, end = m.start(), m.end()
                        alias_str = question[start:end]
                        candidate = {
                            "pred": entry["pred"],
                            "direction": entry.get("direction", "out"),
                            "target_type": entry.get("target_type"),
                            "seed_type": entry.get("seed_type"),
                            "alias": alias_str,
                            "start": start,
                            "end": end,
                            "entity_side": alias.get("entity_side", "right"),
                        }
                        if (not best) or ((end - start) > len(best.get("alias", ""))):
                            best = candidate
                        continue
            alias_text = (alias.get("text") or "").lower()
            if not alias_text:
                continue
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
            if not best or len(alias_text) > len(best.get("alias", "")):
                best = candidate
    return best


def _extract_entity(question: str, pred_match: Optional[dict]) -> Optional[str]:
    if not pred_match:
        return None
    side = pred_match.get("entity_side", "right")
    if side == "right":
        span = question[pred_match["end"] :]
        cleaned = _clean_entity_span(span, take_tail=False)
        if cleaned:
            return cleaned
        # 右侧抽空兜底：试一次左侧
        span_left = question[: pred_match["start"]]
        return _clean_entity_span(span_left, take_tail=True)
    span = question[: pred_match["start"]]
    cleaned = _clean_entity_span(span, take_tail=True)
    if cleaned:
        return cleaned
    # 左侧抽空兜底：试右侧
    span_right = question[pred_match["end"] :]
    return _clean_entity_span(span_right, take_tail=False)


def _extract_fallback_entity(question: str) -> Optional[str]:
    quoted = re.search(r'["“”\']([^"“”\']+)["“”\']', question)
    if quoted:
        return quoted.group(1).strip()
    tail_match = re.search(r"(?:of|about|for|from|regarding)\s+([^?]+)$", question, re.I)
    if tail_match:
        return tail_match.group(1).strip(" ?.,")
    capital = re.findall(r"([A-Z][A-Za-z0-9'&\-]+(?:\s+[A-Z][A-Za-z0-9'&\-]+)*)", question)
    if capital:
        candidate = capital[-1].strip()
        candidate = re.sub(r"(’s|'s)\b$", "", candidate).strip()
        return candidate
    return None


def _clean_entity_span(span: str, take_tail: bool) -> Optional[str]:
    if not span:
        return None
    cleaned = span.strip()
    if not cleaned:
        return None
    # 去成对引号/括号（含中英文）
    pairs = {
        '"': '"', "'": "'", "“": "”", "‘": "’", "(": ")", "[": "]", "{": "}", "《": "》", "「": "」"
    }
    changed = True
    while changed and cleaned:
        changed = False
        for left, right in pairs.items():
            if cleaned.startswith(left) and cleaned.endswith(right) and len(cleaned) > 2:
                cleaned = cleaned[1:-1].strip()
                changed = True
    # 被动语序优先：by 之后的1-6个token
    by_match = re.search(r"\bby\b\s+(.+)$", cleaned, re.I)
    by_taken = None
    if by_match:
        tail = by_match.group(1).strip()
        by_tokens = _tokenize_with_stops(tail)
        if by_tokens:
            by_taken = " ".join(by_tokens[:6]).strip()
    # 停用词分割，保留靠近alias一侧的最长名词短语（5–8 tokens）
    lowered = cleaned.lower()
    # 第一个强分隔符前截断
    separators = [",", "?", ";", ":"]
    cut_idx = min((lowered.find(sep) for sep in separators if sep in lowered), default=-1)
    if cut_idx > 0:
        cleaned = cleaned[:cut_idx]
    tokens = _tokenize_with_stops(cleaned)
    if not tokens:
        return by_taken
    # 去除前导系动词/助动词
    leading_stops = {"is", "was", "are", "do", "does", "did"}
    while tokens and tokens[0].lower().strip("?.,") in leading_stops:
        tokens = tokens[1:]
    window = 8 if not take_tail else 6
    tokens = tokens[-window:] if take_tail else tokens[:window]
    # 去尾随动词/虚词
    trailing_stop = {"born", "located", "written", "wrote", "write", "called", "named"}
    while tokens and tokens[-1].lower().strip("?.,") in trailing_stop:
        tokens = tokens[:-1]
    result = " ".join(tokens).strip(" ?.,;:")
    if not result and by_taken:
        return by_taken
    return result or None


def _tokenize_with_stops(text: str) -> List[str]:
    raw_tokens = re.split(r"\s+", (text or "").strip())
    stops = {
        ",", "?", ".", "but", "when", "who", "which", "what", "how",
        "by", "in", "on", "at", "from", "for", "to", "is", "was", "are", "do", "does", "did"
    }
    tokens: List[str] = []
    for t in raw_tokens:
        tt = t.strip()
        if not tt:
            continue
        if tt.lower().strip("?.,") in stops:
            break
        tokens.append(tt)
    return tokens
