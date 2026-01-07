from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from relrag.schema.note_schema_v1 import PRED_SYNONYM_SETS


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


ATTRIBUTE_HINTS = [
    {
        "name": "occupation",
        "weight": 0.9,
        "regex": [
            re.compile(r"occupation of (?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"what (?:is|was) (?P<entity>.+?)'?s occupation", re.I),
            re.compile(r"what does (?P<entity>.+?) do for a living", re.I),
            re.compile(r"what does (?P<entity>.+?) do", re.I),
            re.compile(r"what (?P<entity>.+?) does for a living", re.I),
            re.compile(r"what (?P<entity>.+?) works as", re.I),
            re.compile(r"(?P<entity>.+?)\s+的\s*职业", re.I),
            re.compile(r"(?P<entity>.+?)\s+的\s*工作", re.I),
            re.compile(r"做什么职业\s*(?P<entity>.+?)?", re.I),
        ],
        "keywords": ["occupation", "job", "profession", "works as"],
    },
    {
        "name": "title",
        "weight": 0.85,
        "regex": [
            re.compile(r"title of (?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"what (?:is|was) (?P<entity>.+?)'?s title", re.I),
            re.compile(r"which title does (?P<entity>.+?) hold", re.I),
        ],
        "keywords": ["title", "position", "role"],
    },
    {
        "name": "nationality",
        "weight": 0.85,
        "regex": [
            re.compile(r"nationality of (?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"what nationality is (?P<entity>.+?)", re.I),
            re.compile(r"where is (?P<entity>.+?) from", re.I),
        ],
        "keywords": ["nationality", "citizenship", "from which country"],
    },
    {
        "name": "born_on",
        "weight": 0.8,
        "regex": [
            re.compile(r"when was (?P<entity>.+?) born", re.I),
            re.compile(r"birth date of (?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["birth date", "born", "date of birth"],
    },
    {
        "name": "died_on",
        "weight": 0.8,
        "regex": [
            re.compile(r"when did (?P<entity>.+?) die", re.I),
            re.compile(r"death date of (?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["died", "death date", "passed away"],
    },
    {
        "name": "headquartered_in",
        "weight": 0.8,
        "regex": [
            re.compile(r"where is (?P<entity>.+?) headquartered", re.I),
            re.compile(r"headquarters of (?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["headquarters", "headquartered"]
    },
    {
        "name": "authored_by",
        "weight": 0.85,
        "regex": [
            re.compile(r"who\s+(?:wrote|authored)\s+(?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"written\s+by\s+(?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"author\s+of\s+(?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["written by", "wrote", "author of"],
    },
    {
        "name": "born_in",
        "weight": 0.85,
        "regex": [
            re.compile(r"where\s+was\s+(?P<entity>.+?)\s+born", re.I),
            re.compile(r"(?P<entity>.+?)\s+was\s+born\s+(?:in|at|on)\s+.+", re.I),
            re.compile(r"birthplace\s+of\s+(?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["born in", "birthplace", "native of"],
    },
    {
        "name": "spouse",
        "weight": 0.8,
        "regex": [
            re.compile(r"spouse\s+of\s+(?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"who\s+is\s+(?P<entity>.+?)'?s\s+(?:husband|wife|spouse)", re.I),
        ],
        "keywords": ["spouse of", "wife of", "husband of"],
    },
    {
        "name": "directed_by",
        "weight": 0.8,
        "regex": [
            re.compile(r"who\s+directed\s+(?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"directed\s+by\s+(?P<entity>.+?)(?:\?|$)", re.I),
            re.compile(r"director\s+of\s+(?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["directed by", "director of"],
    },
    {
        "name": "acted_in",
        "weight": 0.8,
        "regex": [
            re.compile(r"who\s+(?:starred|acted)\s+in\s+(?P<entity>.+?)(?:\?|$)", re.I),
        ],
        "keywords": ["starred in", "acted in"],
    },
]


ATTRIBUTE_TYPE_HINTS = {
    "occupation": "PERSON",
    "title": "PERSON",
    "nationality": "PERSON",
    "born_on": "PERSON",
    "died_on": "PERSON",
    "headquartered_in": "ORG",
    "member_of": "PERSON",
}


@dataclass
class AnswerIntent:
    entity: Optional[str]
    entity_type: Optional[str]
    attribute: Optional[str]
    question_type: Optional[str]
    confidence: float
    signals: Dict[str, float]

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "attribute": self.attribute,
            "question_type": self.question_type,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
        }


class AnswerIntentDetector:
    def detect(self, question: str) -> AnswerIntent:
        question = (question or "").strip()
        lowered = question.lower()
        signals: Dict[str, float] = {}

        question_type = self._detect_question_type(lowered)
        if question_type:
            signals[f"question_type:{question_type}"] = 0.15

        attr_name, attr_conf, entity_from_attr = self._detect_attribute(lowered, question)
        if attr_name:
            signals[f"attribute:{attr_name}"] = attr_conf

        entity = entity_from_attr or self._extract_entity(question)
        if entity:
            signals["entity_detected"] = 0.2

        entity_type = self._infer_entity_type(attr_name, question_type)
        if entity_type:
            signals[f"entity_type:{entity_type}"] = 0.1

        confidence = 0.4
        for weight in signals.values():
            confidence += weight
        confidence = min(0.95, max(0.0, confidence))

        canonical_attr = self._canonical_predicate(attr_name) if attr_name else None

        return AnswerIntent(
            entity=entity,
            entity_type=entity_type,
            attribute=canonical_attr,
            question_type=question_type,
            confidence=confidence,
            signals=signals,
        )

    @staticmethod
    def _detect_question_type(lowered_question: str) -> Optional[str]:
        # 先尝试句首命中，再尝试宽松命中
        for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
            if not isinstance(patterns, list):
                patterns = [patterns]
            if patterns and patterns[0].search(lowered_question):
                return qtype
        for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
            if not isinstance(patterns, list):
                patterns = [patterns]
            for pattern in patterns[1:]:
                if pattern.search(lowered_question):
                    return qtype
        return None

    def _detect_attribute(self, lowered: str, original: str) -> tuple[Optional[str], float, Optional[str]]:
        for hint in ATTRIBUTE_HINTS:
            for pattern in hint["regex"]:
                match = pattern.search(original)
                if match:
                    entity = (match.group("entity") or "").strip(" ?.,")
                    return hint["name"], hint["weight"], entity
            for keyword in hint.get("keywords", []):
                if keyword in lowered:
                    return hint["name"], hint["weight"] - 0.1, None
        return None, 0.0, None

    def _infer_entity_type(self, attribute: Optional[str], question_type: Optional[str]) -> Optional[str]:
        if attribute and attribute in ATTRIBUTE_TYPE_HINTS:
            return ATTRIBUTE_TYPE_HINTS[attribute]
        if question_type == "who":
            return "PERSON"
        if question_type == "where":
            return "PLACE"
        if question_type == "when":
            return "TIME"
        return None

    def _extract_entity(self, question: str) -> Optional[str]:
        quoted = re.search(r'["“”\']([^"“”\']+)["“”\']', question)
        if quoted:
            return quoted.group(1).strip()

        tail_match = re.search(r"(?:of|about|for|from|regarding)\s+([^?]+)$", question, re.I)
        if tail_match:
            return tail_match.group(1).strip(" ?.,")

        capital = re.findall(r"([A-Z][A-Za-z0-9'&\-]+(?:\s+[A-Z][A-Za-z0-9'&\-]+)*)", question)
        if capital:
            # 使用末尾连续大写 Token 合并为实体候选
            return capital[-1].strip()
        return None

    @staticmethod
    def _canonical_predicate(pred: Optional[str]) -> Optional[str]:
        value = (pred or "").strip().lower()
        if not value:
            return None
        for canon, synonyms in PRED_SYNONYM_SETS.items():
            if value == canon or value in synonyms:
                return canon
        return value

