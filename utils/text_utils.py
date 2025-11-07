import re
from typing import List, Dict


class TextUtils:
    # --- Pronoun lists (non-exhaustive, practical) ---
    EN_PRONOUNS = {
        "he", "she", "it", "they", "him", "her", "them", "his", "her", "its", "their",
        "this", "that", "these", "those", "former", "latter", "the former", "the latter",
    }
    ZH_PRONOUNS = {
        "他", "她", "它", "他们", "她们", "它们", "其", "该", "此", "本", "前者", "后者", "上述",
        "该公司", "该机构", "该团队", "该部门",
    }

    # Common English verb triggers for subject position
    EN_SUBJECT_VERBS = (
        "be", "was", "were", "is", "are", "said", "announced", "joined", "founded", "born",
        "died", "won", "created", "wrote", "worked", "served", "became",
    )
    # Chinese triggers following subject pronouns
    ZH_SUBJECT_TRIGGERS = (
        "于", "在", "为", "是", "曾", "宣布", "加入", "出生", "去世", "，",
    )

    # Entity suffixes (Chinese organizations etc.)
    ZH_ENTITY_SUFFIX = r"(?:公司|集团|大学|学院|研究院|政府|委员会|部|局|厅|办|社|行|银行|法院|检察院|省|市|县|区|处|所|署)"

    # English entity patterns
    EN_ENTITY_PATTERNS = [
        # Capitalized multi-word sequences: "New York Times", allow dots
        re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\.| Inc\.| LLC\.| Ltd\.)?)\b"),
        # Acronyms: NASA, W3C
        re.compile(r"\b([A-Z]{2,})\b"),
    ]

    @staticmethod
    def split_by_sentence(text: str) -> List[str]:
        """Legacy splitter: keep for backward compatibility."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []
        parts = re.split(r"(?<=[。！？.!?])\s+", cleaned)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def split_with_spans(text: str) -> List[Dict]:
        """
        Split text into sentences with span offsets, supporting English and Chinese.

        Rules:
        - English: split on [.!?]+(\s+|$), keep punctuation.
        - Chinese: split on [。！？]+, keep punctuation.
        Returns list of dicts: {"text": str, "start": int, "end": int}
        """
        raw = text if isinstance(text, str) else ""
        if not raw:
            return []
        # Normalize internal whitespace for stability in downstream joining
        cleaned = re.sub(r"\s+", " ", raw)
        spans: List[Dict] = []
        start = 0
        # Combined terminator regex: a sentence terminator and following spaces (English),
        # or just Chinese terminators.
        # Match either Chinese terminators (no trailing space required) or English terminators followed by space/EOS
        terminator_re = re.compile(r"([。！？]+)|([.!?]+)(\s+|$)")
        for m in terminator_re.finditer(cleaned):
            end = m.end()
            sentence = cleaned[start:end].strip()
            if sentence:
                spans.append({"text": sentence, "start": start, "end": end})
            start = end
        if start < len(cleaned):
            tail = cleaned[start:].strip()
            if tail:
                spans.append({"text": tail, "start": start, "end": len(cleaned)})
        return spans

    @staticmethod
    def is_pronoun(token: str) -> bool:
        t = (token or "").strip().lower()
        if not t:
            return False
        return t in TextUtils.EN_PRONOUNS or token in TextUtils.ZH_PRONOUNS

    @staticmethod
    def is_pronoun_subject_sentence(sentence: str) -> bool:
        s = (sentence or "").strip()
        if not s:
            return False
        # English: ^Pronoun + verb trigger
        en = s.split()
        if en:
            first = en[0].lower()
            if first in TextUtils.EN_PRONOUNS:
                if len(en) > 1:
                    second = en[1].lower()
                    if second in TextUtils.EN_SUBJECT_VERBS:
                        return True
                # This/That/These/Those formerly included
                return True
        # Chinese: ^Pronoun + trigger
        zh_match = re.match(rf"^({'|'.join(TextUtils.ZH_PRONOUNS)})({ '|'.join(TextUtils.ZH_SUBJECT_TRIGGERS) })", s)
        if zh_match:
            return True
        return False

    @staticmethod
    def is_entity_sentence(sentence: str) -> bool:
        s = (sentence or "").strip()
        if not s:
            return False
        # English entity candidates
        for pat in TextUtils.EN_ENTITY_PATTERNS:
            if pat.search(s):
                return True
        # Parentheses Full (Abbrev) or Abbrev (Full)
        if re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,})\)", s):
            return True
        if re.search(r"\b([A-Z]{2,})\s*\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\)", s):
            return True
        # Chinese suffix
        if re.search(TextUtils.ZH_ENTITY_SUFFIX, s):
            return True
        # Simple Chinese person name heuristic: 2-3 Han chars optionally with title words
        if re.search(r"^[\u4e00-\u9fa5]{2,3}.*(先生|女士|教授|博士|主席|市长|部长)", s):
            return True
        return False

    @staticmethod
    def extract_entity_candidates(sentence: str) -> List[str]:
        """Return surface entity candidates from sentence using heuristics.
        Includes English capitalized sequences, acronyms, and Chinese suffix-based entities,
        plus paired parentheses forms Full (Abbrev) / Abbrev (Full).
        """
        s = (sentence or "").strip()
        if not s:
            return []
        candidates: List[str] = []
        # English patterns
        for pat in TextUtils.EN_ENTITY_PATTERNS:
            for m in pat.finditer(s):
                grp = m.group(1)
                if grp and grp not in candidates:
                    candidates.append(grp)
        # Parentheses pair mapping: Full (Abbrev) → add both
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\(([A-Z]{2,})\)", s):
            full, abbr = m.group(1), m.group(2)
            for item in (full, abbr):
                if item not in candidates:
                    candidates.append(item)
        # Abbrev (Full Name)
        for m in re.finditer(r"\b([A-Z]{2,})\s*\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\)", s):
            abbr, full = m.group(1), m.group(2)
            for item in (full, abbr):
                if item not in candidates:
                    candidates.append(item)
        # Chinese suffix entities
        for m in re.finditer(rf"([\u4e00-\u9fa5]{{2,}}(?:{TextUtils.ZH_ENTITY_SUFFIX}))", s):
            val = m.group(1)
            if val and val not in candidates:
                candidates.append(val)
        return candidates
