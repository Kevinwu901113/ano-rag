import re
from typing import List, Dict, Optional


class TextUtils:
    @staticmethod
    def rough_token_len(s: str) -> int:
        """Heuristic char-to-token estimator for routing/scheduling."""
        if not s:
            return 0
        return int(len(s) / 3.2) + 1

    # --- Pronoun lists (non-exhaustive, practical) ---
    EN_PERSONAL_PRONOUNS = {
        "he",
        "she",
        "it",
        "they",
        "him",
        "her",
        "them",
    }
    EN_POSSESSIVE_PRONOUNS = {
        "his",
        "her",
        "its",
        "their",
    }
    EN_DEMONSTRATIVES = {
        "this", "that", "these", "those", "former", "latter", "the former", "the latter",
    }
    EN_PRONOUNS = EN_PERSONAL_PRONOUNS | EN_POSSESSIVE_PRONOUNS
    EN_TITLE_PREFIXES = ("mr.", "mrs.", "ms.", "miss", "dr.", "prof.", "sir", "madam", "lord", "lady")
    EN_ORG_HINTS = (
        "university", "college", "party", "republic", "revolutionary", "committee", "council",
        "company", "limited", "ltd", "inc", "llc", "press", "newspaper", "bank", "association",
        "foundation", "society", "team", "government", "ministry", "agency", "agency", "church",
    )
    EN_PLACE_HINTS = (
        "city", "county", "province", "state", "republic", "kingdom", "village", "municipality",
        "river", "lake", "mount", "mountain", "bay", "harbor", "harbour", "island", "peninsula",
    )
    REPORTING_VERBS = ("said", "stated", "told", "wrote", "added", "according to")
    ZH_PRONOUNS = {
        "他", "她", "它", "他们", "她们", "它们", "其", "该", "此", "本", "前者", "后者", "上述",
        "该公司", "该机构", "该团队", "该部门",
    }

    # Common English verb triggers for subject position
    EN_SUBJECT_VERBS = (
        "be", "was", "were", "is", "are", "has", "have", "had", "said", "announced", "joined",
        "founded", "born", "died", "won", "created", "wrote", "worked", "served", "became",
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
        - English: split on [.!?]+(\\s+|$), keep punctuation.
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
        terminator_re = re.compile(r"([。！？]+)|([.!?;]+)(\s+|$)")
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
        return t in TextUtils.EN_PRONOUNS or t in TextUtils.ZH_PRONOUNS

    @staticmethod
    def is_pronoun_at_start(sentence: str) -> bool:
        """Lightweight check for pronoun at sentence start (first 1–2 tokens)."""
        s = (sentence or "").strip()
        if not s:
            return False
        parts = s.split()
        if not parts:
            return False
        first_raw = parts[0]
        first_norm = TextUtils._normalize_pronoun_token(first_raw)
        if TextUtils.is_pronoun(first_norm):
            return True
        # Chinese pronouns may appear without whitespace; also consider short prefixes/punctuation
        zh_pronoun_pat = rf"^({'|'.join(re.escape(p) for p in TextUtils.ZH_PRONOUNS)})"
        if re.match(zh_pronoun_pat, s):
            return True
        # Allow simple two-token English noun phrases like "the company"
        if len(parts) >= 2:
            lead_two = " ".join(parts[:2]).lower()
            if lead_two in {"the company", "the organization", "the team", "the group"}:
                return True
        return False

    @staticmethod
    def _normalize_pronoun_token(token: str) -> str:
        cleaned = (token or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.replace("’", "'")
        cleaned = cleaned.strip(" \"'“”‘’()[]{}.,;:!?-")
        cleaned = cleaned.lower()
        cleaned = re.sub(r"[^a-z']", "", cleaned)
        return cleaned

    @staticmethod
    def starts_with_pronoun(sentence: str) -> bool:
        return TextUtils.is_pronoun_at_start(sentence)

    @staticmethod
    def is_pronoun_subject_sentence(sentence: str) -> bool:
        s = (sentence or "").strip()
        if not s:
            return False
        # English: ^Pronoun + verb trigger
        en = s.split()
        if en:
            first_raw = en[0]
            first_norm = TextUtils._normalize_pronoun_token(first_raw)
            second_norm: Optional[str] = None
            english_candidate = False
            # Handle contractions such as He's/She's/It's → ("he", "is")
            if first_norm.endswith("'s"):
                base = first_norm[:-2]
                if base in TextUtils.EN_PERSONAL_PRONOUNS:
                    first_norm = base
                    second_norm = "is"
                    english_candidate = True
            allowed_pronouns = TextUtils.EN_PRONOUNS
            if first_norm in allowed_pronouns:
                english_candidate = True
                if len(en) > 1 and second_norm is None:
                    second_norm = TextUtils._normalize_pronoun_token(en[1])
                if second_norm in TextUtils.EN_SUBJECT_VERBS:
                    return True
                return False
            if english_candidate:
                return False
        # Chinese: ^Pronoun + trigger
        zh_pronoun = "|".join(re.escape(p) for p in TextUtils.ZH_PRONOUNS)
        zh_triggers = "|".join(re.escape(t) for t in TextUtils.ZH_SUBJECT_TRIGGERS)
        pattern = rf"^({zh_pronoun})({zh_triggers})"
        return re.match(pattern, s) is not None

    @staticmethod
    def is_entity_sentence(sentence: str) -> bool:
        s = (sentence or "").strip()
        if not s:
            return False
        lowered = s.lower()
        # Explicit reporting constructions: "..., said John Doe"
        if TextUtils._has_reporting_clause(s):
            return True
        if TextUtils._looks_like_titled_name(s):
            return True
        if any(token in lowered for token in TextUtils.EN_ORG_HINTS):
            return True
        if any(token in lowered for token in TextUtils.EN_PLACE_HINTS):
            return True
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
    def _looks_like_titled_name(sentence: str) -> bool:
        lowered = (sentence or "").strip().lower()
        if not lowered:
            return False
        for prefix in TextUtils.EN_TITLE_PREFIXES:
            if lowered.startswith(prefix + " "):
                return True
        if re.search(r"\b(Dr|Sir|Prof|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+", sentence):
            return True
        if re.search(r"[A-Z][a-z]+\s+[A-Z][a-z]+", sentence):
            return True
        return False

    @staticmethod
    def _has_reporting_clause(sentence: str) -> bool:
        if not sentence:
            return False
        pattern = r"[\"“”‘’'].+?[\"“”‘’']\s*,?\s*(%s)\s+[A-Z][a-z]+" % "|".join(TextUtils.REPORTING_VERBS)
        return bool(re.search(pattern, sentence, flags=re.IGNORECASE))

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

    @staticmethod
    def extract_entities(sentence: str) -> List[str]:
        """Lightweight entity extractor using common English/Chinese patterns."""
        s = (sentence or "").strip()
        if not s:
            return []
        entities: List[str] = []
        for pat in TextUtils.EN_ENTITY_PATTERNS:
            for m in pat.finditer(s):
                val = m.group(1)
                if val and val not in entities:
                    entities.append(val)
        zh_pattern = re.compile(rf"([\u4e00-\u9fa5]{{2,}}(?:{TextUtils.ZH_ENTITY_SUFFIX}))")
        for m in zh_pattern.finditer(s):
            val = m.group(1)
            if val and val not in entities:
                entities.append(val)
        return entities

    @staticmethod
    def guess_entity_type(entity: Optional[str]) -> Optional[str]:
        text = (entity or "").strip()
        if not text:
            return None
        lowered = text.lower()
        if re.match(r"[A-Z][a-z]+\s+[A-Z][a-z]+", text):
            return "PERSON"
        if any(lowered.startswith(prefix) for prefix in TextUtils.EN_TITLE_PREFIXES):
            return "PERSON"
        if any(token in lowered for token in TextUtils.EN_ORG_HINTS):
            return "ORG"
        if any(token in lowered for token in TextUtils.EN_PLACE_HINTS):
            return "PLACE"
        if re.search(r"(?:Inc|LLC|Ltd|Co)\.?$", text):
            return "ORG"
        if re.search(r"(City|County|Province|Village|Town)$", text):
            return "PLACE"
        if re.search(r"大学|公司|集团|研究院", text):
            return "ORG"
        if re.search(r"省|市|县|镇|乡|河|湖|山", text):
            return "PLACE"
        return None
