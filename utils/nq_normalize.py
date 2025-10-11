import re

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NQ/SQuAD 风格规范化：小写、去标点、去冠词、压空白。"""
    if s is None:
        return ""
    s = s.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def is_yes_no_question(q: str) -> bool:
    if not q:
        return False
    ql = q.strip().lower()
    # 英文/常见中文“是否”类启发
    if ql.startswith((
        "is ",
        "are ",
        "was ",
        "were ",
        "do ",
        "does ",
        "did ",
        "can ",
        "could ",
        "will ",
        "would ",
        "has ",
        "have ",
        "had ",
    )):
        return True
    return "是否" in ql or "是不是" in ql or "有无" in ql


YES = "yes"
NO = "no"
