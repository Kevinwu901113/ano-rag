from typing import Optional
from utils import TextUtils

# Optional fuzzy helpers for alias/title similarity
def _normalize_token(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("-", " ").replace(".", " ")
    return " ".join(t.split())

def _jaccard(a: str, b: str) -> float:
    ta = set(_normalize_token(a).split())
    tb = set(_normalize_token(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return (inter / union) if union else 0.0


def score_path(path, *, doc_name: Optional[str] = None) -> float:
    base = 1.0
    hop_penalty = 0.05 * (len(path) - 1)
    presence_penalty = 0.0
    alias_bonus = 0.0
    for edge in path or []:
        subj = (edge.get("subj") or "").strip()
        obj = (edge.get("obj") or "").strip()
        for tok in (subj, obj):
            if not tok or len(tok) <= 1:
                presence_penalty += 0.02
                continue
            # 代词或全小写（更可能是普通词）扣分；避免 Cone→cone 混淆软匹配
            if TextUtils.is_pronoun(tok) or tok.islower():
                presence_penalty += 0.02
        # doc_name 别名约束：若主语命中 doc_name 相似，则加分（消歧）
        if isinstance(doc_name, str) and doc_name.strip():
            sim = _jaccard(subj, doc_name)
            if sim >= 0.8:
                alias_bonus += 0.08
            elif sim >= 0.6:
                alias_bonus += 0.05
            elif sim >= 0.4:
                alias_bonus += 0.02
    return max(0.0, base - hop_penalty - presence_penalty + alias_bonus)
