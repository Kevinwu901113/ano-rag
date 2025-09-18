import re, math
from collections import defaultdict

def _char_ngrams(text: str, n_gram=(3,5)) -> set[str]:
    t = re.sub(r"\s+", " ", (text or "").lower())
    t = f" {t} "
    grams = set()
    a, b = n_gram
    for n in range(a, b+1):
        for i in range(0, max(0, len(t)-n+1)):
            grams.add(t[i:i+n])
    return grams

def char_jaccard(a: str, b: str, n_gram=(3,5)) -> float:
    A = _char_ngrams(a, n_gram); B = _char_ngrams(b, n_gram)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

class LSHIndex:
    """
    简化实现：先暴力评分，但用 IDF 加权的字符 n-gram Jaccard。
    以后可把 self._score 换成 MinHash/LSH 的近似召回再重算加权分。
    """
    def __init__(self, n_gram=(3,5), bands=32, rows=4):
        self._n = n_gram
        self._docs = {}        # id -> raw text
        self._grams = {}       # id -> set of grams
        self._idf = {}         # gram -> idf
        self._finalized = False

    def add(self, doc_id: int, text: str) -> None:
        self._docs[doc_id] = text or ""
        self._finalized = False

    def _ensure(self):
        if self._finalized: return
        df = defaultdict(int)
        self._grams = {}
        for doc_id, txt in self._docs.items():
            gset = _char_ngrams(txt, self._n)
            self._grams[doc_id] = gset
            for g in gset:
                df[g] += 1
        N = max(1, len(self._docs))
        # IDF：log(1 + N/df)
        self._idf = {g: math.log(1.0 + N/dfc) for g, dfc in df.items()}
        self._finalized = True

    def _weighted_jaccard(self, Gq: set[str], Gd: set[str]) -> float:
        if not Gq or not Gd: return 0.0
        inter = Gq & Gd
        union = Gq | Gd
        num = sum(self._idf.get(g, 0.0) for g in inter)
        den = sum(self._idf.get(g, 0.0) for g in union) or 1.0
        return num / den

    def query(self, text: str, topn: int = 20):
        self._ensure()
        Gq = _char_ngrams(text or "", self._n)
        scores = []
        for doc_id, Gd in self._grams.items():
            wj = self._weighted_jaccard(Gq, Gd)
            scores.append((doc_id, wj))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topn]