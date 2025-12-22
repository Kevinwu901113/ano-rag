from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, docs: Sequence[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.docs = list(docs)
        self.doc_len: List[int] = []
        self.avgdl = 0.0
        self.df: Counter[str] = Counter()
        self.postings: Dict[str, List[tuple[int, int]]] = {}
        self.idf: Dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        for idx, doc in enumerate(self.docs):
            tokens = tokenize(doc)
            freq = Counter(tokens)
            self.doc_len.append(len(tokens))
            for token, tf in freq.items():
                self.df[token] += 1
                self.postings.setdefault(token, []).append((idx, tf))
        total_docs = len(self.docs)
        self.avgdl = sum(self.doc_len) / max(1, total_docs)
        self.idf = {
            token: math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            for token, df in self.df.items()
        }

    def get_scores(self, query: Iterable[str] | str) -> List[float]:
        tokens = tokenize(query) if isinstance(query, str) else list(query)
        scores = [0.0] * len(self.docs)
        if not tokens:
            return scores
        for token in tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            postings = self.postings.get(token, [])
            for doc_id, tf in postings:
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / max(1.0, self.avgdl)))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0) / denom)
        return scores


def rrf_fuse(rank_lists: Sequence[Sequence[int]], *, k: int = 60) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores
