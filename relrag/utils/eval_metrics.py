import math
import re
from typing import Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _bleu_score(references: List[str], hypothesis: str, max_n: int) -> float:
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    ref_tokens_list = [_tokenize(ref) for ref in references if ref]
    if not ref_tokens_list:
        return 0.0

    hyp_len = len(hyp_tokens)
    ref_lens = [len(ref) for ref in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    if hyp_len > closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (closest_ref_len / max(1, hyp_len)))

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_counts = _ngram_counts(hyp_tokens, n)
        max_ref_counts: Dict[Tuple[str, ...], int] = {}
        for ref_tokens in ref_tokens_list:
            ref_counts = _ngram_counts(ref_tokens, n)
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)
        match = sum(min(count, max_ref_counts.get(gram, 0)) for gram, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        if total == 0:
            precision = 0.0
        else:
            precision = match / total
        if precision == 0.0:
            precision = (match + 1.0) / (total + 1.0)
        precisions.append(precision)

    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return float(score)


def _lcs_alignment(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[int, int]]:
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0 or m == 0:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    i = n
    j = m
    alignment: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        if ref_tokens[i - 1] == hyp_tokens[j - 1]:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    alignment.reverse()
    return alignment


def _rouge_l_score(references: List[str], hypothesis: str) -> float:
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    best = 0.0
    beta = 1.2
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue
        alignment = _lcs_alignment(ref_tokens, hyp_tokens)
        lcs_len = len(alignment)
        if lcs_len == 0:
            continue
        prec = lcs_len / len(hyp_tokens)
        rec = lcs_len / len(ref_tokens)
        denom = rec + (beta * beta * prec)
        if denom == 0:
            f_score = 0.0
        else:
            f_score = (1 + beta * beta) * prec * rec / denom
        best = max(best, f_score)
    return float(best)


def _meteor_score(references: List[str], hypothesis: str) -> float:
    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = _tokenize(ref)
        if not ref_tokens:
            continue
        alignment = _lcs_alignment(ref_tokens, hyp_tokens)
        matches = len(alignment)
        if matches == 0:
            continue
        prec = matches / len(hyp_tokens)
        rec = matches / len(ref_tokens)
        denom = rec + 9 * prec
        if denom == 0:
            f_mean = 0.0
        else:
            f_mean = (10 * prec * rec) / denom
        chunks = 1
        for idx in range(1, len(alignment)):
            prev = alignment[idx - 1]
            curr = alignment[idx]
            if curr[0] != prev[0] + 1 or curr[1] != prev[1] + 1:
                chunks += 1
        penalty = 0.5 * (chunks / matches) ** 3
        score = (1 - penalty) * f_mean
        best = max(best, score)
    return float(best)


def score_metrics(prediction: str, references: List[str]) -> Dict[str, float]:
    if not references:
        return {"bleu1": 0.0, "bleu4": 0.0, "rougeL": 0.0, "meteor": 0.0}
    bleu1 = _bleu_score(references, prediction, 1)
    bleu4 = _bleu_score(references, prediction, 4)
    rouge_l = _rouge_l_score(references, prediction)
    meteor = _meteor_score(references, prediction)
    return {
        "bleu1": round(bleu1, 4),
        "bleu4": round(bleu4, 4),
        "rougeL": round(rouge_l, 4),
        "meteor": round(meteor, 4),
    }
