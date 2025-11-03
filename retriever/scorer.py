def score_path(path) -> float:
    base = 1.0
    hop_penalty = 0.05 * (len(path) - 1)
    return base - hop_penalty
