import math


def answerable(candidates, threshold: float) -> bool:
    return any(
        c.node.text.strip()
        and c.reranker_score is not None
        and math.isfinite(c.reranker_score)
        and c.reranker_score >= threshold
        for c in candidates
    )
