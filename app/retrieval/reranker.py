import asyncio
import math


class Reranker:
    def __init__(self, models, settings):
        self.models = models
        self.top_k = settings.rerank_top_k

    async def rank(self, question, candidates):
        if not candidates:
            return []
        scores = await asyncio.to_thread(
            self.models.rerank,
            question,
            [c.node.retrieval_text for c in candidates],
        )
        ranked = []
        for candidate, score in zip(candidates, scores, strict=True):
            if math.isfinite(score):
                ranked.append(candidate.model_copy(update={"reranker_score": float(score)}))
        return sorted(ranked, key=lambda c: (-c.reranker_score, c.node.id))[: self.top_k]
