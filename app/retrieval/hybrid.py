import asyncio
from time import perf_counter

from app.domain.models import Candidate


def rrf(rankings: list[list[tuple[str, float]]], limit: int, k: int = 60):
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, (node_id, _score) in enumerate(ranking, start=1):
            if node_id not in seen:
                scores[node_id] = scores.get(node_id, 0.0) + 1 / (k + rank)
                seen.add(node_id)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


class HybridRetriever:
    def __init__(self, repository, store, settings):
        self.repository = repository
        self.store = store
        self.settings = settings

    async def retrieve(self, chat_id, encoded_question, mode="hybrid"):
        ready = await asyncio.to_thread(self.repository.ready_ids, chat_id)
        if not ready:
            return [], {"dense_ms": 0.0, "sparse_ms": 0.0, "fusion_ms": 0.0}
        dense_vector, sparse_vector = encoded_question
        timing = {}

        async def search(kind, vector, limit):
            start = perf_counter()
            result = await self.store.search(chat_id, ready, vector, kind, limit)
            timing[f"{kind}_ms"] = (perf_counter() - start) * 1000
            return result

        if mode == "dense":
            rankings = [await search("dense", dense_vector, self.settings.retrieval_dense_top_k)]
        elif mode == "sparse":
            rankings = [await search("sparse", sparse_vector, self.settings.retrieval_sparse_top_k)]
        elif mode == "hybrid":
            rankings = await asyncio.gather(
                search("dense", dense_vector, self.settings.retrieval_dense_top_k),
                search("sparse", sparse_vector, self.settings.retrieval_sparse_top_k),
            )
        else:
            raise ValueError("Unknown retrieval mode")
        start = perf_counter()
        fused = rrf(rankings, self.settings.retrieval_fused_top_k)
        timing["fusion_ms"] = (perf_counter() - start) * 1000
        # Never trust a vector-store payload as evidence. Rehydrate READY raw nodes from SQLite.
        raw = await asyncio.to_thread(self.repository.nodes_by_ids, chat_id, [i for i, _ in fused])
        by_id = {n.id: n for n in raw}
        candidates = [Candidate(node=by_id[i], retrieval_score=s) for i, s in fused if i in by_id]
        return candidates, timing
