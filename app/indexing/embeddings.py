import threading

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from tokenizers import Tokenizer


class LocalModels:
    """One copy per API process. Serialize native inference to avoid CPU oversubscription."""

    def __init__(self, settings):
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        options = {
            "cache_dir": str(settings.model_cache_dir),
            "threads": settings.model_threads,
            "providers": ["CPUExecutionProvider"],
        }
        self.dense = TextEmbedding(model_name=settings.dense_model, **options)
        self.sparse = SparseTextEmbedding(model_name=settings.sparse_model, **options)
        self.cross_encoder = TextCrossEncoder(model_name=settings.reranker_model, **options)
        # Clone the loaded tokenizer: FastEmbed's inference tokenizer truncates/pads.
        # Keep the adapter here, checked against our pinned FastEmbed version.
        self.tokenizer = Tokenizer.from_str(self.dense.model.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        model_limit = self.dense.model.tokenizer.truncation["max_length"]
        if settings.retrieval_node_max_tokens > model_limit:
            raise ValueError(f"Node limit exceeds dense model input limit ({model_limit})")
        self.dimension = self.dense.embedding_size
        self.batch_size = settings.embedding_batch_size
        # ponytail: serialize native inference; add per-model locks if CPU profiling warrants it.
        self.lock = threading.Lock()

    def encode_documents(self, texts: list[str]):
        with self.lock:
            dense = [
                v.tolist() for v in self.dense.passage_embed(texts, batch_size=self.batch_size)
            ]
            sparse = list(self.sparse.passage_embed(texts, batch_size=self.batch_size))
        return dense, sparse

    def encode_questions(self, questions: list[str]):
        with self.lock:
            dense = [
                v.tolist() for v in self.dense.query_embed(questions, batch_size=self.batch_size)
            ]
            sparse = list(self.sparse.query_embed(questions, batch_size=self.batch_size))
        return list(zip(dense, sparse, strict=True))

    def rerank(self, question: str, texts: list[str]):
        with self.lock:
            return list(self.cross_encoder.rerank(question, texts, batch_size=self.batch_size))
