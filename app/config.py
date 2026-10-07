from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = Field(30, gt=0)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_nodes"
    qdrant_timeout_seconds: float = Field(15, gt=0)
    database_path: Path = Path("data/metadata.sqlite3")
    model_cache_dir: Path = Path("models")
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_threads: int = Field(2, ge=1)
    embedding_batch_size: int = Field(64, ge=1)

    retrieval_node_target_tokens: int = Field(350, ge=16)
    retrieval_node_max_tokens: int = Field(512, ge=32, le=700)
    retrieval_dense_top_k: int = Field(25, ge=1)
    retrieval_sparse_top_k: int = Field(25, ge=1)
    retrieval_fused_top_k: int = Field(20, ge=1)
    rerank_top_k: int = Field(6, ge=1)
    # Calibrated on evaluation/questions.json; requires validation on independent documents.
    answerability_threshold: float = -6.05
    max_context_tokens: int = Field(5000, ge=64)
    max_questions_per_request: int = Field(50, ge=1, le=1000)
    max_question_characters: int = Field(2000, ge=1)
    max_llm_concurrency: int = Field(8, ge=1)
    max_retrieval_concurrency: int = Field(4, ge=1)
    max_ingestion_concurrency: int = Field(2, ge=1)
    max_documents_per_request: int = Field(10, ge=1)
    max_upload_bytes: int = Field(50 * 1024 * 1024, ge=1)
    max_questions_file_bytes: int = Field(256 * 1024, ge=1)
    max_pdf_pages: int = Field(1500, ge=1)
    max_document_nodes: int = Field(100_000, ge=1)
    max_json_depth: int = Field(64, ge=1)
    pdf_parse_timeout_seconds: float = Field(3600, gt=0)

    @model_validator(mode="after")
    def token_limits(self):
        if self.retrieval_node_target_tokens > self.retrieval_node_max_tokens:
            raise ValueError("Target node size cannot exceed maximum node size")
        return self

    @property
    def max_request_bytes(self) -> int:
        return self.max_upload_bytes * self.max_documents_per_request + 1024 * 1024
