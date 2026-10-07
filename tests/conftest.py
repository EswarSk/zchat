from types import SimpleNamespace

import httpx
import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.indexing.qdrant_store import QdrantStore
from app.ingestion.segmenter import Segmenter
from app.main import Services, create_app
from app.retrieval.context_expansion import ContextExpander
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.services import DocumentService, QAService
from app.storage.repository import Repository
from tests.helpers import FakeGenerator, FakeModels, FakeParser


@pytest.fixture
async def system(tmp_path):
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "db.sqlite",
        model_cache_dir=tmp_path / "models",
        max_llm_concurrency=2,
    )
    repository = Repository(settings.database_path)
    models = FakeModels()
    parser = FakeParser(settings)
    generator = FakeGenerator()
    store = QdrantStore(AsyncQdrantClient(location=":memory:"), settings)
    await store.initialize(models.dimension)
    documents = DocumentService(
        repository, parser, Segmenter(models.tokenizer, settings), models, store, settings
    )
    qa = QAService(
        repository,
        models,
        HybridRetriever(repository, store, settings),
        Reranker(models, settings),
        ContextExpander(repository, settings),
        generator,
        settings,
    )
    services = Services(repository, store, documents, qa)
    app = create_app(settings, services)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield SimpleNamespace(
                settings=settings,
                repo=repository,
                models=models,
                parser=parser,
                generator=generator,
                store=store,
                docs=documents,
                qa=qa,
                app=app,
                client=client,
            )
    await services.close()
