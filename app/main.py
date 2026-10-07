import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.api.errors import AppError
from app.api.input import question_file, question_request, read_upload
from app.config import Settings
from app.generation.grounded_answer import GroundedGenerator
from app.indexing.embeddings import LocalModels
from app.indexing.qdrant_store import QdrantStore
from app.ingestion.docling_parser import DoclingParser
from app.ingestion.segmenter import Segmenter
from app.logging import configure_logging
from app.retrieval.context_expansion import ContextExpander
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.services import DocumentService, QAService
from app.storage.repository import Repository


@dataclass
class Services:
    repository: Repository
    store: QdrantStore
    documents: DocumentService
    qa: QAService

    async def close(self):
        await self.documents.close()
        await self.qa.generator.close()
        await self.store.close()
        self.repository.close()


async def build_services(settings: Settings):
    repository = Repository(settings.database_path)
    store = QdrantStore(
        AsyncQdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
        ),
        settings,
    )
    try:
        repository.recover_interrupted_ingestion()
        models = await asyncio.to_thread(LocalModels, settings)
        await store.initialize(models.dimension)
        documents = DocumentService(
            repository,
            DoclingParser(settings),
            Segmenter(models.tokenizer, settings),
            models,
            store,
            settings,
        )
        qa = QAService(
            repository,
            models,
            HybridRetriever(repository, store, settings),
            Reranker(models, settings),
            ContextExpander(repository, settings),
            GroundedGenerator(settings),
            settings,
        )
        return Services(repository, store, documents, qa)
    except BaseException:
        await store.close()
        repository.close()
        raise


class BodyLimitMiddleware:
    """Bound multipart spooling even when clients omit Content-Length."""

    def __init__(self, app, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0
        headers = dict(scope["headers"])
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = self.limit + 1
        if declared > self.limit:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "FILE_TOO_LARGE",
                        "message": "Request exceeds configured upload limit",
                    }
                },
            )
            return await response(scope, receive, send)

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.limit:
                raise AppError(413, "FILE_TOO_LARGE", "Request exceeds configured upload limit")
            return message

        return await self.app(scope, bounded_receive, send)


def create_app(settings: Settings | None = None, services: Services | None = None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        configure_logging()
        app.state.services = services or await build_services(settings)
        try:
            yield
        finally:
            if services is None:
                await app.state.services.close()

    app = FastAPI(title="ZChat — Grounded Document QA", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_request_bytes)

    @app.exception_handler(AppError)
    async def app_error(_request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {"code": exc.code, "message": exc.message},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "Check required fields and uploaded files",
                }
            },
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        bundle = request.app.state.services
        try:
            await asyncio.to_thread(bundle.repository.ping)
            await bundle.store.ping()
        except Exception as exc:
            raise AppError(503, "NOT_READY", "A required backend is unavailable") from exc
        return {"status": "ok", "answer_provider_configured": bool(settings.openai_api_key)}

    @app.post("/api/v1/chats", status_code=201)
    async def create_chat(request: Request):
        return {
            "chat_id": await asyncio.to_thread(request.app.state.services.repository.create_chat)
        }

    @app.get("/api/v1/chats/{chat_id}/documents")
    async def list_documents(chat_id: str, request: Request):
        documents = await asyncio.to_thread(
            request.app.state.services.repository.documents, chat_id
        )
        return {"documents": [d.model_dump() for d in documents]}

    @app.post("/api/v1/chats/{chat_id}/documents")
    async def upload_documents(
        chat_id: str,
        request: Request,
        documents: Annotated[list[UploadFile], File()],
        background: bool = False,
    ):
        bundle = request.app.state.services
        bundle.repository.require_chat(chat_id)
        if not documents or len(documents) > settings.max_documents_per_request:
            raise AppError(400, "INVALID_REQUEST", "Document count exceeds configured limit")
        results = []
        for upload in documents:
            try:
                data = await read_upload(upload, settings.max_upload_bytes)
                results.append(
                    await (bundle.documents.submit if background else bundle.documents.ingest)(
                        chat_id,
                        upload.filename or "",
                        upload.content_type or "",
                        data,
                    )
                )
            except AppError as exc:
                if len(documents) == 1:
                    raise
                results.append(
                    {
                        "filename": upload.filename,
                        "status": "FAILED",
                        "error": {
                            "code": exc.code,
                            "message": exc.message,
                            "http_status": exc.status,
                        },
                    }
                )
            finally:
                await upload.close()
        return JSONResponse(
            status_code=202 if background else 200,
            content={"documents": results},
        )

    async def results(bundle, chat_id, questions, previous_questions=None, output="detailed"):
        answers = await bundle.qa.answer_many(
            chat_id, [q.question for q in questions], previous_questions
        )
        for question, answer in zip(questions, answers, strict=True):
            answer.id = question.id
        if output == "questionnaire":
            return [a.questionnaire_record() for a in answers]
        return {"chat_id": chat_id, "results": [a.model_dump() for a in answers]}

    @app.post(
        "/api/v1/chats/{chat_id}/questions",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["questions"],
                            "properties": {
                                "questions": {
                                    "type": "array",
                                    "items": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "object",
                                                "required": ["question"],
                                                "properties": {
                                                    "id": {"type": "string"},
                                                    "question": {"type": "string"},
                                                },
                                            },
                                        ]
                                    },
                                },
                                "previous_questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 5,
                                    "description": "Earlier user questions for intent only",
                                },
                            },
                        }
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["questions"],
                            "properties": {"questions": {"type": "string", "format": "binary"}},
                        }
                    },
                },
            }
        },
    )
    async def ask_questions(
        chat_id: str, request: Request, output: Literal["detailed", "questionnaire"] = "detailed"
    ):
        bundle = request.app.state.services
        bundle.repository.require_chat(chat_id)
        kind = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if kind == "application/json":
            # Streaming read keeps a forged/missing Content-Length bounded.
            chunks, total = [], 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > settings.max_questions_file_bytes:
                    raise AppError(413, "FILE_TOO_LARGE", "Questions exceed configured byte limit")
                chunks.append(chunk)
            questions, previous = question_request(b"".join(chunks), settings)
        elif kind == "multipart/form-data":
            async with request.form(max_files=1, max_fields=1) as form:
                upload = form.get("questions")
                if not isinstance(upload, StarletteUploadFile):
                    raise AppError(400, "INVALID_QUESTIONS_JSON", "Upload the questions JSON file")
                questions, previous = question_request(
                    await read_upload(upload, settings.max_questions_file_bytes),
                    settings,
                )
        else:
            raise AppError(400, "INVALID_QUESTIONS_JSON", "Use JSON or multipart/form-data")
        return await results(bundle, chat_id, questions, previous, output)

    @app.post("/api/v1/qa")
    async def challenge_qa(
        request: Request,
        document: Annotated[UploadFile, File()],
        questions: Annotated[UploadFile, File()],
        output: Literal["detailed", "questionnaire"] = "detailed",
    ):
        parsed = question_file(
            await read_upload(questions, settings.max_questions_file_bytes), settings
        )
        data = await read_upload(document, settings.max_upload_bytes)
        bundle = request.app.state.services
        chat_id = await asyncio.to_thread(bundle.repository.create_chat)
        await bundle.documents.ingest(
            chat_id,
            document.filename or "",
            document.content_type or "",
            data,
        )
        return await results(bundle, chat_id, parsed, output=output)

    return app


app = create_app()
