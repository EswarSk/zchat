import asyncio
import hashlib
import logging
import tempfile
from pathlib import Path
from time import perf_counter

from app.api.errors import AppError
from app.domain.models import AnswerResult, QuestionError
from app.generation.validation import validate_answer
from app.ingestion.json_parser import parse_json
from app.retrieval.answerability import answerable

logger = logging.getLogger("zchat")


def document_type(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    expected = {".pdf": "application/pdf", ".json": "application/json"}.get(suffix)
    allowed = {expected, "application/octet-stream", "", None}
    if suffix == ".json":
        allowed.add("text/json")
    if not expected or content_type not in allowed:
        raise AppError(400, "UNSUPPORTED_DOCUMENT_TYPE", "Upload a PDF or JSON source document")
    return expected


class DocumentService:
    def __init__(self, repository, parser, segmenter, models, store, settings):
        self.repository, self.parser, self.segmenter = repository, parser, segmenter
        self.models, self.store, self.settings = models, store, settings
        self.semaphore = asyncio.Semaphore(settings.max_ingestion_concurrency)
        self.submission_lock = asyncio.Lock()
        self.background_tasks: set[asyncio.Task] = set()

    async def reserve(self, chat_id: str, filename: str, content_type: str, data: bytes):
        self.repository.require_chat(chat_id)
        filename = Path(filename.replace("\\", "/")).name[:255]
        kind = document_type(filename, content_type)
        if len(data) > self.settings.max_upload_bytes:
            raise AppError(413, "FILE_TOO_LARGE", "Document exceeds configured upload limit")
        if not data:
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "Document is empty")
        digest = hashlib.sha256(data).hexdigest()
        # Unique(chat_id,sha256) arbitrates duplicate concurrent uploads, including across threads.
        document, duplicate = await asyncio.to_thread(
            self.repository.reserve_document,
            chat_id,
            filename,
            kind,
            digest,
        )
        return document, duplicate, filename, kind

    @staticmethod
    def duplicate_response(document):
        return {
            "document_id": document.id,
            "filename": document.filename,
            "status": document.status,
            "duplicate": True,
            "error_message": document.error_message,
        }

    async def ingest(self, chat_id: str, filename: str, content_type: str, data: bytes):
        document, duplicate, filename, kind = await self.reserve(
            chat_id, filename, content_type, data
        )
        if duplicate:
            return self.duplicate_response(document)
        return await self.process(document, filename, kind, data)

    async def submit(self, chat_id: str, filename: str, content_type: str, data: bytes):
        async with self.submission_lock:
            document, duplicate, filename, kind = await self.reserve(
                chat_id, filename, content_type, data
            )
            if duplicate:
                return self.duplicate_response(document)
            if len(self.background_tasks) >= self.settings.max_ingestion_concurrency:
                await asyncio.to_thread(
                    self.repository.set_status,
                    chat_id,
                    document.id,
                    "FAILED",
                    error="INGESTION_BUSY",
                )
                raise AppError(429, "INGESTION_BUSY", "Ingestion is busy; retry this upload")
            task = asyncio.create_task(self.process(document, filename, kind, data))
            self.background_tasks.add(task)
            task.add_done_callback(self._background_done)
            return {
                "document_id": document.id,
                "filename": filename,
                "status": "PENDING",
                "duplicate": False,
            }

    def _background_done(self, task: asyncio.Task):
        self.background_tasks.discard(task)
        if not task.cancelled():
            task.exception()  # The document status and structured log carry the failure.

    async def close(self):
        for task in self.background_tasks:
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

    async def process(self, document, filename: str, kind: str, data: bytes):
        chat_id = document.chat_id
        started = perf_counter()
        index_started = False
        try:
            async with self.semaphore:
                # A failed upload can have a partial index; retries must start clean.
                await self.store.delete_document(chat_id, document.id)
                await asyncio.to_thread(self.repository.set_status, chat_id, document.id, "PARSING")
                parse_started = perf_counter()
                if kind == "application/json":
                    nodes = await asyncio.to_thread(
                        parse_json, data, chat_id, document.id, self.settings
                    )
                    page_count = None
                    parser_route = "json"
                else:
                    with tempfile.TemporaryDirectory(prefix="zchat-") as directory:
                        path = Path(directory) / "source.pdf"
                        await asyncio.to_thread(path.write_bytes, data)
                        nodes, page_count = await self.parser.parse_async(
                            path, chat_id, document.id
                        )
                    parser_route = (
                        "native_text"
                        if any(n.metadata.get("parser") == "pypdfium2" for n in nodes)
                        else "docling"
                    )
                parse_ms = (perf_counter() - parse_started) * 1000
                nodes = await asyncio.to_thread(self.segmenter.segment, nodes)
                leaves = [n for n in nodes if n.searchable]
                if not leaves:
                    raise AppError(422, "DOCUMENT_PARSE_FAILED", "Document has no searchable text")
                await asyncio.to_thread(self.repository.save_nodes, chat_id, document.id, nodes)
                await asyncio.to_thread(
                    self.repository.set_status,
                    chat_id,
                    document.id,
                    "INDEXING",
                    page_count=page_count,
                )
                index_started = True
                for start in range(0, len(leaves), self.settings.embedding_batch_size):
                    batch = leaves[start : start + self.settings.embedding_batch_size]
                    dense, sparse = await asyncio.to_thread(
                        self.models.encode_documents,
                        [n.retrieval_text for n in batch],
                    )
                    await self.store.upsert(chat_id, document, batch, dense, sparse)
                await asyncio.to_thread(self.repository.set_status, chat_id, document.id, "READY")
        except BaseException as exc:
            if isinstance(exc, AppError):
                code = exc.code
            elif isinstance(exc, asyncio.CancelledError):
                code = "INGESTION_INTERRUPTED"
            else:
                code = "DOCUMENT_INDEX_FAILED"
            await asyncio.to_thread(
                self.repository.set_status,
                chat_id,
                document.id,
                "FAILED",
                error=code,
            )
            if index_started:
                try:
                    await self.store.delete_document(chat_id, document.id)
                except Exception:
                    # READY filtering remains authoritative even if Qdrant cleanup fails.
                    logger.warning(
                        "index_cleanup_failed",
                        extra={
                            "metrics": {
                                "chat_id": chat_id,
                                "document_id": document.id,
                            }
                        },
                    )
            logger.warning(
                "document_ingestion_failed",
                extra={
                    "metrics": {
                        "chat_id": chat_id,
                        "document_id": document.id,
                        "error_code": code,
                        "ingestion_ms": round((perf_counter() - started) * 1000, 3),
                    }
                },
            )
            if isinstance(exc, (AppError, asyncio.CancelledError)):
                raise
            raise AppError(
                503, code, "Document could not be indexed; retry with a new chat"
            ) from exc
        logger.info(
            "document_ingested",
            extra={
                "metrics": {
                    "chat_id": chat_id,
                    "document_id": document.id,
                    "page_count": page_count,
                    "retrieval_nodes": len(leaves),
                    "parser_route": parser_route,
                    "parse_ms": round(parse_ms, 3),
                    "ingestion_ms": round((perf_counter() - started) * 1000, 3),
                }
            },
        )
        return {
            "document_id": document.id,
            "filename": filename,
            "status": "READY",
            "duplicate": False,
        }


class QAService:
    def __init__(self, repository, models, retriever, reranker, expander, generator, settings):
        self.repository, self.models, self.retriever = repository, models, retriever
        self.reranker, self.expander, self.generator = reranker, expander, generator
        self.settings = settings
        # Global per process, not per request: batches share the same limits.
        self.llm_semaphore = asyncio.Semaphore(settings.max_llm_concurrency)
        self.retrieval_semaphore = asyncio.Semaphore(settings.max_retrieval_concurrency)

    async def answer_many(self, chat_id: str, questions: list[str], previous_questions=None):
        self.repository.require_chat(chat_id)
        documents = self.repository.documents(chat_id)
        if documents and not any(d.status == "READY" for d in documents):
            raise AppError(409, "DOCUMENT_NOT_READY", "This chat has no READY documents")
        started = perf_counter()
        queries, context_ms = list(questions), [0.0] * len(questions)

        async def contextualize(index, question):
            async with self.llm_semaphore:
                tick = perf_counter()
                try:
                    return await self.generator.contextualize(question, previous_questions)
                except AppError as exc:
                    return exc
                except Exception:
                    return AppError(
                        502, "QUESTION_CONTEXT_ERROR", "Could not interpret the follow-up"
                    )
                finally:
                    context_ms[index] = (perf_counter() - tick) * 1000

        if documents and previous_questions:
            queries = await asyncio.gather(*(contextualize(i, q) for i, q in enumerate(questions)))
        encoding_started = perf_counter()
        if documents:
            try:
                valid = [q for q in queries if isinstance(q, str)]
                vectors = iter(
                    await asyncio.to_thread(self.models.encode_questions, valid) if valid else []
                )
                encoded = [q if isinstance(q, AppError) else next(vectors) for q in queries]
            except Exception:
                encoded = [
                    q
                    if isinstance(q, AppError)
                    else AppError(503, "RETRIEVAL_ERROR", "Question encoding failed")
                    for q in queries
                ]
        else:
            encoded = [None] * len(questions)
        encoding_ms = (perf_counter() - encoding_started) * 1000
        return await asyncio.gather(
            *(
                self.answer_one(
                    chat_id,
                    question,
                    vector,
                    index,
                    started,
                    max(0, encoding_ms),
                    queries[index] if isinstance(queries[index], str) else None,
                    context_ms[index],
                )
                for index, (question, vector) in enumerate(zip(questions, encoded, strict=True))
            )
        )

    async def answer_one(
        self,
        chat_id,
        question,
        encoded,
        index,
        started=None,
        encoding_ms=0.0,
        resolved_question=None,
        contextualization_ms=0.0,
    ):
        start = started if started is not None else perf_counter()
        result = AnswerResult(question=question)
        query = resolved_question or question
        if query != question:
            result.resolved_question = query
        timing = {
            "contextualization_ms": contextualization_ms,
            "encoding_ms": encoding_ms,
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "expansion_ms": 0.0,
            "generation_ms": 0.0,
        }
        candidate_count, evidence = 0, []
        stage = "retrieval"
        try:
            if isinstance(encoded, AppError):
                raise encoded
            if encoded is not None:
                async with self.retrieval_semaphore:
                    tick = perf_counter()
                    candidates, metrics = await self.retriever.retrieve(chat_id, encoded)
                    timing.update(metrics)
                    timing["retrieval_ms"] = (perf_counter() - tick) * 1000
                    candidate_count = len(candidates)
                    tick = perf_counter()
                    ranked = await self.reranker.rank(query, candidates)
                    timing["rerank_ms"] = (perf_counter() - tick) * 1000
                    if answerable(ranked, self.settings.answerability_threshold):
                        # Weak candidates must not consume context or drive parent expansion.
                        ranked = [
                            c
                            for c in ranked
                            if c.reranker_score >= self.settings.answerability_threshold
                        ]
                        tick = perf_counter()
                        evidence = await asyncio.to_thread(self.expander.expand, chat_id, ranked)
                        timing["expansion_ms"] = (perf_counter() - tick) * 1000
                if evidence:
                    stage = "generation"
                    async with self.llm_semaphore:
                        tick = perf_counter()
                        try:
                            if contextualization_ms:
                                model_result = await self.generator.answer(
                                    query,
                                    evidence,
                                    timeout_seconds=max(
                                        0,
                                        self.settings.openai_timeout_seconds
                                        - contextualization_ms / 1000,
                                    ),
                                )
                            else:
                                model_result = await self.generator.answer(query, evidence)
                            result.supported, result.answer, result.citations = validate_answer(
                                model_result,
                                evidence,
                            )
                            if result.supported:
                                result.comments = model_result.comments
                                result.confidence = model_result.confidence
                        finally:
                            timing["generation_ms"] = (perf_counter() - tick) * 1000
        except AppError as exc:
            result.error = QuestionError(code=exc.code, message=exc.message, http_status=exc.status)
        except Exception:
            result.error = QuestionError(
                code="RETRIEVAL_ERROR" if stage == "retrieval" else "LLM_PROVIDER_ERROR",
                message="Evidence retrieval failed"
                if stage == "retrieval"
                else "Answer generation failed",
                http_status=503 if stage == "retrieval" else 502,
            )
        timing["total_ms"] = (perf_counter() - start) * 1000
        result.timing = {k: round(v, 3) for k, v in timing.items()}
        logger.info(
            "question_answered",
            extra={
                "metrics": {
                    "chat_id": chat_id,
                    "question_id": f"q_{index + 1}",
                    **result.timing,
                    "candidate_count": candidate_count,
                    "final_evidence_count": len(evidence),
                    "context_tokens": self.expander.tokens(evidence) if evidence else 0,
                    "supported": result.supported,
                    "error_code": result.error.code if result.error else None,
                }
            },
        )
        return result
