import asyncio
import threading
from pathlib import Path

import httpx
import pytest

from app.api.errors import AppError
from app.config import Settings
from app.ingestion.docling_parser import DoclingParser
from app.ingestion.json_parser import parse_json
from app.main import BodyLimitMiddleware
from app.storage.repository import Repository


def test_json_numbers_preserve_precision_without_float_overflow():
    nodes = parse_json(
        b'{"number":1e9999,"precise":0.1234567890123456789}',
        "chat",
        "doc",
        Settings(_env_file=None),
    )
    assert nodes[-2].text == "number: 1E+9999"
    assert nodes[-1].text == "precise: 0.1234567890123456789"


def test_recovery_leaves_ready_documents_intact(tmp_path):
    repo = Repository(tmp_path / "db.sqlite")
    chat = repo.create_chat()
    ready, _ = repo.reserve_document(chat, "ready.json", "application/json", "first")
    partial, _ = repo.reserve_document(chat, "partial.json", "application/json", "second")
    repo.set_status(chat, ready.id, "READY")
    repo.set_status(chat, partial.id, "INDEXING")
    repo.close()
    restarted = Repository(tmp_path / "db.sqlite")
    restarted.recover_interrupted_ingestion()
    assert [d.status for d in restarted.documents(chat)] == ["READY", "FAILED"]
    assert restarted.ready_ids(chat) == [ready.id]
    restarted.close()


def test_pdf_parser_timeout_terminates_worker(monkeypatch):
    state = {"alive": True, "terminated": False, "closed": 0}

    class Connection:
        def close(self):
            state["closed"] += 1

        def poll(self, timeout):
            assert 0 < timeout <= 0.01
            return False

    class Process:
        def start(self):
            pass

        def is_alive(self):
            return state["alive"]

        def terminate(self):
            state.update(alive=False, terminated=True)

        def join(self, timeout=None):
            pass

        def close(self):
            state["closed"] += 1

    class Context:
        def Pipe(self, duplex):
            assert duplex is False
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(
        "app.ingestion.docling_parser.multiprocessing.get_context", lambda _: Context()
    )
    with pytest.raises(AppError) as error:
        DoclingParser(Settings(_env_file=None, pdf_parse_timeout_seconds=0.01)).parse(
            Path("x.pdf"), "c", "d"
        )
    assert error.value.code == "PDF_PARSE_TIMEOUT"
    assert state["terminated"] and state["closed"] == 3


async def test_pdf_cancellation_waits_for_parser_exit(monkeypatch):
    parser = DoclingParser(Settings(_env_file=None))
    started, stopped = threading.Event(), threading.Event()

    def wait_for_cancellation(path, chat_id, document_id, cancelled):
        started.set()
        cancelled.wait(timeout=2)
        stopped.set()
        raise AppError(499, "PDF_PARSE_CANCELLED", "PDF parsing was cancelled")

    monkeypatch.setattr(parser, "parse", wait_for_cancellation)
    task = asyncio.create_task(parser.parse_async(Path("x.pdf"), "chat", "doc"))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


async def test_chunked_json_body_limit(system):
    chat = system.repo.create_chat()

    async def stream():
        yield b'{"questions":["' + b"x" * 60
        yield b'"]}'

    limited = BodyLimitMiddleware(system.app, 32)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=limited), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/chats/{chat}/questions",
            content=stream(),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 413


async def test_chunked_multipart_body_limit(system):
    chat = system.repo.create_chat()
    body = (
        b'--boundary\r\nContent-Disposition: form-data; name="documents"; filename="x.json"\r\n'
        b'Content-Type: application/json\r\n\r\n{"x":"' + b"x" * 128 + b'"}\r\n--boundary--\r\n'
    )

    async def stream():
        for offset in range(0, len(body), 16):
            yield body[offset : offset + 16]

    limited = BodyLimitMiddleware(system.app, 64)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=limited), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/v1/chats/{chat}/documents",
            content=stream(),
            headers={"content-type": "multipart/form-data; boundary=boundary"},
        )
    assert response.status_code == 413
    assert not system.repo.documents(chat)


async def test_batch_encoding_failure_has_per_question_timings(system, monkeypatch):
    chat = system.repo.create_chat()
    await system.docs.ingest(chat, "source.json", "application/json", b'{"cloud":"AWS"}')

    def fail(*args):
        raise RuntimeError("secret details")

    monkeypatch.setattr(system.models, "encode_questions", fail)
    answers = await system.qa.answer_many(chat, ["cloud", "region"])
    assert all(a.error.code == "RETRIEVAL_ERROR" and a.timing["total_ms"] >= 0 for a in answers)


def test_calibration_uses_labels_and_conservative_tie_break():
    from scripts.evaluate import calibrate, classification_metrics, percentile

    result = calibrate([True, True, False, False], [3.0, -9.0, -11.0, -12.0])
    assert result["threshold"] == -10.0 and result["answerable_accuracy"] == 1
    metrics = classification_metrics([True, False, False], [True, False, True])
    assert metrics["NOT_FOUND_precision"] == 1.0 and metrics["NOT_FOUND_recall"] == 0.5
    assert percentile([1, 2, 3, 4, 100], 0.95) == 100
