"""Reviewer flow with deterministic HTTP responses; browser checks use the real API."""

import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

st = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
ui = importlib.import_module("streamlit_app")


def test_processing_poll_reruns_only_on_status_change(monkeypatch):
    document = {"id": "doc", "status": "PARSING"}
    current = [document.copy()]
    reruns = []
    monkeypatch.setattr(ui, "api", lambda *args: {"documents": current})
    monkeypatch.setattr(st, "rerun", lambda: reruns.append(True))
    ui.watch_documents.__wrapped__("chat", [document])
    assert not reruns
    current[0]["status"] = "READY"
    ui.watch_documents.__wrapped__("chat", [document])
    assert reruns == [True]


@pytest.mark.parametrize("failure", ["unreachable", "timeout", "http", "invalid"])
def test_api_failures_are_actionable(monkeypatch, failure):
    def request(*args, **kwargs):
        if failure == "unreachable":
            raise httpx.ConnectError("private transport details")
        if failure == "timeout":
            raise httpx.ReadTimeout("private transport details")
        if failure == "http":
            return httpx.Response(
                422, json={"error": {"code": "BAD_DOC", "message": "Invalid PDF"}}
            )
        return httpx.Response(502, text="private proxy details")

    monkeypatch.setattr(httpx, "request", request)
    with pytest.raises(ui.APIError) as error:
        ui.api("GET", "/ready")
    assert "private" not in str(error.value)
    assert str(error.value)


def test_upload_chat_citation_duplicates_and_reset(monkeypatch):
    chats, calls = {}, []
    uploads = []
    questions_upload = None
    downloads = []
    fail_reset = False
    upload_started, allow_upload = threading.Event(), threading.Event()
    allow_upload.set()

    def request(method, url, **kwargs):
        path = httpx.URL(url).path
        calls.append((method, path, kwargs))
        if path == "/ready":
            return httpx.Response(200, json={"answer_provider_configured": True})
        if path == "/api/v1/chats":
            if fail_reset:
                return httpx.Response(503, json={"error": {"message": "Try again"}})
            chat_id = f"chat-{len(chats) + 1}"
            chats[chat_id] = []
            return httpx.Response(201, json={"chat_id": chat_id})
        chat_id = path.split("/")[4]
        docs = chats[chat_id]
        if path.endswith("/documents"):
            if method == "GET":
                return httpx.Response(200, json={"documents": docs})
            filename, data, kind = kwargs["files"]["documents"]
            duplicate = any(doc["filename"] == filename for doc in docs)
            if not duplicate:
                docs.append(
                    {
                        "id": f"doc-{len(docs) + 1}",
                        "filename": filename,
                        "status": "PARSING",
                        "content_type": kind,
                    }
                )
            doc = next(doc for doc in docs if doc["filename"] == filename)
            upload_started.set()
            assert allow_upload.wait(timeout=30)
            doc["status"] = "READY"
            return httpx.Response(
                200,
                json={
                    "documents": [
                        {
                            "document_id": doc["id"],
                            "filename": filename,
                            "status": "READY",
                            "duplicate": duplicate,
                        }
                    ]
                },
            )
        if "files" in kwargs:
            rows = json.loads(kwargs["files"]["questions"][1])
            if not isinstance(rows, list):
                return httpx.Response(
                    400, json={"error": {"code": "INVALID_QUESTIONS_JSON", "message": "Use a list"}}
                )
            return httpx.Response(
                200,
                json=[
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer": "$7,500" if "fee" in row["question"] else "Data-Not-Found",
                        "comments": "Supported by billing.json"
                        if "fee" in row["question"]
                        else "No supporting evidence",
                        "confidence": "high" if "fee" in row["question"] else "low",
                    }
                    for row in rows
                ],
            )
        question = kwargs["json"]["questions"][0]
        supported = bool(docs) and "fee" in question
        error = (
            {"code": "LLM_NOT_CONFIGURED", "message": "Configure provider"}
            if "error" in question
            else None
        )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "question": question,
                        "supported": supported,
                        "answer": "$7,500" if supported else None,
                        "error": error,
                        "citations": [
                            {
                                "document_id": docs[0]["id"],
                                "filename": docs[0]["filename"],
                                "page_start": None,
                                "page_end": None,
                                "heading_path": ["billing", "monthly_support_fee"],
                            }
                        ]
                        if supported
                        else [],
                        "timing": {"total_ms": 1},
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "request", request)
    download_button = st.download_button

    def download(*args, **kwargs):
        downloads.append(kwargs)
        return download_button(*args, **kwargs)

    monkeypatch.setattr(st, "download_button", download)
    # Streamlit AppTest does not expose file-uploader input yet.
    monkeypatch.setattr(
        st,
        "file_uploader",
        lambda label, **kwargs: uploads if label == "PDF or JSON sources" else questions_upload,
    )
    app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py"), default_timeout=15)
    app.run()
    assert not app.exception
    first_chat = app.session_state["chat_id"]
    uploads.append(SimpleNamespace(name="billing.json", getvalue=lambda: b'{"fee":"$7,500"}'))
    allow_upload.clear()
    try:
        next(button for button in app.button if button.label == "Upload & index").click().run()
        assert upload_started.wait(timeout=3)
        assert not app.session_state["upload_job"].done()
        assert next(b for b in app.button if b.label == "Generate batch answers").disabled
    finally:
        allow_upload.set()
    if "upload_job" in app.session_state:
        app.session_state["upload_job"].result(timeout=3)
        app.run()
    assert not app.exception
    assert not next(b for b in app.button if b.label == "Generate batch answers").disabled
    assert len(chats[first_chat]) == 1
    assert "$7,500" in app.code[0].value
    # Re-upload while the scripted uploader retains the file.
    next(button for button in app.button if button.label == "Upload & index").click().run()
    if "upload_job" in app.session_state:
        app.session_state["upload_job"].result(timeout=3)
        app.run()
    assert len(chats[first_chat]) == 1
    assert any("already uploaded" in caption.value for caption in app.caption)
    uploads.append(SimpleNamespace(name="other.json", getvalue=lambda: b'{"other":true}'))
    next(button for button in app.button if button.label == "Upload & index").click().run()
    if "upload_job" in app.session_state:
        app.session_state["upload_job"].result(timeout=3)
        app.run()
    assert len(chats[first_chat]) == 2
    assert app.session_state["preview_id"] == "doc-2"
    uploads.clear()
    app.chat_input[0].set_value("What is the fee?").run()
    assert not app.exception
    assert any(markdown.value == "$7,500" for markdown in app.markdown)
    app.button(key="citation_0_0").click().run()
    assert not app.exception
    assert app.session_state["preview_id"] == "doc-1"
    app.chat_input[0].set_value("Who is the CEO?").run()
    payload = next(c[2]["json"] for c in reversed(calls) if c[1].endswith("/questions"))
    assert payload["previous_questions"] == ["What is the fee?"]
    assert "$7,500" not in str(payload)
    assert any("No supporting evidence" in info.value for info in app.info)
    app.chat_input[0].set_value("Trigger error").run()
    assert any("LLM_NOT_CONFIGURED" in error.value for error in app.error)
    batch = [
        {"id": "fee", "question": "What is the fee?"},
        {"id": "missing", "question": "Who is the CEO?"},
    ]
    questions_upload = SimpleNamespace(name="questions.json", getvalue=lambda: json.dumps(batch))
    next(button for button in app.button if button.label == "Generate batch answers").click().run()
    assert not app.exception
    records = app.session_state["batch_result"]["records"]
    assert [row["id"] for row in records] == ["fee", "missing"]
    assert records[1]["answer"] == "Data-Not-Found"
    assert json.loads(downloads[-1]["data"]) == records
    assert downloads[-1]["mime"] == "application/json"
    questions_upload = SimpleNamespace(name="invalid.json", getvalue=lambda: "{}")
    next(button for button in app.button if button.label == "Generate batch answers").click().run()
    assert any("INVALID_QUESTIONS_JSON" in error.value for error in app.error)
    assert app.session_state["batch_result"]["records"] == records
    fail_reset = True
    next(button for button in app.button if button.label == "New chat").click().run()
    assert not app.exception
    assert app.session_state["chat_id"] == first_chat
    assert len(app.session_state["sources"]) == 2
    assert len(app.session_state["messages"]) == 3
    assert app.session_state["batch_result"]["records"] == records
    fail_reset = False
    next(button for button in app.button if button.label == "New chat").click().run()
    assert not app.exception
    assert app.session_state["chat_id"] != first_chat
    assert not app.session_state["sources"] and not app.session_state["messages"]
    assert app.session_state["batch_result"] is None
    assert len(chats[first_chat]) == 2  # Reset never deletes the prior backend session.
    app.chat_input[0].set_value("What is the fee?").run()
    payload = next(c[2]["json"] for c in reversed(calls) if c[1].endswith("/questions"))
    assert payload["previous_questions"] == []
    assert any("No supporting evidence" in info.value for info in app.info)
    assert calls[-1][1].startswith(f"/api/v1/chats/{app.session_state['chat_id']}/")
