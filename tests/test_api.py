import asyncio
import json

import pytest

from app.api.errors import AppError
from tests.helpers import pdf_bytes


async def chat(system):
    response = await system.client.post("/api/v1/chats")
    assert response.status_code == 201
    return response.json()["chat_id"]


async def upload(system, chat_id, filename, data, content_type):
    return await system.client.post(
        f"/api/v1/chats/{chat_id}/documents", files={"documents": (filename, data, content_type)}
    )


async def ask(system, chat_id, questions):
    return await system.client.post(
        f"/api/v1/chats/{chat_id}/questions", json={"questions": questions}
    )


async def test_health_and_empty_chat_abstention(system):
    assert (await system.client.get("/health")).json() == {"status": "ok"}
    assert (await system.client.get("/ready")).status_code == 200
    chat_id = await chat(system)
    response = await ask(system, chat_id, ["Who is the CEO?"])
    answer = response.json()["results"][0]
    assert answer["supported"] is False and answer["answer"] is None and answer["citations"] == []
    assert system.models.question_calls == 0 and not system.generator.calls


async def test_follow_up_resolution_precedes_retrieval_and_keeps_scope(system, monkeypatch):
    own = await chat(system)
    other = await chat(system)
    await upload(
        system, own, "own.json", b'{"Alex":{"master":"University North"}}', "application/json"
    )
    await upload(
        system, other, "other.json", b'{"Alex":{"master":"SECRET-OTHER"}}', "application/json"
    )
    seen = []

    async def contextualize(question, previous):
        seen.append((question, previous))
        return "Where did Alex earn the master degree?"

    monkeypatch.setattr(system.generator, "contextualize", contextualize)
    payload = {
        "questions": ["Where did he earn his master degree?"],
        "previous_questions": ["What does Alex do?"],
    }
    response = await system.client.post(f"/api/v1/chats/{own}/questions", json=payload)
    result = response.json()["results"][0]
    assert seen == [(payload["questions"][0], payload["previous_questions"])]
    assert result["question"] == payload["questions"][0]
    assert result["resolved_question"] == "Where did Alex earn the master degree?"
    assert result["supported"] and "University North" in result["answer"]
    assert "SECRET-OTHER" not in result["answer"]
    assert system.generator.calls[0][0] == result["resolved_question"]
    assert all(e.document_id in system.repo.ready_ids(own) for e in system.generator.calls[0][1])
    empty = await chat(system)
    result = (await system.client.post(f"/api/v1/chats/{empty}/questions", json=payload)).json()[
        "results"
    ][0]
    assert not result["supported"] and len(seen) == 1


async def test_ambiguous_reference_is_actionable_and_does_not_fail_siblings(system, monkeypatch):
    own = await chat(system)
    await upload(system, own, "cloud.json", b'{"cloud":"AWS"}', "application/json")

    async def contextualize(question, previous):
        if "he" in question:
            raise AppError(400, "QUESTION_NEEDS_CONTEXT", "Please name who you mean")
        return question

    monkeypatch.setattr(system.generator, "contextualize", contextualize)
    response = await system.client.post(
        f"/api/v1/chats/{own}/questions",
        json={
            "questions": ["Where did he work?", "Which cloud provider?"],
            "previous_questions": ["What do Alex and Casey do?"],
        },
    )
    unclear, valid = response.json()["results"]
    assert unclear["error"]["code"] == "QUESTION_NEEDS_CONTEXT" and not unclear["citations"]
    assert valid["supported"] and "AWS" in valid["answer"]


@pytest.mark.parametrize("context", [None, "Who is Alex?", [7], [""], ["x"] * 6, ["x" * 2001]])
async def test_previous_questions_are_bounded_user_text(system, context):
    own = await chat(system)
    response = await system.client.post(
        f"/api/v1/chats/{own}/questions",
        json={"questions": ["Where did he work?"], "previous_questions": context},
    )
    assert response.status_code == 400
    assert not system.generator.calls and system.models.question_calls == 0


async def test_pdf_upload_index_answer_and_page_provenance(system):
    chat_id = await chat(system)
    response = await upload(
        system,
        chat_id,
        "policy.pdf",
        pdf_bytes(["Cover", "Cloud provider is AWS."]),
        "application/pdf",
    )
    assert response.status_code == 200
    assert response.json()["documents"][0]["status"] == "READY"
    response = await ask(system, chat_id, ["What cloud provider is used?"])
    answer = response.json()["results"][0]
    assert answer["supported"] and "AWS" in answer["answer"]
    # Small same-section nodes merge; citation covers the correct raw source page range.
    citation = answer["citations"][0]
    assert citation["page_start"] <= 2 <= citation["page_end"]
    assert citation["filename"] == "policy.pdf"
    assert answer["timing"]["total_ms"] >= answer["timing"]["generation_ms"]


async def test_background_pdf_upload_is_bounded_and_keeps_api_responsive(system, monkeypatch):
    chat_id = await chat(system)
    release = asyncio.Event()
    two_started = asyncio.Event()
    original = system.parser.parse_async
    started = 0

    async def slow_parse(*args):
        nonlocal started
        started += 1
        if started == 2:
            two_started.set()
        await release.wait()
        return await original(*args)

    monkeypatch.setattr(system.parser, "parse_async", slow_parse)
    payloads = {name: pdf_bytes([name]) for name in ("first", "second")}
    for name in ("first", "second"):
        response = await system.client.post(
            f"/api/v1/chats/{chat_id}/documents?background=true",
            files={"documents": (f"{name}.pdf", payloads[name], "application/pdf")},
        )
        assert response.status_code == 202
        assert response.json()["documents"][0]["status"] == "PENDING"
    await asyncio.wait_for(two_started.wait(), 2)
    assert (await ask(system, chat_id, ["What is first?"])).status_code == 409

    duplicate = await system.client.post(
        f"/api/v1/chats/{chat_id}/documents?background=true",
        files={"documents": ("first.pdf", payloads["first"], "application/pdf")},
    )
    assert duplicate.status_code == 202 and duplicate.json()["documents"][0]["duplicate"]
    busy = await system.client.post(
        f"/api/v1/chats/{chat_id}/documents?background=true",
        files={"documents": ("third.pdf", pdf_bytes(["third"]), "application/pdf")},
    )
    assert busy.status_code == 429 and busy.json()["error"]["code"] == "INGESTION_BUSY"
    release.set()
    async with asyncio.timeout(3):
        while len(system.repo.ready_ids(chat_id)) != 2:
            await asyncio.sleep(0.01)
    assert started == 2
    assert {d.status for d in system.repo.documents(chat_id)} == {"READY", "FAILED"}


async def test_background_pdf_failure_is_visible_in_document_status(system):
    chat_id = await chat(system)
    response = await system.client.post(
        f"/api/v1/chats/{chat_id}/documents?background=true",
        files={"documents": ("broken.pdf", b"invalid", "application/pdf")},
    )
    assert response.status_code == 202
    async with asyncio.timeout(3):
        while system.repo.documents(chat_id)[0].status != "FAILED":
            await asyncio.sleep(0.01)
    assert system.repo.documents(chat_id)[0].error_message == "DOCUMENT_PARSE_FAILED"


async def test_background_pdf_shutdown_marks_interrupted_upload_failed(system, monkeypatch):
    chat_id = await chat(system)
    started = asyncio.Event()

    async def slow_parse(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(system.parser, "parse_async", slow_parse)
    response = await system.client.post(
        f"/api/v1/chats/{chat_id}/documents?background=true",
        files={"documents": ("slow.pdf", pdf_bytes(["slow"]), "application/pdf")},
    )
    assert response.status_code == 202
    await asyncio.wait_for(started.wait(), 2)
    await system.docs.close()
    document = system.repo.documents(chat_id)[0]
    assert document.status == "FAILED" and document.error_message == "INGESTION_INTERRUPTED"


async def test_json_source_multi_question_file(system):
    chat_id = await chat(system)
    assert (
        await upload(
            system,
            chat_id,
            "policy.json",
            b'{"cloud":"AWS","region":"us-east-1"}',
            "application/json",
        )
    ).status_code == 200
    response = await system.client.post(
        f"/api/v1/chats/{chat_id}/questions",
        files={
            "questions": (
                "questions.json",
                json.dumps(["What cloud provider?", "What region?"]),
                "application/json",
            ),
        },
    )
    assert response.status_code == 200
    answers = response.json()["results"]
    assert all(a["supported"] for a in answers)
    assert answers[0]["citations"][0]["page_start"] is None
    assert [a["question"] for a in answers] == ["What cloud provider?", "What region?"]


async def test_mandatory_chat_isolation_direct_candidates(system):
    a, b = await chat(system), await chat(system)
    for chat_id, code in [(a, "ALPHA-123"), (b, "BETA-456")]:
        response = await upload(
            system,
            chat_id,
            "code.pdf",
            pdf_bytes([f"The launch code is {code}."]),
            "application/pdf",
        )
        assert response.status_code == 200
    encoded = system.models.encode_questions(["What is the launch code?"])[0]
    candidates, _ = await system.qa.retriever.retrieve(b, encoded)
    assert candidates
    assert all(c.node.chat_id == b and "ALPHA-123" not in c.node.text for c in candidates)
    assert any("BETA-456" in c.node.text for c in candidates)
    response = await ask(system, b, ["What is the launch code?"])
    assert "BETA-456" in response.json()["results"][0]["answer"]


async def test_mandatory_not_found_does_not_call_llm(system):
    chat_id = await chat(system)
    await upload(system, chat_id, "source.json", b'{"cloud":"AWS"}', "application/json")
    response = await ask(system, chat_id, ["Who is the CEO?"])
    answer = response.json()["results"][0]
    assert answer["supported"] is False and answer["answer"] is None and answer["citations"] == []
    assert not system.generator.calls


async def test_mandatory_fabricated_citation_rejected(system):
    chat_id = await chat(system)
    await upload(system, chat_id, "source.json", b'{"cloud":"AWS"}', "application/json")
    system.generator.fabricate = True
    answer = (await ask(system, chat_id, ["What cloud provider?"])).json()["results"][0]
    assert answer["supported"] is False and answer["answer"] is None and answer["citations"] == []


async def test_incremental_pdf_ingestion_and_duplicate_index_entries(system):
    chat_id = await chat(system)
    pdf_a = pdf_bytes(["Cloud provider is AWS."])
    first = (await upload(system, chat_id, "a.pdf", pdf_a, "application/pdf")).json()["documents"][
        0
    ]
    await ask(system, chat_id, ["What cloud provider?"])
    assert system.parser.calls == 1 and system.models.document_calls == 1
    await upload(
        system, chat_id, "b.pdf", pdf_bytes(["Recovery RTO is 4 hours."]), "application/pdf"
    )
    await ask(system, chat_id, ["What is the RTO?"])
    assert system.parser.calls == 2 and system.models.document_calls == 2
    duplicate = (await upload(system, chat_id, "renamed.pdf", pdf_a, "application/pdf")).json()[
        "documents"
    ][0]
    assert duplicate["duplicate"] and duplicate["document_id"] == first["document_id"]
    assert system.parser.calls == 2 and system.models.document_calls == 2
    points = await system.store.client.count(system.store.collection, exact=True)
    assert points.count == 2


async def test_concurrent_duplicates_parse_once(system):
    chat_id = await chat(system)
    data = pdf_bytes(["Launch code is BETA-456."])
    results = await asyncio.gather(
        *[system.docs.ingest(chat_id, "x.pdf", "application/pdf", data) for _ in range(8)]
    )
    assert sum(not r["duplicate"] for r in results) == 1
    assert len({r["document_id"] for r in results}) == 1
    assert system.parser.calls == 1


async def test_failed_upload_retry_cleans_partial_index_and_claims_once(system, monkeypatch):
    chat_id = await chat(system)
    data = pdf_bytes(["Launch code is BETA-456."])
    first = await system.docs.ingest(chat_id, "x.pdf", "application/pdf", data)
    system.repo.set_status(chat_id, first["document_id"], "FAILED", error="PDF_PARSE_TIMEOUT")
    deleted = []
    delete = system.store.delete_document

    async def track_delete(chat, document):
        deleted.append(document)
        await delete(chat, document)

    monkeypatch.setattr(system.store, "delete_document", track_delete)
    results = await asyncio.gather(
        *[system.docs.ingest(chat_id, "x.pdf", "application/pdf", data) for _ in range(8)]
    )
    assert sum(not r["duplicate"] for r in results) == 1
    assert {r["document_id"] for r in results} == {first["document_id"]}
    assert system.parser.calls == 2 and deleted == [first["document_id"]]
    assert system.repo.documents(chat_id)[0].status == "READY"
    points = await system.store.client.count(system.store.collection, exact=True)
    assert points.count == 1


async def test_timeout_is_per_question_and_other_answers_survive(system):
    chat_id = await chat(system)
    await upload(
        system, chat_id, "source.json", b'{"cloud":"AWS","region":"us-east-1"}', "application/json"
    )
    system.generator.timeout_question = "What cloud provider?"
    response = await ask(system, chat_id, ["What cloud provider?", "What region?"])
    assert response.status_code == 200
    first, second = response.json()["results"]
    assert first["error"]["code"] == "LLM_TIMEOUT" and first["error"]["http_status"] == 504
    assert not first["supported"] and second["supported"]


async def test_invalid_pdf_records_failed_status(system):
    chat_id = await chat(system)
    response = await upload(system, chat_id, "invalid.pdf", b"invalid", "application/pdf")
    assert response.status_code == 422
    assert system.repo.documents(chat_id)[0].status == "FAILED"
    assert (await ask(system, chat_id, ["What code?"])).status_code == 409


async def test_malformed_question_file_and_limits(system):
    chat_id = await chat(system)
    response = await system.client.post(
        f"/api/v1/chats/{chat_id}/questions",
        files={
            "questions": ("questions.json", b"{", "application/json"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_QUESTIONS_JSON"
    assert (await ask(system, chat_id, ["q"] * 51)).status_code == 429
    assert (await ask(system, "missing", ["q"])).status_code == 404


async def test_challenge_endpoint_reuses_services(system):
    response = await system.client.post(
        "/api/v1/qa",
        files={
            "document": ("source.json", b'{"cloud":"AWS"}', "application/json"),
            "questions": (
                "questions.json",
                b'["What cloud provider?","Who is the CEO?"]',
                "application/json",
            ),
        },
    )
    assert response.status_code == 200
    first, second = response.json()["results"]
    assert first["supported"] and "AWS" in first["answer"]
    assert not second["supported"]


async def test_file_size_and_unsupported_type(system):
    chat_id = await chat(system)
    system.settings.max_upload_bytes = 5
    response = await upload(system, chat_id, "source.json", b'{"a":1}', "application/json")
    assert response.status_code == 413
    response = await upload(system, chat_id, "source.txt", b"a", "text/plain")
    assert response.status_code == 400
