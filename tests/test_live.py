"""Opt-in real Docling, FastEmbed, Qdrant-server and SDK path; OpenAI HTTP is mocked."""

import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from openai import AsyncOpenAI

from app.config import Settings
from app.domain.models import DocumentNode
from app.generation.grounded_answer import GroundedGenerator
from app.main import build_services, create_app
from tests.helpers import pdf_bytes

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="Set RUN_LIVE_TESTS=1 with Qdrant running",
    ),
]


async def test_real_pdf_json_isolation_retrieval_and_mocked_responses(tmp_path):
    settings = Settings(
        _env_file=None,
        database_path=tmp_path / "metadata.sqlite3",
        qdrant_collection=f"zchat_test_{uuid4().hex}",
        model_cache_dir="models",
    )
    bundle = await build_services(settings)
    captured = []

    def provider(request):
        payload = json.loads(request.content)
        captured.append(payload)
        evidence = json.loads(payload["input"][0]["content"].split("EVIDENCE\n", 1)[1])
        text = "\n".join(e["raw_source_text"] for e in evidence)
        assert "ALPHA-123" not in text
        if "assessment" in payload["text"]["format"]["schema"]["properties"]:
            proposal = json.loads(
                payload["input"][0]["content"]
                .split("PROPOSED ANSWER\n", 1)[1]
                .split("\n\nEVIDENCE", 1)[0]
            )
            result = {
                "assessment": "Mock transport check.",
                "supported": True,
                "answer": proposal["answer"],
                "citations": proposal["citations"],
            }
        else:
            question = json.loads(
                payload["input"][0]["content"].split("QUESTION\n", 1)[1].split("\n\nEVIDENCE", 1)[0]
            )
            expected = (
                "4 hours"
                if "RTO" in question
                else ("AWS" if "AWS" in question or "cloud provider" in question else "BETA-456")
            )
            match = next(e for e in evidence if expected in e["raw_source_text"])
            result = {
                "supported": True,
                "answer": match["raw_source_text"],
                "citations": [{"evidence_id": match["evidence_id"]}],
            }
        return httpx.Response(
            200,
            json={
                "id": "resp_live",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg_live",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "annotations": [], "text": json.dumps(result)}
                        ],
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
    )
    bundle.qa.generator = GroundedGenerator(settings, client)
    try:
        app = create_app(settings, bundle)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as api:
                chats = [(await api.post("/api/v1/chats")).json()["chat_id"] for _ in range(2)]
                for chat, code in zip(chats, ["ALPHA-123", "BETA-456"], strict=True):
                    response = await api.post(
                        f"/api/v1/chats/{chat}/documents",
                        files={
                            "documents": (
                                "code.pdf",
                                pdf_bytes([f"The launch code is {code}."]),
                                "application/pdf",
                            ),
                        },
                    )
                    assert response.status_code == 200, response.text
                chat = chats[1]
                vectors = bundle.qa.models.encode_questions(["What is the launch code?"])
                candidates, _ = await bundle.qa.retriever.retrieve(chat, vectors[0])
                assert candidates and all("ALPHA-123" not in c.node.text for c in candidates)
                assert any("BETA-456" in c.node.text for c in candidates)
                assert all(c.node.page_start == c.node.page_end == 1 for c in candidates)
                response = await api.post(
                    f"/api/v1/chats/{chat}/documents",
                    files={
                        "documents": (
                            "cloud.json",
                            b'{"hosting":{"cloud":"AWS","region":"us-east-1"}}',
                            "application/json",
                        ),
                    },
                )
                assert response.status_code == 200, response.text
                before = (
                    await bundle.store.client.count(bundle.store.collection, exact=True)
                ).count
                duplicate = await bundle.documents.ingest(
                    chat, "copy.pdf", "application/pdf", pdf_bytes(["The launch code is BETA-456."])
                )
                assert duplicate["duplicate"]
                assert (
                    await bundle.store.client.count(bundle.store.collection, exact=True)
                ).count == before
                response = await api.post(
                    f"/api/v1/chats/{chat}/questions",
                    json={
                        "questions": [
                            "What is the launch code?",
                            "Where is AWS hosted?",
                            "Who is the CEO?",
                        ]
                    },
                )
                assert response.status_code == 200, response.text
                launch, cloud, ceo = response.json()["results"]
                assert launch["supported"] and "BETA-456" in launch["answer"]
                assert launch["citations"][0]["page_start"] == 1
                assert cloud["supported"] and "AWS" in cloud["answer"]
                assert not ceo["supported"] and ceo["answer"] is None and ceo["citations"] == []
                assert len(captured) == 4  # Two drafts + two checks; CEO skips the provider.
                assert all(
                    not payload.get("tools") and payload["store"] is False for payload in captured
                )
                # Inspect the canonical parser result independently of generated answers.
                rows = bundle.repository.db.execute(
                    "SELECT data_json FROM document_nodes WHERE chat_id=?", (chat,)
                ).fetchall()
                nodes = [DocumentNode.model_validate_json(row[0]) for row in rows]
                assert any(n.node_type == "document" for n in nodes)
                assert any(n.heading_path[-1:] == ["hosting"] and "AWS" in n.text for n in nodes)
                policy_chat = bundle.repository.create_chat()
                policy = Path(__file__).resolve().parents[1] / "evaluation/policy.pdf"
                document = await bundle.documents.ingest(
                    policy_chat,
                    "policy.pdf",
                    "application/pdf",
                    policy.read_bytes(),
                )
                rows = bundle.repository.db.execute(
                    "SELECT data_json FROM document_nodes WHERE document_id=?",
                    (document["document_id"],),
                ).fetchall()
                nodes = [DocumentNode.model_validate_json(row[0]) for row in rows]
                sections = [n for n in nodes if n.node_type == "section"]
                tables = [n for n in nodes if n.node_type == "table"]
                # Layout detection is probabilistic; preserve the headings Docling actually detects.
                assert len(sections) >= 12
                assert any("RTO" in n.text and "4 hours" in n.text for n in tables)
                assert any(n.heading_path for n in nodes if n.searchable)
                assert all(
                    n.page_start == n.page_end
                    for n in nodes
                    if n.searchable and n.metadata.get("parser") == "pypdf"
                )
                print(
                    "Real fixture tree:",
                    len(sections),
                    "sections,",
                    len(tables),
                    "table nodes,",
                    sum(n.searchable for n in nodes),
                    "retrieval leaves",
                )
                response = await api.post(
                    f"/api/v1/chats/{policy_chat}/questions",
                    json={"questions": ["What is the RTO?"]},
                )
                answer = response.json()["results"][0]
                assert answer["supported"] and "4 hours" in answer["answer"], answer
                assert answer["citations"][0]["page_start"] == 3
                assert answer["citations"][0]["page_end"] == 3
                response = await api.post(
                    f"/api/v1/chats/{policy_chat}/questions?output=questionnaire",
                    files={
                        "questions": (
                            "questions.json",
                            json.dumps(
                                [
                                    {"id": "cloud", "question": "Which cloud provider is used?"},
                                    {"id": "missing", "question": "Who is the CEO?"},
                                ]
                            ),
                            "application/json",
                        )
                    },
                )
                cloud, missing = response.json()
                assert cloud["id"] == "cloud" and "AWS" in cloud["answer"]
                assert "policy.pdf (page 1)" in cloud["comments"]
                assert missing["answer"] == "Data-Not-Found" and missing["confidence"] == "low"
    finally:
        await bundle.store.client.delete_collection(settings.qdrant_collection)
        await bundle.close()
