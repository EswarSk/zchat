import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from app.api.errors import AppError
from app.config import Settings
from app.domain.models import Evidence, ModelCitation, ModelGroundedAnswer
from app.generation.grounded_answer import (
    AnswerVerification,
    ContextualizedQuestion,
    GroundedGenerator,
)
from app.generation.validation import validate_answer


@pytest.fixture
def evidence():
    return [
        Evidence(
            evidence_id="E1",
            document_id="document",
            filename="policy.pdf",
            node_id="node",
            parent_id=None,
            page_start=37,
            page_end=37,
            heading_path=["Infrastructure"],
            text="Production runs on AWS.",
        )
    ]


@pytest.mark.parametrize(
    "model_answer",
    [
        ModelGroundedAnswer(
            supported=True, answer="AWS", citations=[ModelCitation(evidence_id="E999")]
        ),
        ModelGroundedAnswer(supported=True, answer="AWS", citations=[]),
        ModelGroundedAnswer(
            supported=True, answer=None, citations=[ModelCitation(evidence_id="E1")]
        ),
        ModelGroundedAnswer(
            supported=True, answer=" ", citations=[ModelCitation(evidence_id="E1")]
        ),
        ModelGroundedAnswer(
            supported=False, answer="fabricated", citations=[ModelCitation(evidence_id="E999")]
        ),
        ModelGroundedAnswer(
            supported=True,
            answer="AWS",
            confidence="low",
            citations=[ModelCitation(evidence_id="E1")],
        ),
    ],
)
def test_citation_validation_fails_closed(model_answer, evidence):
    assert validate_answer(model_answer, evidence) == (False, None, [])


def test_citation_metadata_is_from_trusted_evidence(evidence):
    output = ModelGroundedAnswer(
        supported=True,
        answer="AWS",
        citations=[
            ModelCitation(evidence_id="E1"),
            ModelCitation(evidence_id="E1"),
        ],
    )
    supported, answer, citations = validate_answer(output, evidence)
    assert supported and answer == "AWS" and len(citations) == 1
    assert citations[0].filename == "policy.pdf" and citations[0].page_start == 37


async def test_real_openai_sdk_structured_responses_request(evidence):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": json.dumps(
                                    {
                                        "supported": True,
                                        "answer": "AWS",
                                        "citations": [{"evidence_id": "E1"}],
                                    }
                                    if "assessment"
                                    not in body["text"]["format"]["schema"]["properties"]
                                    else {
                                        "assessment": "The source explicitly names AWS.",
                                        "supported": True,
                                        "answer": "AWS",
                                        "citations": [{"evidence_id": "E1"}],
                                    }
                                ),
                            }
                        ],
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    generator = GroundedGenerator(Settings(_env_file=None), client)
    result = await generator.answer("Which provider?", evidence)
    assert result.supported and result.answer == "AWS"
    body = requests[0]
    assert body["model"] == "gpt-4o-mini" and body["store"] is False
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert not body.get("tools") and not body.get("previous_response_id")
    assert [m["role"] for m in body["input"]] == ["user"]
    assert "raw_source_text" in body["input"][0]["content"]
    assert "Production runs on AWS." in body["input"][0]["content"]
    assert len(requests) == 2
    verification = requests[1]
    assert verification["model"] == "gpt-4o-mini"
    assert "reasoning" not in verification
    assert "PROPOSED ANSWER" in verification["input"][0]["content"]
    assert "Production runs on AWS." in verification["input"][0]["content"]
    assert set(verification["text"]["format"]["schema"]["properties"]) == {
        "assessment",
        "supported",
        "answer",
        "citations",
        "comments",
        "confidence",
    }
    assert verification["store"] is False and not verification.get("tools")
    await generator.close()


@pytest.mark.parametrize("checked_id, supported", [("E2", True), ("E999", False)])
async def test_verified_citations_replace_draft_and_remain_scoped(evidence, checked_id, supported):
    evidence.append(evidence[0].model_copy(update={"evidence_id": "E2", "node_id": "heading"}))

    async def parse(**kwargs):
        if kwargs["text_format"] is AnswerVerification:
            output = AnswerVerification(
                assessment="The answer is supported by the second source record.",
                supported=True,
                answer="AWS",
                citations=[ModelCitation(evidence_id=checked_id)],
                comments="Production runs on AWS.",
                confidence="high",
            )
        else:
            output = ModelGroundedAnswer(
                supported=True,
                answer="AWS",
                citations=[ModelCitation(evidence_id="E1")],
                comments="Unsupported draft wording.",
            )
        return SimpleNamespace(status="completed", output_parsed=output)

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = await generator.answer("Which provider?", evidence)
    assert [c.evidence_id for c in result.citations] == [checked_id]
    assert result.comments == "Production runs on AWS." and result.confidence == "high"
    assert validate_answer(result, evidence)[0] is supported


@pytest.mark.parametrize(
    "question, supported", [("Which cloud provider?", True), ("Which industry prize?", False)]
)
async def test_review_removes_embellishment_without_answering_missing_fact(
    evidence, question, supported
):
    draft = ModelGroundedAnswer(
        supported=True,
        answer="Production runs on AWS and won an industry prize.",
        citations=[ModelCitation(evidence_id="E1")],
    )

    async def parse(**kwargs):
        if kwargs["text_format"] is AnswerVerification:
            assert json.dumps(question) in kwargs["input"][0]["content"]
            output = AnswerVerification(
                assessment="AWS is stated; no prize is listed.",
                supported=supported,
                answer="Production runs on AWS." if supported else None,
                citations=[ModelCitation(evidence_id="E1")] if supported else [],
            )
        else:
            output = draft
        return SimpleNamespace(status="completed", output_parsed=output)

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = await generator.answer(question, evidence)
    assert validate_answer(result, evidence)[0] is supported
    assert result.answer == ("Production runs on AWS." if supported else None)
    assert "prize" not in (result.answer or "")


async def test_real_sdk_network_timeout_is_sanitized(evidence):
    async def handle(request):
        raise httpx.ReadTimeout("secret provider detail", request=request)

    client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    generator = GroundedGenerator(Settings(_env_file=None), client)
    with pytest.raises(AppError) as error:
        await generator.answer("Which provider?", evidence)
    assert error.value.code == "LLM_TIMEOUT" and error.value.status == 504
    assert "secret" not in error.value.message
    await generator.close()


async def test_refusal_and_incomplete_output_are_unsupported(evidence):
    async def parse(**kwargs):
        return SimpleNamespace(status="incomplete", output_parsed=None)

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = await generator.answer("Which provider?", evidence)
    assert not result.supported and result.answer is None and not result.citations


@pytest.mark.parametrize(
    "verdict",
    [
        SimpleNamespace(
            status="completed",
            output_parsed=AnswerVerification(
                assessment="The dates contradict the claim.",
                supported=False,
                answer=None,
                citations=[],
            ),
        ),
        SimpleNamespace(status="incomplete", output_parsed=None),
        SimpleNamespace(status="completed", output_parsed={"unexpected": True}),
    ],
)
async def test_valid_citation_does_not_release_unsupported_claims(evidence, verdict):
    evidence[0].text = "Company A: Jan 2026 - Present. Company B: Nov 2020 - Jul 2022."
    calls = []
    wrong = ModelGroundedAnswer(
        supported=True,
        answer="Company A employed the person before Company B, until November 2020.",
        citations=[ModelCitation(evidence_id="E1")],
    )

    async def parse(**kwargs):
        calls.append(kwargs)
        if kwargs["text_format"] is AnswerVerification:
            content = kwargs["input"][0]["content"]
            assert wrong.answer in content and evidence[0].text in content
            return verdict
        return SimpleNamespace(status="completed", output_parsed=wrong)

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    result = await generator.answer("What happened before Company B?", evidence)
    assert validate_answer(result, evidence) == (False, None, [])
    assert len(calls) == 2


async def test_verification_timeout_does_not_release_draft(evidence):
    async def parse(**kwargs):
        if kwargs["text_format"] is AnswerVerification:
            raise TimeoutError
        return SimpleNamespace(
            status="completed",
            output_parsed=ModelGroundedAnswer(
                supported=True, answer="AWS", citations=[ModelCitation(evidence_id="E1")]
            ),
        )

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    with pytest.raises(AppError) as error:
        await generator.answer("Which provider?", evidence)
    assert error.value.code == "LLM_TIMEOUT"


@pytest.mark.parametrize(
    "output, error_code",
    [
        ({"question": "Where did Alex earn the master degree?"}, None),
        ({"question": None}, "QUESTION_NEEDS_CONTEXT"),
        ({"unexpected": "value"}, "QUESTION_CONTEXT_ERROR"),
    ],
)
async def test_user_question_context_is_intent_only_and_fails_on_ambiguity(output, error_code):
    async def parse(**kwargs):
        assert kwargs["text_format"] is ContextualizedQuestion
        assert kwargs["model"] == "gpt-4o-mini"
        assert "reasoning" not in kwargs
        assert kwargs["store"] is False and not kwargs.get("tools")
        content = json.loads(kwargs["input"][0]["content"])
        assert content == {
            "CURRENT QUESTION": "Where did he earn his master degree?",
            "PREVIOUS USER QUESTIONS": ["What does Alex do?"],
        }
        return SimpleNamespace(status="completed", output_parsed=output)

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    if error_code:
        with pytest.raises(AppError) as error:
            await generator.contextualize(
                "Where did he earn his master degree?", ["What does Alex do?"]
            )
        assert error.value.code == error_code
    else:
        assert (
            await generator.contextualize(
                "Where did he earn his master degree?", ["What does Alex do?"]
            )
            == output["question"]
        )


async def test_generation_honors_remaining_provider_budget(evidence):
    async def parse(**kwargs):
        await asyncio.sleep(0.05)

    generator = GroundedGenerator(
        Settings(_env_file=None), SimpleNamespace(responses=SimpleNamespace(parse=parse))
    )
    with pytest.raises(AppError) as error:
        await generator.answer("Which provider?", evidence, timeout_seconds=0.001)
    assert error.value.code == "LLM_TIMEOUT"
