from pydantic import ValidationError

from app.api.errors import AppError
from app.domain.models import QuestionInput
from app.ingestion.json_parser import strict_json


def normalize_questions(value, settings, *, maximum=None) -> list[QuestionInput]:
    if isinstance(value, dict):
        if set(value) != {"questions"}:
            raise AppError(
                400, "INVALID_QUESTIONS_JSON", "Expected an object with a questions list"
            )
        value = value["questions"]
    if not isinstance(value, list):
        raise AppError(
            400, "INVALID_QUESTIONS_JSON", "Questions must be a list of strings or question records"
        )
    if len(value) > (settings.max_questions_per_request if maximum is None else maximum):
        raise AppError(429, "TOO_MANY_QUESTIONS", "Question count exceeds configured limit")
    if not value:
        raise AppError(400, "INVALID_QUESTIONS_JSON", "Provide at least one nonblank question")
    questions = []
    for row in value:
        # Existing output columns are never intent context or factual evidence.
        if isinstance(row, dict) and set(row) <= {
            "id",
            "question",
            "answer",
            "comments",
            "confidence",
        }:
            row = {k: row[k] for k in ("id", "question") if k in row}
        elif isinstance(row, str):
            row = {"question": row}
        else:
            raise AppError(400, "INVALID_QUESTIONS_JSON", "Use strings or question records")
        try:
            question = QuestionInput.model_validate(row)
        except ValidationError as exc:
            raise AppError(
                400,
                "INVALID_QUESTIONS_JSON",
                "Each record needs question text and an optional string id",
            ) from exc
        if not question.question.strip() or not question.id.strip():
            raise AppError(400, "INVALID_QUESTIONS_JSON", "Questions and IDs cannot be blank")
        if len(question.question) > settings.max_question_characters:
            raise AppError(
                400, "INVALID_QUESTIONS_JSON", "Question exceeds configured character limit"
            )
        question.question = question.question.strip()
        questions.append(question)
    if len({q.id for q in questions}) != len(questions):
        raise AppError(
            400, "INVALID_QUESTIONS_JSON", "Question IDs must be unique within the batch"
        )
    return questions


def question_json(data: bytes):
    try:
        return strict_json(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AppError(
            400, "INVALID_QUESTIONS_JSON", "Questions file must contain valid JSON"
        ) from exc


def question_file(data: bytes, settings):
    return normalize_questions(question_json(data), settings)


def question_request(data: bytes, settings):
    value = question_json(data)
    previous = value.pop("previous_questions", []) if isinstance(value, dict) else []
    if (
        not isinstance(previous, list)
        or len(previous) > 5
        or any(not isinstance(q, str) for q in previous)
    ):
        raise AppError(400, "INVALID_QUESTIONS_JSON", "Use at most five previous user questions")
    context = (
        [q.question for q in normalize_questions(previous, settings, maximum=5)] if previous else []
    )
    return normalize_questions(value, settings), context


async def read_upload(upload, maximum: int):
    chunks, total = [], 0
    while chunk := await upload.read(min(maximum + 1, 1024 * 1024)):
        total += len(chunk)
        if total > maximum:
            raise AppError(413, "FILE_TOO_LARGE", "File exceeds configured upload limit")
        chunks.append(chunk)
    return b"".join(chunks)
