from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictStr


def new_id() -> str:
    return str(uuid4())


class QuestionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr = Field(default_factory=new_id, min_length=1, max_length=255)
    question: StrictStr


class Document(BaseModel):
    id: str
    chat_id: str
    filename: str
    content_type: str
    sha256: str
    status: Literal["PENDING", "PARSING", "INDEXING", "READY", "FAILED"] = "PENDING"
    page_count: int | None = None
    created_at: str
    ingested_at: str | None = None
    error_message: str | None = None


class DocumentNode(BaseModel):
    id: str = Field(default_factory=new_id)
    chat_id: str
    document_id: str
    parent_id: str | None = None
    node_type: Literal["document", "section", "passage", "table", "list"]
    level: int = 0
    heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    text: str = ""
    # JSON has no physical pages: both are null, rather than inventing a page 1.
    page_start: int | None = None
    page_end: int | None = None
    ordinal: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        return self.text

    @property
    def retrieval_text(self) -> str:
        prefix = "\n> ".join(self.heading_path)
        return f"{prefix}\n\n{self.text}" if prefix else self.text

    @property
    def searchable(self) -> bool:
        return bool(self.metadata.get("retrieval", False) and self.text.strip())


class Candidate(BaseModel):
    node: DocumentNode
    retrieval_score: float | None = None
    reranker_score: float | None = None


class Evidence(BaseModel):
    evidence_id: str
    document_id: str
    filename: str
    node_id: str
    parent_id: str | None
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    text: str
    retrieval_score: float | None = None
    reranker_score: float | None = None

    def prompt_record(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "metadata": {
                "filename": self.filename,
                "page_start": self.page_start,
                "page_end": self.page_end,
                "heading_path": self.heading_path,
            },
            "raw_source_text": self.text,
        }


class ModelCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str


class ModelGroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supported: bool
    answer: str | None
    citations: list[ModelCitation]
    comments: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"


class Citation(BaseModel):
    evidence_id: str
    document_id: str
    filename: str
    node_id: str
    page_start: int | None
    page_end: int | None
    heading_path: list[str]


class QuestionError(BaseModel):
    code: str
    message: str
    http_status: int


class AnswerResult(BaseModel):
    id: str = Field(default_factory=new_id)
    question: str
    resolved_question: str | None = None
    supported: bool = False
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)
    error: QuestionError | None = None
    comments: str | None = None
    confidence: Literal["high", "medium", "low"] = "low"

    def questionnaire_record(self) -> dict:
        if self.error:
            answer = "Error"
            comments = f"{self.error.code}: {self.error.message}"
        elif not self.supported:
            answer = "Data-Not-Found"
            comments = "No supporting evidence was found in the uploaded documents."
        else:
            answer = self.answer
            sources = []
            for citation in self.citations:
                location = " > ".join(citation.heading_path) or "JSON source"
                if citation.page_start is not None:
                    location = f"page {citation.page_start}"
                    if citation.page_end != citation.page_start:
                        location = f"pages {citation.page_start}-{citation.page_end}"
                source = f"{citation.filename} ({location})"
                if source not in sources:
                    sources.append(source)
            comments = (self.comments or "Supported by the uploaded documents.").strip()
            comments += "\nSources: " + "; ".join(sources)
        return {
            "id": self.id,
            "question": self.question,
            "answer": answer,
            "comments": comments,
            "confidence": self.confidence if self.supported and not self.error else "low",
        }
