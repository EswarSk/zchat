import asyncio
import hashlib
import io
import re
from types import SimpleNamespace

import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from app.api.errors import AppError
from app.domain.models import ModelCitation, ModelGroundedAnswer
from app.ingestion.docling_parser import validate_pdf
from app.ingestion.document_tree import ParsedBlock, build_tree


def pdf_bytes(pages: list[str]) -> bytes:
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): writer._add_object(font)}
                ),
            }
        )
        stream = DecodedStreamObject()
        lines = ["BT /F1 12 Tf 50 740 Td 18 TL"]
        for line in text.splitlines():
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            lines.append(f"({escaped}) Tj T*")
        lines.append("ET")
        stream.set_data("\n".join(lines).encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def words(text):
    aliases = {
        "aws": "cloud",
        "amazon": "cloud",
        "infrastructure": "cloud",
        "hosts": "cloud",
        "hosted": "cloud",
        "workloads": "cloud",
        "provider": "cloud",
        "web": "cloud",
        "services": "cloud",
    }
    stop = {"what", "which", "is", "the", "a", "an", "of", "on", "are", "used", "who"}
    return {aliases.get(w, w) for w in re.findall(r"[\w-]+", text.lower()) if w not in stop}


class FakeModels:
    dimension = 64

    def __init__(self):
        self.tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.document_calls = 0
        self.question_calls = 0

    def _encode(self, texts):
        dense, sparse = [], []
        for text in texts:
            ids = sorted(
                {int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % 64 for w in words(text)}
            )
            vector = np.zeros(64)
            vector[ids] = 1
            norm = np.linalg.norm(vector)
            dense.append((vector / norm if norm else vector).tolist())
            sparse.append(SimpleNamespace(indices=np.array(ids), values=np.ones(len(ids))))
        return dense, sparse

    def encode_documents(self, texts):
        self.document_calls += 1
        return self._encode(texts)

    def encode_questions(self, texts):
        self.question_calls += 1
        return list(zip(*self._encode(texts), strict=True))

    def rerank(self, question, texts):
        return [8.0 if words(question) & words(text) else -12.0 for text in texts]


class FakeParser:
    def __init__(self, settings):
        self.settings = settings
        self.calls = 0

    async def parse_async(self, path, chat_id, document_id):
        return await asyncio.to_thread(self.parse, path, chat_id, document_id)

    def parse(self, path, chat_id, document_id):
        self.calls += 1
        data = path.read_bytes()
        count = validate_pdf(data, self.settings.max_pdf_pages)
        reader = PdfReader(io.BytesIO(data))
        blocks = [
            ParsedBlock("passage", p.extract_text(), i, i) for i, p in enumerate(reader.pages, 1)
        ]
        return build_tree(chat_id, document_id, blocks), count


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.delay = 0.0
        self.active = 0
        self.maximum_active = 0
        self.fabricate = False
        self.timeout_question = None

    async def contextualize(self, question, previous_questions):
        return question

    async def answer(self, question, evidence, *, timeout_seconds=None):
        self.calls.append((question, evidence))
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if question == self.timeout_question:
                raise AppError(504, "LLM_TIMEOUT", "Answer provider timed out")
            # CI tests validate the wiring, never substitute this for the real generator.
            item = next((e for e in evidence if words(question) & words(e.text)), None)
            if item is None:
                return ModelGroundedAnswer(supported=False, answer=None, citations=[])
            return ModelGroundedAnswer(
                supported=True,
                answer=item.text,
                citations=[
                    ModelCitation(evidence_id="E999" if self.fabricate else item.evidence_id),
                ],
            )
        finally:
            self.active -= 1

    async def close(self):
        pass
