from concurrent.futures import ThreadPoolExecutor

import pytest

from app.api.errors import AppError
from app.api.input import question_file
from app.config import Settings
from app.domain.models import DocumentNode
from app.ingestion.docling_parser import (
    convert_document,
    native_text_document,
    validate_pdf,
)
from app.ingestion.document_tree import ParsedBlock, build_tree
from app.ingestion.json_parser import parse_json
from app.ingestion.segmenter import Segmenter
from app.storage.repository import Repository
from tests.helpers import FakeModels, pdf_bytes


def test_pdf_structure_conversion():
    from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
    from docling_core.types.doc.base import BoundingBox
    from docling_core.types.doc.document import ProvenanceItem

    doc = DoclingDocument(name="policy")
    provenance = ProvenanceItem(
        page_no=37, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, 10)
    )
    doc.add_heading("Security", level=1, prov=provenance)
    doc.add_heading("Access Control", level=2, prov=provenance)
    doc.add_text(label=DocItemLabel.TEXT, text="Use MFA.", prov=provenance)
    doc.add_text(label=DocItemLabel.CAPTION, text="Table 1: recovery targets", prov=provenance)
    doc.add_table(
        data=TableData(
            num_rows=2,
            num_cols=2,
            table_cells=[
                TableCell(
                    text=text,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    column_header=r == 0,
                )
                for r, row in enumerate([["Metric", "Value"], ["RTO", "4 hours"]])
                for c, text in enumerate(row)
            ],
        ),
        prov=provenance,
    )
    nodes = convert_document(doc, "chat", "document")
    passage = next(n for n in nodes if n.text == "Use MFA.")
    assert passage.heading_path == ["Security", "Access Control"]
    assert passage.page_start == passage.page_end == 37
    assert nodes[0].page_end == 37
    table = next(n for n in nodes if n.node_type == "table")
    assert "|" in table.text and "RTO" in table.text and "4 hours" in table.text
    assert any(n.text == "Table 1: recovery targets" for n in nodes)


def test_json_structure_and_source_paths():
    nodes = parse_json(
        b'{"security":{"incident_response":{"notification_hours":24}}}',
        "chat",
        "document",
        Settings(_env_file=None),
    )
    leaf = nodes[-1]
    assert leaf.text == "notification_hours: 24"
    assert leaf.heading_path[-2:] == ["security", "incident_response"]
    assert leaf.metadata["json_pointer"] == "/security/incident_response/notification_hours"
    assert leaf.page_start is None and leaf.page_end is None
    assert leaf.parent_id == nodes[-2].id


def test_native_text_on_picture_pages_is_preserved(tmp_path):
    from docling_core.types.doc import DocItemLabel, DoclingDocument
    from docling_core.types.doc.base import BoundingBox
    from docling_core.types.doc.document import ProvenanceItem

    path = tmp_path / "policy.pdf"
    path.write_bytes(
        pdf_bytes(
            [
                "1. Cloud Hosting\nProduction runs on AWS.",
                "2. Unrelated Heading\nAn unrelated page.",
            ]
        )
    )
    doc = DoclingDocument(name="policy")
    provenance = ProvenanceItem(
        page_no=1, bbox=BoundingBox(l=0, t=0, r=100, b=100), charspan=(0, 0)
    )
    doc.add_picture(prov=provenance)
    doc.add_picture(prov=provenance.model_copy(update={"page_no": 2}))
    doc.add_text(label=DocItemLabel.TEXT, text="Production runs on AWS.", prov=provenance)
    nodes = convert_document(doc, "chat", "document", path)
    fallback = next(n for n in nodes if n.metadata.get("parser") == "pypdf")
    assert "1. Cloud Hosting" in fallback.text and "AWS" in fallback.text
    assert fallback.parent_id == nodes[0].id and fallback.heading_path == []
    assert fallback.page_start == fallback.page_end == 1
    leaves = Segmenter(FakeModels().tokenizer, Settings(_env_file=None)).segment(nodes)
    fallback_leaves = [n for n in leaves if n.searchable and n.metadata.get("parser") == "pypdf"]
    assert len(fallback_leaves) == 2
    assert {(n.page_start, n.page_end) for n in fallback_leaves} == {(1, 1), (2, 2)}
    assert fallback_leaves[0].parent_id == fallback.id
    assert any("Cloud Hosting" in n.text for n in fallback_leaves)


def test_large_pdf_native_text_keeps_pages_separate(tmp_path):
    path = tmp_path / "book.pdf"
    path.write_bytes(pdf_bytes(["First page. " * 20, "Second page. " * 20]))
    nodes = native_text_document(path, "chat", "document", 2)
    assert nodes is not None
    leaves = Segmenter(FakeModels().tokenizer, Settings(_env_file=None)).segment(nodes)
    leaves = [node for node in leaves if node.searchable]
    assert {(node.page_start, node.page_end) for node in leaves} == {(1, 1), (2, 2)}
    assert all(node.parent_id != nodes[0].id for node in leaves)
    assert all(node.metadata["parser"] == "pypdfium2" for node in leaves)


def test_large_pdf_fast_path_preserves_graphic_pages_and_stops_early(monkeypatch, tmp_path):
    visited = []

    class Image:
        pass

    class Page:
        def __init__(self, index):
            self.index = index

        def get_textpage(self):
            return self

        def get_text_range(self):
            return "" if self.index == 1 else "Native text. " * 20

        def get_objects(self):
            return [Image()] if self.index == 1 else []

        def close(self):
            pass

    class PDF:
        def __len__(self):
            return 100

        def __getitem__(self, index):
            visited.append(index)
            return Page(index)

        def close(self):
            pass

    monkeypatch.setattr("app.ingestion.docling_parser.PdfDocument", lambda _: PDF())
    monkeypatch.setattr("app.ingestion.docling_parser.PdfImage", Image)
    assert native_text_document(tmp_path / "unused.pdf", "chat", "document", 100) is None
    assert visited == [0, 1]


def test_pdf_table_column_padding_does_not_exhaust_token_budget():
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    doc = DoclingDocument(name="resume")
    value = " ".join(f"skill{i}" for i in range(50)) + " &#124; 001.2300"
    doc.add_table(
        data=TableData(
            num_rows=2,
            num_cols=2,
            table_cells=[
                TableCell(
                    text=text,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    column_header=r == 0,
                )
                for r, row in enumerate([["Category", "Skills"], ["AI", value]])
                for c, text in enumerate(row)
            ],
        ),
    )
    tree = convert_document(doc, "chat", "doc")
    table = next(node for node in tree if node.node_type == "table")
    assert value in table.text  # No source text or numerical precision is discarded.
    assert len(table.text.splitlines()[1]) < 25  # Separator width is independent of cell width.
    segmenter = Segmenter(FakeModels().tokenizer, Settings(_env_file=None))
    nodes = segmenter.segment(tree)
    leaves = [node for node in nodes if node.searchable]
    assert len(leaves) == 1
    assert leaves[0].metadata["parent_table_id"] == table.id
    assert all(segmenter.count(node.retrieval_text) <= segmenter.maximum for node in leaves)


@pytest.mark.parametrize("data", [b"{", b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"])
def test_invalid_json(data):
    with pytest.raises(AppError, match="valid JSON"):
        parse_json(data, "chat", "document", Settings(_env_file=None))


def test_json_depth_limit():
    with pytest.raises(AppError, match="limits"):
        parse_json(
            b'{"a":{"b":{"c":1}}}', "chat", "document", Settings(_env_file=None, max_json_depth=2)
        )


def test_heading_hierarchy_and_bounded_segmentation():
    settings = Settings(
        _env_file=None, retrieval_node_max_tokens=64, retrieval_node_target_tokens=32
    )
    segmenter = Segmenter(FakeModels().tokenizer, settings)
    text = " ".join(f"word{i}" for i in range(190))
    tree = build_tree(
        "chat",
        "doc",
        [
            ParsedBlock("section", "Security", 1, 1, 1),
            ParsedBlock("section", "Passwords", 2, 2, 2),
            ParsedBlock("passage", text, 2, 3),
            ParsedBlock("section", "Availability", 4, 4, 1),
            ParsedBlock("passage", "RTO is four hours.", 4, 4),
        ],
    )
    nodes = segmenter.segment(tree)
    leaves = [n for n in nodes if n.searchable]
    assert len(leaves) > 3
    assert all(segmenter.count(n.retrieval_text) <= 64 for n in leaves)
    password = [n for n in leaves if n.heading_path == ["Security", "Passwords"]]
    assert "".join(n.text for n in password) == text
    assert all(n.page_start == 2 and n.page_end == 3 for n in password)
    assert leaves[-1].heading_path == ["Availability"]
    assert all(not n.searchable for n in nodes[: len(tree)])


def test_split_table_repeats_header_and_retains_parent():
    settings = Settings(
        _env_file=None, retrieval_node_max_tokens=64, retrieval_node_target_tokens=32
    )
    segmenter = Segmenter(FakeModels().tokenizer, settings)
    header = "| Metric | Value |\n| --- | --- |"
    text = header + "\n" + "\n".join(f"| RTO-{i} | {i} hours |" for i in range(30))
    tree = build_tree("chat", "doc", [ParsedBlock("table", text, 5, 6)])
    nodes = segmenter.segment(tree)
    leaves = [n for n in nodes if n.searchable]
    assert len(leaves) > 1
    assert all(n.text.startswith(header) for n in leaves)
    assert all(n.parent_id == tree[1].id for n in leaves)
    assert all(n.metadata["parent_table_id"] == tree[1].id for n in leaves)
    assert all(n.page_start == 5 and n.page_end == 6 for n in leaves)
    assert sum(n.text.count("RTO-") for n in leaves) == 30


def test_table_with_genuinely_oversized_row_still_fails_without_truncation():
    segmenter = Segmenter(
        FakeModels().tokenizer,
        Settings(_env_file=None, retrieval_node_max_tokens=64, retrieval_node_target_tokens=32),
    )
    text = "| Skill | Description |\n| - | - |\n| AI | " + "word " * 100 + "|"
    tree = build_tree("chat", "doc", [ParsedBlock("table", text, 1, 1)])
    with pytest.raises(AppError) as error:
        segmenter.segment(tree)
    assert error.value.code == "TABLE_ROW_TOO_LARGE"
    assert tree[1].text == text


def test_small_adjacent_nodes_merge_only_under_same_parent():
    segmenter = Segmenter(FakeModels().tokenizer, Settings(_env_file=None))
    tree = build_tree(
        "chat",
        "doc",
        [
            ParsedBlock("section", "A"),
            ParsedBlock("passage", "First."),
            ParsedBlock("passage", "Second."),
            ParsedBlock("section", "B"),
            ParsedBlock("passage", "Third."),
        ],
    )
    leaves = [n for n in segmenter.segment(tree) if n.searchable]
    assert [n.text for n in leaves] == ["First.\n\nSecond.", "Third."]
    assert len(leaves[0].metadata["source_node_ids"]) == 2


def test_duplicate_reservation_is_atomic(tmp_path):
    repo = Repository(tmp_path / "db.sqlite")
    chat = repo.create_chat()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: repo.reserve_document(chat, "x.json", "application/json", "hash"),
                range(16),
            )
        )
    assert sum(not duplicate for _, duplicate in results) == 1
    assert len({doc.id for doc, _ in results}) == 1
    other = repo.create_chat()
    doc, duplicate = repo.reserve_document(other, "x.json", "application/json", "hash")
    assert not duplicate and doc.id != results[0][0].id
    repo.close()


def test_node_chat_provenance_rejected(tmp_path):
    repo = Repository(tmp_path / "db.sqlite")
    chat = repo.create_chat()
    doc, _ = repo.reserve_document(chat, "x.json", "application/json", "hash")
    with pytest.raises(ValueError, match="provenance"):
        repo.save_nodes(
            chat, doc.id, [DocumentNode(chat_id="other", document_id=doc.id, node_type="passage")]
        )
    repo.close()


def test_pdf_validation(tmp_path):
    pdf = pdf_bytes(["Hello"])
    path = tmp_path / "source.pdf"
    path.write_bytes(pdf)
    assert validate_pdf(pdf, 10) == validate_pdf(path, 10) == 1
    with pytest.raises(AppError):
        validate_pdf(b"not a PDF", 10)
    with pytest.raises(AppError) as error:
        validate_pdf(pdf_bytes(["a", "b"]), 1)
    assert error.value.status == 413


@pytest.mark.parametrize(
    "data",
    [b"[]", b'[""]', b'["ok",4]', b"null", b"{", b"[true]", b'{"questions":["ok"],"extra":1}'],
)
def test_question_file_rejects_invalid_inputs(data):
    with pytest.raises(AppError):
        question_file(data, Settings(_env_file=None))


def test_question_file_forms_and_limits():
    settings = Settings(_env_file=None, max_questions_per_request=1)
    assert [q.question for q in question_file(b'[" question? "]', settings)] == ["question?"]
    assert [q.question for q in question_file(b'{"questions":["question?"]}', settings)] == [
        "question?"
    ]
    with pytest.raises(AppError) as error:
        question_file(b'["one", "two"]', settings)
    assert error.value.status == 429
