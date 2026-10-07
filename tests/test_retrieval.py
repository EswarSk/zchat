import asyncio

import pytest

from app.api.errors import AppError
from app.domain.models import Candidate, DocumentNode
from app.retrieval.answerability import answerable
from app.retrieval.hybrid import rrf


def test_rrf_uses_ranks_and_shared_candidates():
    fused = rrf([[("a", 1000), ("b", -100)], [("b", 0.001), ("c", 999)]], 3)
    assert [i for i, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([[("a", 2), ("a", 1)]], 3) == [("a", 1 / 61)]


def test_answerability_requires_finite_reranker_evidence():
    node = DocumentNode(chat_id="chat", document_id="doc", node_type="passage", text="source")
    assert not answerable([], 0)
    assert not answerable([Candidate(node=node, retrieval_score=100)], 0)
    assert not answerable([Candidate(node=node, reranker_score=float("nan"))], 0)
    assert not answerable([Candidate(node=node, reranker_score=-1)], 0)
    assert answerable([Candidate(node=node, reranker_score=1)], 0)


async def test_dense_sparse_and_rerank(system):
    chat = system.repo.create_chat()
    await system.docs.ingest(
        chat, "source.json", "application/json", b'{"cloud":"AWS","region":"us-east-1"}'
    )
    encoded = system.models.encode_questions(["What cloud provider?"])[0]
    for mode in ["dense", "sparse", "hybrid"]:
        candidates, metrics = await system.qa.retriever.retrieve(chat, encoded, mode)
        assert candidates and all(c.node.chat_id == chat for c in candidates)
        assert "AWS" in candidates[0].node.text
        ranked = await system.qa.reranker.rank("What cloud provider?", candidates)
        assert ranked[0].reranker_score == 8
        assert metrics["fusion_ms"] >= 0


async def test_non_ready_partial_index_never_retrieved(system, monkeypatch):
    chat = system.repo.create_chat()
    await system.docs.ingest(chat, "ready.json", "application/json", b'{"code":"READY-123"}')
    original = system.store.upsert

    async def fail_after_upsert(*args):
        await original(*args)
        raise RuntimeError("simulated failure after partial write")

    async def fail_cleanup(*args):
        raise RuntimeError("cleanup unavailable")

    monkeypatch.setattr(system.store, "upsert", fail_after_upsert)
    monkeypatch.setattr(system.store, "delete_document", fail_cleanup)
    with pytest.raises(AppError):
        await system.docs.ingest(
            chat, "partial.json", "application/json", b'{"code":"SECRET-PARTIAL"}'
        )
    assert [d.status for d in system.repo.documents(chat)] == ["READY", "FAILED"]
    encoded = system.models.encode_questions(["What code?"])[0]
    candidates, _ = await system.qa.retriever.retrieve(chat, encoded)
    assert candidates and all("SECRET-PARTIAL" not in c.node.text for c in candidates)


async def test_expansion_neighbors_and_parent_span(system):
    from app.ingestion.document_tree import ParsedBlock, build_tree
    from app.ingestion.segmenter import Segmenter

    chat = system.repo.create_chat()
    document, _ = system.repo.reserve_document(chat, "source.pdf", "application/pdf", "hash")
    system.settings.retrieval_node_target_tokens = 16
    system.settings.retrieval_node_max_tokens = 32
    tree = build_tree(
        chat,
        document.id,
        [
            ParsedBlock("section", "A", 1, 1),
            *[ParsedBlock("passage", " ".join([f"block{i}"] * 20), i + 1, i + 1) for i in range(5)],
            ParsedBlock("section", "B", 10, 10),
            ParsedBlock("passage", "Unrelated section.", 10, 10),
        ],
    )
    nodes = Segmenter(system.models.tokenizer, system.settings).segment(tree)
    system.repo.save_nodes(chat, document.id, nodes)
    system.repo.set_status(chat, document.id, "READY")
    leaves = [n for n in nodes if n.searchable]
    heading = next(n for n in nodes if n.node_type == "section" and n.raw_text == "A")
    one = [Candidate(node=leaves[2], reranker_score=8)]
    evidence = system.qa.expander.expand(chat, one)
    assert {e.node_id for e in evidence} == {heading.id, *(n.id for n in leaves[1:4])}
    assert any(e.text == "A" and e.node_id == heading.id for e in evidence)
    two = [Candidate(node=leaves[1], reranker_score=8), Candidate(node=leaves[3], reranker_score=7)]
    evidence = system.qa.expander.expand(chat, two)
    assert {e.node_id for e in evidence} == {heading.id, *(n.id for n in leaves[1:4])}
    assert all(e.heading_path == ["A"] for e in evidence)
    system.qa.expander.budget = system.qa.expander.tokens(evidence[:1])
    bounded = system.qa.expander.expand(chat, two)
    assert len(bounded) == 1 and bounded[0].node_id == leaves[1].id
    assert system.qa.expander.tokens(bounded) <= system.qa.expander.budget
    other = system.repo.create_chat()
    with pytest.raises(ValueError, match="Cross-chat"):
        system.qa.expander.expand(other, one)


async def test_bounded_concurrency_is_shared_across_batches(system):
    chat = system.repo.create_chat()
    await system.docs.ingest(chat, "source.json", "application/json", b'{"cloud":"AWS"}')
    system.generator.delay = 0.02
    first, second = await asyncio.gather(
        system.qa.answer_many(chat, ["cloud one", "cloud two", "cloud three"]),
        system.qa.answer_many(chat, ["cloud four", "cloud five"]),
    )
    assert system.generator.maximum_active == system.settings.max_llm_concurrency
    assert [r.question for r in first + second] == [
        "cloud one",
        "cloud two",
        "cloud three",
        "cloud four",
        "cloud five",
    ]
    assert all(r.supported for r in first + second)


async def test_large_tree_has_document_length_independent_context_budget(system):
    from app.ingestion.document_tree import ParsedBlock, build_tree
    from app.ingestion.segmenter import Segmenter

    chat = system.repo.create_chat()
    document, _ = system.repo.reserve_document(chat, "large.pdf", "application/pdf", "large")
    blocks = []
    for page in range(1, 1001):
        blocks.extend(
            [
                ParsedBlock("section", f"Section {page}", page, page),
                ParsedBlock("passage", " ".join([f"value-{page}"] * 100), page, page),
            ]
        )
    nodes = Segmenter(system.models.tokenizer, system.settings).segment(
        build_tree(chat, document.id, blocks),
    )
    system.repo.save_nodes(chat, document.id, nodes)
    system.repo.set_status(chat, document.id, "READY", page_count=1000)
    leaves = [n for n in nodes if n.searchable]
    assert len(leaves) == 1000
    ranked = [Candidate(node=n, reranker_score=8) for n in leaves[-6:]]
    evidence = system.qa.expander.expand(chat, ranked)
    assert system.qa.expander.tokens(evidence) <= system.settings.max_context_tokens
    assert any(e.page_start == 1000 for e in evidence)
    assert len(evidence) <= 12
    assert any(e.text == "Section 1000" for e in evidence)
