"""Real local retrieval evaluation; --generate explicitly enables paid OpenAI calls."""

# ruff: noqa: E402
import argparse
import asyncio
import json
import math
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.domain.models import Candidate
from app.generation.validation import validate_answer
from app.main import build_services
from app.retrieval.answerability import answerable


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def classification_metrics(labels, predictions):
    true_negative = sum(
        not label and not pred for label, pred in zip(labels, predictions, strict=True)
    )
    predicted_not_found = sum(not p for p in predictions)
    actual_not_found = sum(not label for label in labels)
    positives = sum(labels)
    return {
        "answerable_accuracy": sum(
            label == pred for label, pred in zip(labels, predictions, strict=True)
        )
        / len(labels),
        "answerable_recall": sum(
            label and pred for label, pred in zip(labels, predictions, strict=True)
        )
        / positives
        if positives
        else None,
        "NOT_FOUND_precision": true_negative / predicted_not_found if predicted_not_found else None,
        "NOT_FOUND_recall": true_negative / actual_not_found if actual_not_found else None,
    }


def calibrate(labels, scores):
    """Choose the most conservative threshold among equal fixture classification scores."""
    finite = sorted({score for score in scores if math.isfinite(score)})
    thresholds = [a + (b - a) / 2 for a, b in zip(finite, finite[1:], strict=False)]
    thresholds += [finite[0] - 1, finite[-1] + 1] if finite else [0.0]
    rows = []
    for threshold in thresholds:
        metrics = classification_metrics(labels, [s >= threshold for s in scores])
        rows.append({"threshold": threshold, **metrics})
    return max(rows, key=lambda row: (row["answerable_accuracy"], row["threshold"]))


def relevant(node, gold, filenames):
    if filenames[node.document_id] != gold.get("document"):
        return False
    pages = gold["relevant_pages"]
    if not pages:
        return node.page_start is None
    return node.page_start is not None and any(node.page_start <= p <= node.page_end for p in pages)


async def evaluate(args):
    fixture_path = Path(args.fixture).resolve()
    fixture = json.loads(fixture_path.read_text())
    with tempfile.TemporaryDirectory(prefix="zchat-evaluation-") as directory:
        settings = Settings(
            database_path=Path(directory) / "metadata.sqlite3",
            qdrant_collection=f"zchat_eval_{uuid4().hex}",
        )
        if args.generate and not settings.openai_api_key:
            raise SystemExit(
                "--generate requires OPENAI_API_KEY; default retrieval evaluation is free"
            )
        bundle = await build_services(settings)
        try:
            chat = bundle.repository.create_chat()
            for filename in fixture["documents"]:
                path = fixture_path.parent / filename
                kind = "application/pdf" if path.suffix == ".pdf" else "application/json"
                await bundle.documents.ingest(chat, filename, kind, path.read_bytes())
            filenames = {d.id: d.filename for d in bundle.repository.documents(chat)}
            questions = [row["question"] for row in fixture["questions"]]
            tick = perf_counter()
            vectors = await asyncio.to_thread(bundle.qa.models.encode_questions, questions)
            encoding_per_question_ms = (perf_counter() - tick) * 1000 / len(questions)
            labels = [row["answerable"] for row in fixture["questions"]]
            modes = {}
            calibration_scores = []
            for mode in ["dense", "sparse", "hybrid", "hybrid_reranker"]:
                rows = []
                for gold, vector in zip(fixture["questions"], vectors, strict=True):
                    start = perf_counter()
                    candidates, _ = await bundle.qa.retriever.retrieve(
                        chat,
                        vector,
                        "hybrid" if mode == "hybrid_reranker" else mode,
                    )
                    gate = bool(candidates)
                    if mode == "hybrid_reranker":
                        # Rank the full shortlist for Recall@10, then select generation top K.
                        scores = await asyncio.to_thread(
                            bundle.qa.models.rerank,
                            gold["question"],
                            [c.node.retrieval_text for c in candidates],
                        )
                        candidates = sorted(
                            [
                                c.model_copy(update={"reranker_score": float(score)})
                                for c, score in zip(candidates, scores, strict=True)
                                if math.isfinite(score)
                            ],
                            key=lambda c: -c.reranker_score,
                        )
                        calibration_scores.append(
                            candidates[0].reranker_score if candidates else -math.inf
                        )
                        gate = answerable(candidates, settings.answerability_threshold)
                    retrieval_ms = (perf_counter() - start) * 1000 + encoding_per_question_ms
                    hits = [relevant(c.node, gold, filenames) for c in candidates]
                    row = {
                        "question": gold["question"],
                        "answerable": gold["answerable"],
                        "recall_at_5": int(any(hits[:5])) if gold["answerable"] else None,
                        "recall_at_10": int(any(hits[:10])) if gold["answerable"] else None,
                        "reciprocal_rank": 1 / (hits.index(True) + 1) if any(hits) else 0.0,
                        "gate_supported": gate,
                        "retrieval_ms": retrieval_ms,
                        "supported": None,
                        "citation_page_hits": 0,
                        "citation_page_count": 0,
                        "error_code": None,
                    }
                    if args.generate:
                        selected = candidates[: settings.rerank_top_k]
                        if mode == "hybrid_reranker":
                            selected = [
                                c
                                for c in selected
                                if c.reranker_score >= settings.answerability_threshold
                            ]
                        evidence = (
                            await asyncio.to_thread(bundle.qa.expander.expand, chat, selected)
                            if gate
                            else []
                        )
                        if evidence:
                            try:
                                generated = await bundle.qa.generator.answer(
                                    gold["question"], evidence
                                )
                                supported, answer, citations = validate_answer(generated, evidence)
                                row["supported"] = supported
                                row["answer_text_match"] = bool(
                                    answer
                                    and any(
                                        expected.lower() in answer.lower()
                                        for expected in gold.get("answer_contains", [])
                                    )
                                )
                                for citation in citations:
                                    if citation.page_start is not None:
                                        row["citation_page_count"] += 1
                                        proxy = Candidate(node=selected[0].node).node.model_copy(
                                            update={
                                                "document_id": citation.document_id,
                                                "page_start": citation.page_start,
                                                "page_end": citation.page_end,
                                            }
                                        )
                                        row["citation_page_hits"] += int(
                                            relevant(proxy, gold, filenames)
                                        )
                            except Exception as exc:
                                row["error_code"] = getattr(exc, "code", "GENERATION_ERROR")
                        else:
                            row["supported"] = False
                    row["total_ms"] = (perf_counter() - start) * 1000 + encoding_per_question_ms
                    rows.append(row)
                positives = [row for row in rows if row["answerable"]]
                page_count = sum(row["citation_page_count"] for row in rows)
                summary = {
                    "Recall@5": sum(row["recall_at_5"] for row in positives) / len(positives),
                    "Recall@10": sum(row["recall_at_10"] for row in positives) / len(positives),
                    "MRR": sum(row["reciprocal_rank"] for row in positives) / len(positives),
                    "gate_metrics": classification_metrics(
                        labels, [row["gate_supported"] for row in rows]
                    ),
                    "answerable_accuracy": None,
                    "NOT_FOUND_precision": None,
                    "NOT_FOUND_recall": None,
                    "citation_page_accuracy": sum(row["citation_page_hits"] for row in rows)
                    / page_count
                    if page_count
                    else None,
                    "retrieval_p50_ms": percentile([row["retrieval_ms"] for row in rows], 0.5),
                    "retrieval_p95_ms": percentile([row["retrieval_ms"] for row in rows], 0.95),
                    "total_p50_ms": percentile([row["total_ms"] for row in rows], 0.5),
                    "total_p95_ms": percentile([row["total_ms"] for row in rows], 0.95),
                    "generation_errors": sum(row["error_code"] is not None for row in rows),
                }
                if args.generate:
                    successful = [
                        (label, row)
                        for label, row in zip(labels, rows, strict=True)
                        if row["supported"] is not None
                    ]
                    if successful:
                        summary.update(
                            classification_metrics(
                                [label for label, _ in successful],
                                [row["supported"] for _, row in successful],
                            )
                        )
                modes[mode] = {"summary": summary, "questions": rows}
            report = {
                "timestamp": datetime.now(UTC).isoformat(),
                "fixture": str(fixture_path),
                "generation_enabled": args.generate,
                "answerability_threshold": settings.answerability_threshold,
                "models": {
                    "dense": settings.dense_model,
                    "sparse": settings.sparse_model,
                    "reranker": settings.reranker_model,
                },
                "calibration": calibrate(labels, calibration_scores),
                "calibration_note": (
                    "Fixture training estimate only; validate on independent documents "
                    "before production."
                ),
                "modes": modes,
            }
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "modes": {mode: result["summary"] for mode, result in modes.items()},
                        "calibration": report["calibration"],
                    },
                    indent=2,
                )
            )
        finally:
            await bundle.store.client.delete_collection(settings.qdrant_collection)
            await bundle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="evaluation/questions.json")
    parser.add_argument("--output", default="evaluation-report.json")
    parser.add_argument(
        "--generate", action="store_true", help="Enable paid OpenAI answer/citation evaluation"
    )
    asyncio.run(evaluate(parser.parse_args()))
