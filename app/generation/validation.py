from app.domain.models import Citation, ModelGroundedAnswer


def validate_answer(result: ModelGroundedAnswer, evidence):
    allowed = {e.evidence_id: e for e in evidence}
    if not result.supported or result.confidence == "low":
        return False, None, []
    if not result.answer or not result.answer.strip() or not result.citations:
        return False, None, []
    if any(c.evidence_id not in allowed for c in result.citations):
        return False, None, []
    citations, seen = [], set()
    for citation in result.citations:
        if citation.evidence_id in seen:
            continue
        seen.add(citation.evidence_id)
        raw = allowed[citation.evidence_id]
        citations.append(
            Citation(
                evidence_id=raw.evidence_id,
                document_id=raw.document_id,
                filename=raw.filename,
                node_id=raw.node_id,
                page_start=raw.page_start,
                page_end=raw.page_end,
                heading_path=raw.heading_path,
            )
        )
    return True, result.answer.strip(), citations
