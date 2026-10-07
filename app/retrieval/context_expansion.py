import json
from collections import defaultdict

import tiktoken

from app.domain.models import Candidate, Evidence


def evidence_json(evidence: list[Evidence]) -> str:
    return json.dumps([e.prompt_record() for e in evidence], ensure_ascii=False)


class ContextExpander:
    def __init__(self, repository, settings):
        self.repository = repository
        self.budget = settings.max_context_tokens
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def tokens(self, evidence):
        return len(self.tokenizer.encode(evidence_json(evidence), disallowed_special=()))

    def expand(self, chat_id: str, ranked: list[Candidate]) -> list[Evidence]:
        filenames = {
            d.id: d.filename for d in self.repository.documents(chat_id) if d.status == "READY"
        }
        groups = defaultdict(list)
        for candidate in ranked:
            if candidate.node.chat_id != chat_id:
                raise ValueError("Cross-chat candidate rejected")
            groups[(candidate.node.document_id, candidate.node.parent_id)].append(candidate)
        additions = []
        for siblings in groups.values():
            first = siblings[0].node
            additions.extend(self.repository.ancestors(first))
            if len(siblings) > 1:
                ordinals = [c.node.ordinal for c in siblings]
                additions.extend(self.repository.sibling_span(first, min(ordinals), max(ordinals)))
            else:
                additions.extend(self.repository.neighbors(first))
        candidates = [*ranked, *(Candidate(node=n) for n in additions)]
        evidence, seen = [], set()
        # Retrieved leaves get first use of budget; extras never evict higher-ranked evidence.
        for candidate in candidates:
            node = candidate.node
            if node.id in seen or node.document_id not in filenames:
                continue
            seen.add(node.id)
            item = Evidence(
                evidence_id=f"E{len(evidence) + 1}",
                document_id=node.document_id,
                filename=filenames[node.document_id],
                node_id=node.id,
                parent_id=node.parent_id,
                page_start=node.page_start,
                page_end=node.page_end,
                heading_path=node.heading_path,
                text=node.raw_text,
                retrieval_score=candidate.retrieval_score,
                reranker_score=candidate.reranker_score,
            )
            if self.tokens([*evidence, item]) <= self.budget:
                evidence.append(item)
        return evidence
