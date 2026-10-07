from app.api.errors import AppError
from app.domain.models import DocumentNode, new_id


class Segmenter:
    """Derive bounded retrieval leaves; retain original structural nodes unchanged."""

    def __init__(self, tokenizer, settings):
        self.tokenizer = tokenizer
        self.maximum = settings.retrieval_node_max_tokens
        self.target = settings.retrieval_node_target_tokens
        self.max_nodes = settings.max_document_nodes

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)

    def split_text(self, node):
        text = node.text
        if self.count(node.retrieval_text) <= self.maximum:
            return [text]
        prefix = "\n> ".join(node.heading_path)
        budget = self.maximum - self.count(prefix) - 8
        if budget < 8:
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "Heading context exceeds node token limit")
        parts = []
        while text:
            encoding = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoding.ids) <= budget:
                parts.append(text)
                break
            end = encoding.offsets[budget][0]
            # Prefer a paragraph/sentence boundary inside this oversized structural node.
            boundary = max(text.rfind("\n", 0, end), text.rfind(". ", 0, end))
            if boundary > end // 2:
                end = boundary + 1
            if end <= 0:
                raise AppError(422, "DOCUMENT_PARSE_FAILED", "Cannot segment source text safely")
            parts.append(text[:end])
            text = text[end:]
        return parts

    def split_table(self, node):
        rows = node.text.splitlines()
        if len(rows) < 2:
            return self.split_text(node)
        header = "\n".join(rows[:2])
        pieces, group = [], []
        for row in rows[2:]:
            trial = "\n".join([header, *group, row])
            if self.count(node.model_copy(update={"text": trial}).retrieval_text) > self.maximum:
                if not group:
                    raise AppError(
                        422, "TABLE_ROW_TOO_LARGE", "A table row exceeds node token limit"
                    )
                pieces.append("\n".join([header, *group]))
                group = []
                single = node.model_copy(update={"text": f"{header}\n{row}"})
                if self.count(single.retrieval_text) > self.maximum:
                    raise AppError(
                        422, "TABLE_ROW_TOO_LARGE", "A table row exceeds node token limit"
                    )
            group.append(row)
        if group or not pieces:
            pieces.append("\n".join([header, *group]))
        return pieces

    def segment(self, nodes: list[DocumentNode]):
        canonical = [n.model_copy(deep=True) for n in nodes]
        leaves = []
        for node in canonical:
            node.metadata["retrieval"] = False
            if node.node_type in ("document", "section") or not node.text.strip():
                continue
            texts = self.split_table(node) if node.node_type == "table" else self.split_text(node)
            for text in texts:
                leaf = node.model_copy(
                    deep=True,
                    update={
                        "id": new_id(),
                        "text": text,
                        "parent_id": node.id
                        if node.node_type == "table"
                        or node.metadata.get("parser") in {"pypdf", "pypdfium2"}
                        else node.parent_id,
                    },
                )
                leaf.metadata.update(retrieval=True, source_node_ids=[node.id])
                if node.node_type == "table":
                    leaf.metadata["parent_table_id"] = node.id
                if leaf.parent_id == node.id:
                    leaf.level += 1
                if self.count(leaf.retrieval_text) > self.maximum:
                    raise AppError(
                        422, "DOCUMENT_PARSE_FAILED", "Retrieval node exceeds token limit"
                    )
                # Merge only adjacent small leaves under exactly the same structural parent.
                previous = leaves[-1] if leaves else None
                can_merge = (
                    previous is not None
                    and previous.parent_id == leaf.parent_id
                    and previous.heading_path == leaf.heading_path
                    and previous.node_type == leaf.node_type
                    and leaf.node_type != "table"
                    and self.count(previous.retrieval_text) < self.target
                )
                merged = (
                    previous.model_copy(update={"text": previous.text + "\n\n" + text})
                    if can_merge
                    else None
                )
                if merged and self.count(merged.retrieval_text) <= self.maximum:
                    previous.text = merged.text
                    pages = [p for p in [previous.page_start, leaf.page_start] if p is not None]
                    ends = [p for p in [previous.page_end, leaf.page_end] if p is not None]
                    previous.page_start = min(pages) if pages else None
                    previous.page_end = max(ends) if ends else None
                    previous.metadata["source_node_ids"].extend(leaf.metadata["source_node_ids"])
                else:
                    leaves.append(leaf)
                if len(canonical) + len(leaves) > self.max_nodes:
                    raise AppError(422, "DOCUMENT_PARSE_FAILED", "Document exceeds node limit")
        for ordinal, leaf in enumerate(leaves):
            leaf.ordinal = ordinal
        return canonical + leaves
