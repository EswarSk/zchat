from dataclasses import dataclass, field
from typing import Any

from app.domain.models import DocumentNode


@dataclass
class ParsedBlock:
    kind: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    heading_level: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


def build_tree(chat_id: str, document_id: str, blocks: list[ParsedBlock]) -> list[DocumentNode]:
    root = DocumentNode(chat_id=chat_id, document_id=document_id, node_type="document")
    nodes = [root]
    stack = [root]
    for block in blocks:
        if not block.text.strip():
            continue
        if block.kind == "section":
            level = max(1, block.heading_level)
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            node = DocumentNode(
                chat_id=chat_id,
                document_id=document_id,
                parent_id=stack[-1].id,
                node_type="section",
                level=level,
                heading=block.text,
                heading_path=[*stack[-1].heading_path, block.text],
                text=block.text,
                page_start=block.page_start,
                page_end=block.page_end,
                ordinal=len(nodes),
                metadata=block.metadata,
            )
            stack.append(node)
        else:
            node = DocumentNode(
                chat_id=chat_id,
                document_id=document_id,
                parent_id=stack[-1].id,
                node_type=block.kind,
                level=stack[-1].level + 1,
                heading_path=list(stack[-1].heading_path),
                text=block.text,
                page_start=block.page_start,
                page_end=block.page_end,
                ordinal=len(nodes),
                metadata=block.metadata,
            )
        nodes.append(node)
    # Propagate descendant page ranges without fabricating section summaries.
    by_id = {n.id: n for n in nodes}
    for node in reversed(nodes):
        if node.parent_id and node.page_start is not None:
            parent = by_id[node.parent_id]
            parent.page_start = min(parent.page_start or node.page_start, node.page_start)
            parent.page_end = max(parent.page_end or node.page_end, node.page_end)
    return nodes
