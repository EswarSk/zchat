import json
from decimal import Decimal

from app.api.errors import AppError
from app.domain.models import DocumentNode


def strict_json(data: bytes | str):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Non-finite JSON number")

    return json.loads(
        data,
        object_pairs_hook=object_pairs,
        parse_constant=invalid_constant,
        parse_float=Decimal,
    )


def parse_json(data: bytes, chat_id: str, document_id: str, settings) -> list[DocumentNode]:
    try:
        source = strict_json(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AppError(422, "DOCUMENT_PARSE_FAILED", "Source is not valid JSON") from exc
    root = DocumentNode(chat_id=chat_id, document_id=document_id, node_type="document")
    nodes = [root]

    def visit(value, parent, key: str | None, pointer: str, depth: int):
        if depth > settings.max_json_depth or len(nodes) >= settings.max_document_nodes:
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "JSON structure exceeds configured limits")
        container = isinstance(value, (dict, list)) and bool(value)
        label = key if key is not None else "value"
        literal = (
            str(value)
            if isinstance(value, Decimal)
            else (json.dumps(value, ensure_ascii=False, allow_nan=False) if not container else "")
        )
        text = label if container else f"{label}: {literal}"
        node = DocumentNode(
            chat_id=chat_id,
            document_id=document_id,
            parent_id=parent.id,
            node_type="section" if container else "passage",
            level=depth,
            heading=label if container else None,
            heading_path=[*parent.heading_path, label] if container else list(parent.heading_path),
            text=text,
            ordinal=len(nodes),
            metadata={"json_pointer": pointer},
        )
        nodes.append(node)
        if container:
            pairs = value.items() if isinstance(value, dict) else enumerate(value)
            for child_key, child_value in pairs:
                escaped = str(child_key).replace("~", "~0").replace("/", "~1")
                visit(child_value, node, str(child_key), f"{pointer}/{escaped}", depth + 1)

    visit(source, root, None, "", 1)
    return nodes
