import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.api.errors import AppError
from app.domain.models import Document, DocumentNode, new_id


def now() -> str:
    return datetime.now(UTC).isoformat()


class Repository:
    """Owns metadata transactions; services never issue SQL directly."""

    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY, user_id TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, chat_id TEXT NOT NULL REFERENCES chat_sessions(id),
                filename TEXT NOT NULL, content_type TEXT NOT NULL, sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN
                    ('PENDING','PARSING','INDEXING','READY','FAILED')),
                page_count INTEGER, created_at TEXT NOT NULL, ingested_at TEXT,
                error_message TEXT, UNIQUE(chat_id, sha256), UNIQUE(id, chat_id)
            );
            CREATE TABLE IF NOT EXISTS document_nodes (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chat_id TEXT NOT NULL,
                parent_id TEXT, ordinal INTEGER NOT NULL, searchable INTEGER NOT NULL,
                data_json TEXT NOT NULL,
                FOREIGN KEY(document_id, chat_id) REFERENCES documents(id, chat_id)
            );
            CREATE INDEX IF NOT EXISTS nodes_chat ON document_nodes(chat_id, document_id);
            CREATE INDEX IF NOT EXISTS nodes_parent
                ON document_nodes(chat_id, document_id, parent_id, ordinal);
        """)

    @contextmanager
    def transaction(self):
        with self.lock, self.db:
            yield self.db

    def close(self):
        with self.lock:
            self.db.close()

    def create_chat(self) -> str:
        chat_id = new_id()
        with self.transaction() as db:
            db.execute("INSERT INTO chat_sessions VALUES (?,NULL,?)", (chat_id, now()))
        return chat_id

    def require_chat(self, chat_id: str):
        with self.lock:
            row = self.db.execute("SELECT id FROM chat_sessions WHERE id=?", (chat_id,)).fetchone()
        if not row:
            raise AppError(404, "CHAT_NOT_FOUND", "Chat does not exist")

    def reserve_document(self, chat_id, filename, content_type, sha256) -> tuple[Document, bool]:
        self.require_chat(chat_id)
        with self.transaction() as db:
            doc_id = new_id()
            result = db.execute(
                """INSERT INTO documents
                (id,chat_id,filename,content_type,sha256,status,created_at)
                VALUES (?,?,?,?,?,'PENDING',?) ON CONFLICT(chat_id,sha256) DO NOTHING""",
                (doc_id, chat_id, filename, content_type, sha256, now()),
            )
            row = db.execute(
                "SELECT * FROM documents WHERE chat_id=? AND sha256=?", (chat_id, sha256)
            ).fetchone()
            duplicate = result.rowcount == 0
            if duplicate and row["status"] == "FAILED":
                # Claim a retry atomically; concurrent uploads still share one ingestion.
                claimed = db.execute(
                    """UPDATE documents SET status='PENDING',error_message=NULL,
                    page_count=NULL,ingested_at=NULL WHERE id=? AND status='FAILED'""",
                    (row["id"],),
                )
                if claimed.rowcount:
                    db.execute("DELETE FROM document_nodes WHERE document_id=?", (row["id"],))
                    duplicate = False
                row = db.execute("SELECT * FROM documents WHERE id=?", (row["id"],)).fetchone()
            return Document(**dict(row)), duplicate

    def set_status(self, chat_id, document_id, status, *, page_count=None, error=None):
        with self.transaction() as db:
            result = db.execute(
                """UPDATE documents SET status=?,page_count=COALESCE(?,page_count),
                error_message=?,ingested_at=? WHERE chat_id=? AND id=?""",
                (
                    status,
                    page_count,
                    error,
                    now() if status == "READY" else None,
                    chat_id,
                    document_id,
                ),
            )
            if not result.rowcount:
                raise AppError(404, "DOCUMENT_NOT_FOUND", "Document does not exist in this chat")

    def documents(self, chat_id: str) -> list[Document]:
        self.require_chat(chat_id)
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM documents WHERE chat_id=? ORDER BY created_at,id", (chat_id,)
            ).fetchall()
        return [Document(**dict(row)) for row in rows]

    def ready_ids(self, chat_id: str) -> list[str]:
        self.require_chat(chat_id)
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM documents WHERE chat_id=? AND status='READY'", (chat_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def save_nodes(self, chat_id: str, document_id: str, nodes: list[DocumentNode]):
        if any(n.chat_id != chat_id or n.document_id != document_id for n in nodes):
            raise ValueError("Node provenance does not match document")
        ids = {n.id for n in nodes}
        if len(ids) != len(nodes) or any(n.parent_id and n.parent_id not in ids for n in nodes):
            raise ValueError("Invalid document tree")
        with self.transaction() as db:
            db.executemany(
                "INSERT INTO document_nodes VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        n.id,
                        n.document_id,
                        n.chat_id,
                        n.parent_id,
                        n.ordinal,
                        int(n.searchable),
                        n.model_dump_json(),
                    )
                    for n in nodes
                ],
            )

    def nodes_by_ids(self, chat_id: str, ids: list[str]) -> list[DocumentNode]:
        if not ids:
            return []
        with self.lock:
            rows = self.db.execute(
                f"""SELECT n.data_json FROM document_nodes n JOIN documents d
                ON n.document_id=d.id AND n.chat_id=d.chat_id
                WHERE n.chat_id=? AND d.status='READY' AND n.searchable=1
                AND n.id IN ({",".join("?" for _ in ids)})""",
                [chat_id, *ids],
            ).fetchall()
        return [DocumentNode.model_validate_json(r[0]) for r in rows]

    def sibling_span(self, node: DocumentNode, start: int, end: int, limit=50):
        with self.lock:
            rows = self.db.execute(
                """SELECT n.data_json FROM document_nodes n JOIN documents d
                ON n.document_id=d.id AND n.chat_id=d.chat_id
                WHERE n.chat_id=? AND n.document_id=? AND n.parent_id IS ?
                AND d.status='READY' AND n.searchable=1 AND n.ordinal BETWEEN ? AND ?
                ORDER BY n.ordinal LIMIT ?""",
                (node.chat_id, node.document_id, node.parent_id, start, end, limit),
            ).fetchall()
        return [DocumentNode.model_validate_json(r[0]) for r in rows]

    def neighbors(self, node: DocumentNode):
        result = []
        for comparison, order in [("<", "DESC"), (">", "ASC")]:
            with self.lock:
                row = self.db.execute(
                    f"""SELECT n.data_json FROM document_nodes n JOIN documents d
                    ON n.document_id=d.id AND n.chat_id=d.chat_id
                    WHERE n.chat_id=? AND n.document_id=? AND n.parent_id IS ?
                    AND d.status='READY' AND n.searchable=1 AND n.ordinal {comparison} ?
                    ORDER BY n.ordinal {order} LIMIT 1""",
                    (node.chat_id, node.document_id, node.parent_id, node.ordinal),
                ).fetchone()
            if row:
                result.append(DocumentNode.model_validate_json(row[0]))
        return result

    def ancestors(self, node: DocumentNode):
        # Canonical headings are source text; a retrieval leaf's metadata is not proof.
        with self.lock:
            rows = self.db.execute(
                """WITH RECURSIVE ancestors(id, parent_id, ordinal, data_json) AS (
                    SELECT n.id, n.parent_id, n.ordinal, n.data_json
                    FROM document_nodes n JOIN documents d
                    ON n.document_id=d.id AND n.chat_id=d.chat_id
                    WHERE n.id=? AND n.chat_id=? AND n.document_id=? AND d.status='READY'
                    UNION
                    SELECT n.id, n.parent_id, n.ordinal, n.data_json
                    FROM document_nodes n JOIN ancestors a ON n.id=a.parent_id
                    WHERE n.chat_id=? AND n.document_id=?
                ) SELECT data_json FROM ancestors ORDER BY ordinal""",
                (node.parent_id, node.chat_id, node.document_id, node.chat_id, node.document_id),
            ).fetchall()
        return [
            n
            for row in rows
            if (n := DocumentNode.model_validate_json(row[0])).node_type == "section"
            and n.raw_text.strip()
        ]

    def recover_interrupted_ingestion(self):
        # Run once before accepting traffic; this deployment uses one API process.
        with self.transaction() as db:
            db.execute("""UPDATE documents SET status='FAILED',
                error_message='INGESTION_INTERRUPTED'
                WHERE status IN ('PENDING','PARSING','INDEXING')""")

    def ping(self):
        with self.lock:
            self.db.execute("SELECT 1").fetchone()
