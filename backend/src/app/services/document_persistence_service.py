import os
import aiosqlite
from datetime import datetime
from typing import Optional
import logging

from app.models.document import Document
from app.core.config import get_settings

settings = get_settings()
DB_PATH = settings.documents_db_uri
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    author TEXT,
    tag TEXT,
    page_count INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""

logger = logging.getLogger(__name__)

class DocumentPersistenceService:
    """Async SQLite persistence for document metadata and status."""

    @staticmethod
    async def init_db():
        async with aiosqlite.connect(DB_PATH) as db:
            await db.executescript(CREATE_TABLE_SQL)
            await db.commit()
        logger.info(f"Initialized SQLite database at {DB_PATH}")

    @staticmethod
    async def insert_document(doc: Document):
        now = datetime.utcnow().isoformat()
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    """
                    INSERT INTO documents (id, name, author, tag, page_count, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (doc.id, doc.name, doc.author, doc.tag, doc.page_count, doc.status, now, now)
                )
                await db.commit()
            logger.info(f"Successfully inserted document {doc.id}")
        except aiosqlite.Error as e:
            logger.error(f"Error inserting document {doc.id}: {e}", exc_info=True)
            raise # Re-raise the exception so the caller knows the operation failed

    @staticmethod
    async def update_status(document_id: str, status: str):
        now = datetime.utcnow().isoformat()
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, document_id)
                )
                await db.commit()
            logger.info(f"Successfully updated status for document {document_id} to {status}")
        except aiosqlite.Error as e:
            logger.error(f"Error updating status for document {document_id} to {status}: {e}", exc_info=True)
            raise # Re-raise the exception

    @staticmethod
    async def get_document(document_id: str) -> Optional[Document]:
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    "SELECT id, name, author, tag, page_count, status FROM documents WHERE id = ?",
                    (document_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return Document(
                            id=row[0], name=row[1], author=row[2], tag=row[3],
                            page_count=row[4], status=row[5]
                        )
            return None
        except aiosqlite.Error as e:
            logger.error(f"Error getting document {document_id}: {e}", exc_info=True)
            return None # Return None on error, indicating document not accessible/found

    @staticmethod
    async def document_exists(document_id: str) -> bool:
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                async with db.execute(
                    "SELECT 1 FROM documents WHERE id = ?",
                    (document_id,)
                ) as cursor:
                    return await cursor.fetchone() is not None
        except aiosqlite.Error as e:
            logger.error(f"Error checking if document {document_id} exists: {e}", exc_info=True)
            return False # Return False on error 