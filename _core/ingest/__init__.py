"""Write-path: chunking, product upsert, document upsert, schema init."""

from _core.ingest.chunking import DEFAULT_CHUNK_CHARS, chunk_text
from _core.ingest.documents import (
    upsert_document,
    upsert_documents_structured,
)
from _core.ingest.products import (
    parse_products_csv,
    update_embedding,
    upsert_products_batch,
)
from _core.ingest.schema import init_db

__all__ = [
    "DEFAULT_CHUNK_CHARS",
    "chunk_text",
    "init_db",
    "parse_products_csv",
    "update_embedding",
    "upsert_document",
    "upsert_documents_structured",
    "upsert_products_batch",
]
