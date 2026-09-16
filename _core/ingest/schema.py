"""Create product and document tables if they do not exist.

Catalog tables are created once (no drop). ``document_embeddings.metadata``
is additive: ``ensure_document_embeddings_metadata`` runs ``ADD COLUMN IF NOT
EXISTS`` so existing databases pick up the JSONB column without a recreate.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from _core.agent.memory import ensure_memory_tables
from _core.db import engine
from _core.embeddings import get_provider
from _core.monitoring.store import ensure_monitoring_tables


def ensure_document_embeddings_metadata(session: Session | None = None) -> None:
    """Add ``metadata`` JSONB on ``document_embeddings`` when the table exists."""

    def _run(active: Session) -> None:
        exists = active.execute(
            text("""
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'document_embeddings'
            """)
        ).first()
        if exists is None:
            return
        active.execute(
            text("""
                ALTER TABLE document_embeddings
                ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            """)
        )

    if session is not None:
        _run(session)
        return
    with Session(engine) as owned:
        _run(owned)
        owned.commit()


def init_db() -> None:
    """Create product and document tables if they do not exist.

    Enables the pgvector extension first. If any of those tables already
    exist, still creates conversation memory tables (`agent_sessions`,
    `agent_messages`) and production monitoring tables (`ask_turns`,
    `conversation_feedback`) if they are missing, then raises RuntimeError. Drop
    the catalog tables manually to recreate that schema.
    """
    provider = get_provider()
    with Session(engine) as session:
        existing = session.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN
                      ('product_embeddings', 'documents', 'document_embeddings')
                ORDER BY table_name
            """)
        ).scalars().all()
        if existing:
            ensure_memory_tables(session)
            ensure_monitoring_tables(session)
            ensure_document_embeddings_metadata(session)
            session.commit()
            raise RuntimeError(
                "Refusing to initialize: these tables already exist: "
                f"{', '.join(existing)}. Drop them manually if you want to "
                "recreate the schema, then retry."
            )

        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.execute(
            text(f"""
                CREATE TABLE product_embeddings (
                    id SERIAL PRIMARY KEY,
                    sku TEXT UNIQUE NOT NULL,
                    name TEXT,
                    price TEXT,
                    description TEXT,
                    content TEXT NOT NULL,
                    embedding VECTOR({provider.embedding_dim}) NOT NULL,
                    embedding_model TEXT NOT NULL
                )
            """)
        )
        session.execute(
            text("""
                CREATE INDEX product_embeddings_embedding_idx
                ON product_embeddings
                USING hnsw (embedding vector_cosine_ops)
            """)
        )

        session.execute(
            text("""
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    has_embedding BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    updated_at TIMESTAMPTZ DEFAULT now(),
                    embedded_at TIMESTAMPTZ
                )
            """)
        )
        session.execute(
            text(f"""
                CREATE TABLE document_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({provider.embedding_dim}) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    UNIQUE (document_id, chunk_index)
                )
            """)
        )
        session.execute(
            text("""
                CREATE INDEX document_embeddings_embedding_idx
                ON document_embeddings
                USING hnsw (embedding vector_cosine_ops)
            """)
        )
        ensure_memory_tables(session)
        ensure_monitoring_tables(session)
        session.commit()
