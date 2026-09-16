"""Knowledge-base document writes."""

from __future__ import annotations

import json
import re
import uuid

from sqlalchemy import text
from sqlalchemy.orm import Session

from _core.db import engine
from _core.embeddings import get_provider
from _core.ingest.chunking import (
    DEFAULT_CHUNK_CHARS,
    chunk_text,
    parse_structured_document,
)
from _core.ingest.schema import ensure_document_embeddings_metadata

_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


def first_heading(text_value: str) -> str | None:
    """Return the first ATX heading title in ``text_value``, if any."""
    match = _ATX_HEADING_RE.search(text_value or "")
    if match is None:
        return None
    title = match.group(2).strip()
    return title or None


def chunk_records_from_sections(
    sections: list[dict],
    chunk_chars: int | None = None,
) -> list[dict]:
    """Character-chunk each section and attach tab / heading metadata."""
    max_chars = chunk_chars or DEFAULT_CHUNK_CHARS
    overlap = max_chars // 2
    records: list[dict] = []
    for section in sections:
        text_value = (section.get("text") or "").strip()
        if not text_value:
            continue
        base: dict = {}
        tab = section.get("tab")
        tab_id = section.get("tab_id")
        if tab:
            base["tab"] = tab
        if tab_id:
            base["tab_id"] = tab_id
        for chunk in chunk_text(text_value, max_chars=max_chars, overlap=overlap):
            metadata = dict(base)
            heading = first_heading(chunk)
            if heading:
                metadata["heading"] = heading
            records.append({"content": chunk, "metadata": metadata})
    return records


def _structured_records_from_sections(
    sections: list[dict],
    summary_tag: str,
    question_tag: str,
) -> tuple[str | None, list[dict]]:
    summary: str | None = None
    records: list[dict] = []
    for section in sections:
        text_value = section.get("text") or ""
        extracted_summary, chunks = parse_structured_document(
            text_value,
            summary_tag=summary_tag,
            question_tag=question_tag,
        )
        if summary is None and extracted_summary:
            summary = extracted_summary
        base: dict = {}
        tab = section.get("tab")
        tab_id = section.get("tab_id")
        if tab:
            base["tab"] = tab
        if tab_id:
            base["tab_id"] = tab_id
        for chunk in chunks:
            metadata = dict(base)
            heading = first_heading(chunk)
            if heading:
                metadata["heading"] = heading
            records.append({"content": chunk, "metadata": metadata})
    return summary, records


def upsert_document(
    filename: str,
    content: str,
    document_id: str | None = None,
    summary: str | None = None,
    chunk_chars: int | None = None,
    chunks: list[str] | None = None,
    chunk_metadata: list[dict] | None = None,
    sections: list[dict] | None = None,
) -> dict:
    """Insert or replace a document and embed its chunks.

    Re-uploading the same `document_id` (or `filename` when no id is
    given) replaces the previous file and its chunks. Pass `document_id`
    to use a stable identifier such as a Google Doc ID. Pass `chunks` to
    skip character-window splitting (used by structured heading ingest).
    Pass `sections` (per-tab text + tab metadata) to chunk each tab and
    store tab / heading metadata on each embedding row.
    """
    ensure_document_embeddings_metadata()
    provider = get_provider()
    max_chars: int | None
    records: list[dict]

    if sections is not None:
        records = chunk_records_from_sections(sections, chunk_chars=chunk_chars)
        max_chars = chunk_chars or DEFAULT_CHUNK_CHARS
        if not content.strip():
            content = "\n".join(
                (section.get("text") or "") for section in sections
            ).strip()
    elif chunks is not None:
        max_chars = chunk_chars
        cleaned = [chunk.strip() for chunk in chunks if chunk.strip()]
        metas = chunk_metadata or [{} for _ in cleaned]
        if len(metas) != len(cleaned):
            raise ValueError("chunk_metadata length must match chunks")
        records = []
        for chunk, meta in zip(cleaned, metas):
            metadata = dict(meta or {})
            heading = metadata.get("heading") or first_heading(chunk)
            if heading:
                metadata["heading"] = heading
            records.append({"content": chunk, "metadata": metadata})
    else:
        max_chars = chunk_chars or DEFAULT_CHUNK_CHARS
        overlap = max_chars // 2
        records = [
            {
                "content": chunk,
                "metadata": (
                    {"heading": heading}
                    if (heading := first_heading(chunk))
                    else {}
                ),
            }
            for chunk in chunk_text(content, max_chars=max_chars, overlap=overlap)
        ]

    if not records:
        raise ValueError(f"Document '{filename}' has no text to embed")

    texts = [record["content"] for record in records]
    vectors = provider.embed(texts)

    with Session(engine) as session:
        existing_id = None
        if document_id is not None:
            existing_id = session.execute(
                text("SELECT id FROM documents WHERE id = :id"),
                {"id": document_id},
            ).scalar_one_or_none()
        else:
            existing_id = session.execute(
                text("SELECT id FROM documents WHERE filename = :filename"),
                {"filename": filename},
            ).scalar_one_or_none()

        if existing_id is None:
            document_id = document_id or f"file_{uuid.uuid4().hex}"
            session.execute(
                text("""
                    INSERT INTO documents (
                        id, filename, content, summary, has_embedding, updated_at
                    )
                    VALUES (
                        :id, :filename, :content, :summary, FALSE, now()
                    )
                """),
                {
                    "id": document_id,
                    "filename": filename,
                    "content": content,
                    "summary": summary,
                },
            )
        else:
            document_id = existing_id
            session.execute(
                text("""
                    UPDATE documents
                    SET filename = :filename,
                        content = :content,
                        summary = COALESCE(:summary, documents.summary),
                        has_embedding = FALSE,
                        updated_at = now(),
                        embedded_at = NULL
                    WHERE id = :id
                """),
                {
                    "id": document_id,
                    "filename": filename,
                    "content": content,
                    "summary": summary,
                },
            )
            session.execute(
                text("DELETE FROM document_embeddings WHERE document_id = :id"),
                {"id": document_id},
            )

        session.execute(
            text("""
                INSERT INTO document_embeddings (
                    document_id, chunk_index, content, embedding,
                    embedding_model, metadata
                )
                VALUES (
                    :document_id, :chunk_index, :content,
                    CAST(:embedding AS vector), :embedding_model,
                    CAST(:metadata AS jsonb)
                )
            """),
            [
                {
                    "document_id": document_id,
                    "chunk_index": index,
                    "content": record["content"],
                    "embedding": str(vector.tolist()),
                    "embedding_model": provider.model_name,
                    "metadata": json.dumps(record["metadata"] or {}),
                }
                for index, (record, vector) in enumerate(zip(records, vectors))
            ],
        )
        session.execute(
            text("""
                UPDATE documents
                SET has_embedding = TRUE, embedded_at = now()
                WHERE id = :id
            """),
            {"id": document_id},
        )
        stored_summary = session.execute(
            text("SELECT summary FROM documents WHERE id = :id"),
            {"id": document_id},
        ).scalar_one()
        session.commit()

    return {
        "document_id": document_id,
        "filename": filename,
        "summary": stored_summary,
        "chunks": len(records),
        "chunk_chars": max_chars,
        "has_embedding": True,
    }


def upsert_documents_structured(
    filename: str,
    content: str,
    summary_tag: str,
    question_tag: str,
    document_id: str | None = None,
    summary: str | None = None,
    sections: list[dict] | None = None,
) -> dict:
    """Insert or replace a heading-structured document (for example a FAQ).

    `summary_tag` and `question_tag` are heading levels such as `h1` /
    `h2`. Text beneath the summary heading is stored on `documents.summary`
    unless `summary` is passed. Each question heading plus the text
    beneath it is embedded as one chunk. When `sections` is passed, each
    tab is parsed separately so chunks keep tab metadata.
    """
    if sections is not None:
        extracted_summary, records = _structured_records_from_sections(
            sections,
            summary_tag=summary_tag,
            question_tag=question_tag,
        )
        if not records:
            raise ValueError(
                f"Document '{filename}' has no question chunks to embed "
                f"(looking for {question_tag} headings)"
            )
        caller_summary = (summary or "").strip() or None
        joined = content.strip() or "\n".join(
            (section.get("text") or "") for section in sections
        )
        return upsert_document(
            filename=filename,
            content=joined,
            document_id=document_id,
            summary=caller_summary or extracted_summary,
            chunks=[record["content"] for record in records],
            chunk_metadata=[record["metadata"] for record in records],
        )

    extracted_summary, chunks = parse_structured_document(
        content,
        summary_tag=summary_tag,
        question_tag=question_tag,
    )
    if not chunks:
        raise ValueError(
            f"Document '{filename}' has no question chunks to embed "
            f"(looking for {question_tag} headings)"
        )
    caller_summary = (summary or "").strip() or None
    return upsert_document(
        filename=filename,
        content=content,
        document_id=document_id,
        summary=caller_summary or extracted_summary,
        chunks=chunks,
        chunk_metadata=[
            {"heading": heading} if (heading := first_heading(chunk)) else {}
            for chunk in chunks
        ],
    )
