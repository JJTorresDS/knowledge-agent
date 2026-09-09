"""Read conversation summaries and chat-UI turns from Postgres."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from _core.agent.memory import _decode_item, ensure_memory_tables
from _core.db import engine
from _core.monitoring.store import ensure_monitoring_tables

_read_tables_ready = False


def _ensure_read_tables() -> None:
    """CREATE IF NOT EXISTS is expensive on remote Postgres; do it once per process."""
    global _read_tables_ready
    if _read_tables_ready:
        return
    ensure_memory_tables()
    ensure_monitoring_tables()
    _read_tables_ready = True


def _aware(value: datetime | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text_part = part.get("text") or part.get("content")
                if text_part:
                    parts.append(str(text_part))
        return "\n".join(parts).strip()
    return ""


def turns_from_sdk_items(items: list[dict]) -> list[dict]:
    """Pair user/assistant SDK items into chat bubbles; skip tool-only items."""
    pending_user: str | None = None
    turns: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text_value = _content_text(item.get("content"))
        if role == "user" and text_value:
            pending_user = text_value
        elif role in {"assistant", "ai"} and text_value and pending_user is not None:
            turns.append(
                {
                    "turn_id": None,
                    "question": pending_user,
                    "answer": text_value,
                }
            )
            pending_user = None
    return turns


def list_conversations() -> list[dict]:
    _ensure_read_tables()
    with Session(engine) as session:
        session_rows = session.execute(
            text("""
                SELECT session_id, created_at, updated_at
                FROM agent_sessions
            """)
        ).all()
        turn_rows = session.execute(
            text("""
                SELECT session_id, question, created_at
                FROM ask_turns
                WHERE session_id IS NOT NULL
                ORDER BY created_at ASC
            """)
        ).all()

    by_id: dict[str, dict] = {}
    for row in session_rows:
        by_id[row.session_id] = {
            "session_id": row.session_id,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "turn_count": 0,
            "last_question": None,
        }
    for row in turn_rows:
        if not row.session_id:
            continue
        entry = by_id.setdefault(
            row.session_id,
            {
                "session_id": row.session_id,
                "created_at": row.created_at,
                "updated_at": row.created_at,
                "turn_count": 0,
                "last_question": None,
            },
        )
        entry["turn_count"] += 1
        entry["last_question"] = row.question
        if row.created_at is not None and _aware(row.created_at) >= _aware(
            entry["updated_at"]
        ):
            entry["updated_at"] = row.created_at
        if entry["created_at"] is None:
            entry["created_at"] = row.created_at

    return sorted(
        by_id.values(),
        key=lambda row: _aware(row["updated_at"]),
        reverse=True,
    )


def get_conversation(session_id: str) -> dict:
    _ensure_read_tables()
    with Session(engine) as session:
        turn_rows = session.execute(
            text("""
                SELECT id, question, answer, created_at
                FROM ask_turns
                WHERE session_id = :session_id
                ORDER BY created_at ASC
            """),
            {"session_id": session_id},
        ).all()
        if turn_rows:
            return {
                "session_id": session_id,
                "turns": [
                    {
                        "turn_id": row.id,
                        "question": row.question,
                        "answer": row.answer,
                    }
                    for row in turn_rows
                ],
            }
        message_rows = session.execute(
            text("""
                SELECT message_data FROM agent_messages
                WHERE session_id = :session_id
                ORDER BY id ASC
            """),
            {"session_id": session_id},
        ).all()
    items: list[dict] = []
    for row in message_rows:
        item = _decode_item(row.message_data)
        if item is not None:
            items.append(item)
    return {"session_id": session_id, "turns": turns_from_sdk_items(items)}
