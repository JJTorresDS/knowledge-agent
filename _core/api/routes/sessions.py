from datetime import datetime
from html import escape
from urllib.parse import quote

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from _core.agent.conversations import get_conversation, list_conversations
from _core.api.schemas import ConversationDetail, ConversationSummary

router = APIRouter()


def conversation_list_html(rows: list[dict]) -> str:
    if not rows:
        return '<p class="empty">No conversations yet.</p>'
    parts: list[str] = []
    for row in rows:
        session_id = str(row["session_id"])
        href = "/?session_id=" + quote(session_id, safe="")
        preview = row.get("last_question") or "(no messages yet)"
        turns = int(row.get("turn_count") or 0)
        label = "1 turn" if turns == 1 else f"{turns} turns"
        updated = row.get("updated_at")
        meta = label
        if updated is not None:
            if isinstance(updated, datetime):
                meta += " · " + updated.isoformat()
            else:
                meta += " · " + str(updated)
        parts.append(
            f'<a class="conversation" href="{escape(href, quote=True)}">'
            f'<div class="conversation-id">{escape(session_id)}</div>'
            f'<div class="preview">{escape(str(preview))}</div>'
            f'<div class="meta">{escape(meta)}</div>'
            "</a>"
        )
    return "".join(parts)


@router.get("/sessions", response_model=list[ConversationSummary])
def sessions() -> list[ConversationSummary]:
    return list_conversations()


@router.get("/admin/conversations")
def admin_conversations(
    limit: int = Query(10, ge=1, le=500),
    min_turns: int = Query(0, ge=0),
) -> HTMLResponse:
    return HTMLResponse(
        conversation_list_html(
            list_conversations(limit=limit, min_turns=min_turns)
        )
    )


@router.get("/sessions/{session_id:path}", response_model=ConversationDetail)
def session_detail(session_id: str) -> ConversationDetail:
    return get_conversation(session_id)
