from fastapi import APIRouter

from _core.agent.conversations import get_conversation, list_conversations
from _core.api.schemas import ConversationDetail, ConversationSummary

router = APIRouter()


@router.get("/sessions", response_model=list[ConversationSummary])
def sessions() -> list[ConversationSummary]:
    return list_conversations()


@router.get("/sessions/{session_id:path}", response_model=ConversationDetail)
def session_detail(session_id: str) -> ConversationDetail:
    return get_conversation(session_id)
