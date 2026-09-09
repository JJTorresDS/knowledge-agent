from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

_DOCUMENT_URL_EXAMPLES = [
    "https://docs.google.com/document/d/1FlKHKxwltF_2S9ADmkfT3B0ajapSMrVKYWRUXf13mno/edit?tab=t.0#heading=h.l1ncunxa9ncf"
]


class Question(BaseModel):
    question: str
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional chat session id (browser auto UUID or a caller-chosen "
            "string such as a support-user id). Turns with the same id are "
            "stored in Postgres and replayed as conversation history. Also "
            "used to group Langfuse traces. Blank values are treated as omitted."
        ),
    )

    @field_validator("session_id", mode="before")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class Answer(BaseModel):
    answer: str
    turn_id: str


def _iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    iso = value.isoformat()
    if iso.endswith("+00:00"):
        return iso[:-6] + "Z"
    if value.tzinfo is None:
        return iso + "Z"
    return iso


class ConversationTurn(BaseModel):
    turn_id: str | None = None
    question: str
    answer: str


class ConversationSummary(BaseModel):
    session_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    turn_count: int = 0
    last_question: str | None = None

    @field_serializer("created_at", "updated_at")
    def serialize_timestamps(self, value: datetime | None) -> str | None:
        return _iso_z(value)


class ConversationDetail(BaseModel):
    session_id: str
    turns: list[ConversationTurn]


class FeedbackIn(BaseModel):
    session_id: str
    rating: Literal[1, -1] = Field(
        description="1 for thumbs up, -1 for thumbs down.",
    )
    turn_id: str | None = Field(
        default=None,
        description="Optional ask turn id from POST /ask so Grafana can join feedback to a reply.",
    )


class FeedbackOut(BaseModel):
    ok: bool = True


class GoogleDocIngest(BaseModel):
    document_url: str = Field(
        ...,
        description=(
            "Paste the Google Doc URL from your browser. "
            "You do not need the document ID — the full link is enough. "
            "Example: https://docs.google.com/document/d/<id>/edit"
        ),
        examples=_DOCUMENT_URL_EXAMPLES,
    )
    summary: str | None = Field(
        default=None,
        description="Optional short summary so the agent can decide when to search this document.",
    )
    chunk_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Max characters per chunk. Omit to use OpenAI File Search's default "
            "(800 tokens ≈ 3200 characters, with 50% overlap). "
            "FAQ / short Q&A pages: 1200–1400, so each page stays in one chunk. "
            "Contracts, statutes, or other long-form docs: 3200 is usually fine."
        ),
        examples=[1300, 3200],
    )


class GoogleDocStructuredIngest(BaseModel):
    document_url: str = Field(
        ...,
        description=(
            "Paste the Google Doc URL from your browser. "
            "You do not need the document ID — the full link is enough. "
            "Example: https://docs.google.com/document/d/<id>/edit"
        ),
        examples=_DOCUMENT_URL_EXAMPLES,
    )
    summary_tag: str = Field(
        ...,
        description="Heading level whose body is the document summary (for example h1).",
        examples=["h1"],
    )
    question_tag: str = Field(
        ...,
        description=(
            "Heading level for each Q&A chunk. The heading title and the text "
            "beneath it are embedded together (for example h2)."
        ),
        examples=["h2"],
    )
    summary: str | None = Field(
        default=None,
        description=(
            "Optional summary override. Omit to use the text beneath summary_tag."
        ),
    )
