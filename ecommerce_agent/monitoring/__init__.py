"""Production monitoring: Prometheus metrics + Postgres persistence."""

from __future__ import annotations

import logging
import uuid

from ecommerce_agent.monitoring.metrics import (
    observe_ask,
    observe_feedback,
    word_count,
)
from ecommerce_agent.monitoring.store import persist_ask_turn, persist_feedback

logger = logging.getLogger(__name__)


def record_ask_turn(
    *,
    session_id: str | None,
    question: str,
    answer: str,
    question_words: int,
    answer_words: int,
    latency_ms: float,
) -> str:
    turn_id: str | None = None
    try:
        turn_id = persist_ask_turn(
            session_id=session_id,
            question=question,
            answer=answer,
            question_words=question_words,
            answer_words=answer_words,
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("Failed to persist ask turn for Grafana/Postgres")
    try:
        observe_ask(
            latency_seconds=latency_ms / 1000.0,
            question_words=question_words,
            answer_words=answer_words,
        )
    except Exception:
        logger.exception("Failed to record Prometheus ask metrics")
    return turn_id or str(uuid.uuid4())


def record_feedback(
    *,
    session_id: str,
    rating: int,
    turn_id: str | None = None,
) -> None:
    try:
        persist_feedback(session_id=session_id, rating=rating, turn_id=turn_id)
    except Exception:
        logger.exception("Failed to persist feedback for Grafana/Postgres")
    try:
        observe_feedback(rating)
    except Exception:
        logger.exception("Failed to record Prometheus feedback metrics")
