import logging

from fastapi import APIRouter

from ecommerce_agent.api.schemas import FeedbackIn, FeedbackOut
from ecommerce_agent.monitoring import record_feedback

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackOut)
def feedback(payload: FeedbackIn) -> FeedbackOut:
    try:
        record_feedback(
            session_id=payload.session_id,
            rating=payload.rating,
            turn_id=payload.turn_id,
        )
    except Exception:
        logger.exception("Failed to record feedback monitoring")
    return FeedbackOut(ok=True)
