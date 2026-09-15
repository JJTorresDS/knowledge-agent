import logging
import uuid
from time import perf_counter

from fastapi import APIRouter
from agents import Runner
from langfuse import propagate_attributes

from _core.agent.factory import agent, build_agent
from _core.agent.memory import PostgresSession
from _core.agent.tracing import get_client
from _core.api.schemas import Answer, ChatModelsOut, Question
from _core.config import chat_model_choices, provider_for_chat_model, settings
from _core.monitoring import record_ask_turn
from _core.monitoring.metrics import word_count

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/models", response_model=ChatModelsOut)
def list_chat_models() -> ChatModelsOut:
    return ChatModelsOut(default=settings.model, models=chat_model_choices())


@router.post("/ask", response_model=Answer)
async def ask(payload: Question) -> Answer:
    print(f"[agent] called with question: {payload.question!r}", flush=True)
    started = perf_counter()
    result = await _run_agent(payload)
    latency_ms = (perf_counter() - started) * 1000
    answer_text = result.final_output or ""
    try:
        turn_id = record_ask_turn(
            session_id=payload.session_id,
            question=payload.question,
            answer=answer_text,
            question_words=word_count(payload.question),
            answer_words=word_count(answer_text),
            latency_ms=latency_ms,
        )
    except Exception:
        logger.exception("Failed to record ask monitoring")
        turn_id = str(uuid.uuid4())
    print(f"[agent] finished, answer: {answer_text!r}", flush=True)
    return Answer(answer=answer_text, turn_id=turn_id)


def _runner_kwargs(payload: Question) -> dict:
    if not payload.session_id:
        return {}
    return {"session": PostgresSession(payload.session_id)}


def _chat_identity(payload: Question) -> tuple[str, str]:
    if payload.model:
        return provider_for_chat_model(payload.model), payload.model
    return settings.llm_provider, settings.model


async def _run_agent(payload: Question):
    used_agent = build_agent(payload.model)
    run_kwargs = _runner_kwargs(payload)
    if not settings.langfuse_enabled:
        return await Runner.run(used_agent, payload.question, **run_kwargs)

    langfuse = get_client()
    llm_provider, model = _chat_identity(payload)
    attribute_kwargs: dict = {
        "tags": ["ask", "chat"],
        "metadata": {
            "llm_provider": llm_provider,
            "model": model,
        },
    }
    if payload.session_id:
        attribute_kwargs["session_id"] = payload.session_id
    with langfuse.start_as_current_observation(
        as_type="span",
        name="ask",
        input=payload.question,
    ) as observation:
        with propagate_attributes(**attribute_kwargs):
            result = await Runner.run(used_agent, payload.question, **run_kwargs)
            observation.update(output=result.final_output)
            return result
