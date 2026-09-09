from pathlib import Path

from agents import Agent

from _core.agent.hooks import PrintToolHooks
from _core.agent.llm import build_model
from _core.agent.tracing import setup_tracing
from _core.tools import AGENT_TOOLS

INSTRUCTIONS = (
    Path(__file__).with_name("instructions.md").read_text(encoding="utf-8").strip()
)

setup_tracing()

agent = Agent(
    name="personal_assistant",
    instructions=INSTRUCTIONS,
    model=build_model(),
    hooks=PrintToolHooks(),
    tools=AGENT_TOOLS,
)


def build_agent() -> Agent:
    return agent
