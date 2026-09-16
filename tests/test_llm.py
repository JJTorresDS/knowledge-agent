from types import SimpleNamespace
from unittest.mock import Mock

from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from _core.agent import llm as llm_mod
from _core.config import OPENAI_BASE_URL, OPENROUTER_BASE_URL


def _settings(**overrides) -> SimpleNamespace:
    values = dict(
        llm_provider="openrouter",
        model="nvidia/nemotron-3.5-lightning:free",
        api_key="or-key",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_model_uses_openai_chat_completions(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        llm_mod,
        "settings",
        _settings(llm_provider="openai", model="gpt-4o-mini", api_key="sk-test"),
    )
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", fake_client)

    model = llm_mod.build_model()

    assert captured["api_key"] == "sk-test"
    assert captured.get("base_url") in (None, OPENAI_BASE_URL)
    assert model.model == "gpt-4o-mini"
    assert isinstance(model, OpenAIChatCompletionsModel)


def test_build_model_uses_ollama_chat_completions(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        llm_mod,
        "settings",
        _settings(
            llm_provider="ollama",
            model="qwen2.5:7b",
            api_key="ollama",
        ),
    )
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", fake_client)

    model = llm_mod.build_model()

    assert captured["api_key"] == "ollama"
    assert captured["base_url"] == llm_mod.OLLAMA_BASE_URL
    assert model.model == "qwen2.5:7b"
    assert isinstance(model, OpenAIChatCompletionsModel)


def test_build_model_uses_openrouter_chat_completions(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(llm_mod, "settings", _settings(llm_provider="openrouter"))
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", fake_client)

    model = llm_mod.build_model()

    assert captured["api_key"] == "or-key"
    assert captured["base_url"] == OPENROUTER_BASE_URL
    assert model.model == "nvidia/nemotron-3.5-lightning:free"
    assert isinstance(model, OpenAIChatCompletionsModel)


def test_build_model_uses_mistral_chat_completions(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        llm_mod,
        "settings",
        _settings(
            llm_provider="mistral",
            model="mistral-small-latest",
            api_key="mistral-key",
        ),
    )
    monkeypatch.setattr(llm_mod, "AsyncOpenAI", fake_client)

    model = llm_mod.build_model()

    assert captured["api_key"] == "mistral-key"
    assert captured["base_url"] == llm_mod.MISTRAL_BASE_URL
    assert "api.mistral.ai" in captured["base_url"]
    assert model.model == "mistral-small-latest"
    assert isinstance(model, OpenAIChatCompletionsModel)


def test_build_model_never_uses_responses_api(monkeypatch):
    for provider, model_name, key in (
        ("openai", "gpt-4o-mini", "sk"),
        ("openrouter", "nvidia/nemotron-3.5-lightning:free", "or"),
        ("ollama", "qwen2.5:7b", "ollama"),
        ("mistral", "mistral-small-latest", "ms"),
    ):
        monkeypatch.setattr(
            llm_mod,
            "settings",
            _settings(llm_provider=provider, model=model_name, api_key=key),
        )
        monkeypatch.setattr(llm_mod, "AsyncOpenAI", lambda **kwargs: Mock())
        model = llm_mod.build_model()
        assert isinstance(model, OpenAIChatCompletionsModel)
        assert not isinstance(model, OpenAIResponsesModel)
