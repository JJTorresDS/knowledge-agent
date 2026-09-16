from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from _core.config import DEFAULT_EMBEDDING_MODELS, MISTRAL_BASE_URL
from _core.embeddings import _PROVIDERS
from _core.embeddings.mistral import MistralEmbeddingProvider


@pytest.fixture(autouse=True)
def mistral_settings(monkeypatch):
    monkeypatch.setattr(
        "_core.embeddings.mistral.settings",
        SimpleNamespace(
            embedding_provider="mistral",
            embedding_model=DEFAULT_EMBEDDING_MODELS["mistral"],
            embedding_api_key="test",
        ),
    )


def _provider_with_data(items: list) -> MistralEmbeddingProvider:
    provider = MistralEmbeddingProvider(api_key="test")
    provider._client = Mock()
    provider._client.embeddings.create.return_value = Mock(data=items)
    return provider


def test_mistral_is_a_registered_embedding_provider():
    assert "mistral" in _PROVIDERS
    assert _PROVIDERS["mistral"] is MistralEmbeddingProvider


def test_mistral_embed_uses_openai_compatible_client_and_1024_dim():
    items = [
        Mock(index=1, embedding=[0.0] * 1024),
        Mock(index=0, embedding=[1.0] + [0.0] * 1023),
    ]
    provider = _provider_with_data(items)

    result = provider.embed(["kid love vest", "windbreaker"])

    assert result.shape == (2, 1024)
    np.testing.assert_array_equal(result[0], np.array([1.0] + [0.0] * 1023))
    kwargs = provider._client.embeddings.create.call_args.kwargs
    assert kwargs["model"] == "mistral-embed-2312"
    assert kwargs["input"] == ["kid love vest", "windbreaker"]
    assert "dimensions" not in kwargs


def test_mistral_provider_uses_mistral_base_url(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr("_core.embeddings.mistral.OpenAI", fake_openai)

    MistralEmbeddingProvider(api_key="mistral-key")

    assert captured["api_key"] == "mistral-key"
    assert captured["base_url"] == MISTRAL_BASE_URL
