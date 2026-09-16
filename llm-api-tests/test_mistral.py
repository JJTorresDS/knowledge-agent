"""Live Mistral chat and embeddings. Skips if MISTRAL_API_KEY is missing."""

from providers import (
    chat_client,
    embed_client,
    embed_model_for,
    model_for,
    ping_chat,
    ping_embed,
    skip_if_unconfigured,
)


def test_mistral_chat_api_responds():
    skip_if_unconfigured("mistral")
    text = ping_chat(chat_client("mistral"), model_for("mistral"))
    assert text


def test_mistral_embeddings_api_responds():
    skip_if_unconfigured("mistral")
    dim = ping_embed(embed_client("mistral"), embed_model_for("mistral"))
    assert dim == 1024
