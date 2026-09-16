import numpy as np
from openai import OpenAI

from _core.config import MISTRAL_BASE_URL, require_embedding_provider, settings
from _core.embeddings.base import EmbeddingProvider


class MistralEmbeddingProvider(EmbeddingProvider):
    """Embeddings via Mistral's OpenAI-compatible embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        embedding_dim: int = 1024,
    ):
        self.model_name = require_embedding_provider("mistral", settings)
        self.embedding_dim = embedding_dim
        key = api_key or settings.embedding_api_key
        if not key:
            raise ValueError(
                "MISTRAL_API_KEY is required for the mistral embedding provider"
            )
        self._client = OpenAI(api_key=key, base_url=MISTRAL_BASE_URL)

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        items = list(response.data)
        if items and all(getattr(item, "index", None) is not None for item in items):
            items = sorted(items, key=lambda item: item.index)
        vectors = np.array([item.embedding for item in items], dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors
