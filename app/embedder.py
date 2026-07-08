"""Embedding: turn text into vectors. Phase 3 makes the model swappable.

`OpenAIEmbedder` implements the Embedder protocol, carrying its own `model` and
`dim` so different OpenAI models (and, behind the same interface, other providers)
can be selected by config. `embed_texts()` / `embed_text()` remain as thin
wrappers over a default instance so existing callers are unaffected.

Deliberately pure: text in, vectors out, same order. Knows nothing about chunks,
the database, or retrieval.
"""

from openai import OpenAI

from app.config import settings

# The embeddings endpoint accepts many inputs per request; cap batch size so a
# large document is sent in a few calls rather than one oversized request.
_BATCH_SIZE = 100


class OpenAIEmbedder:
    """Embeds via the OpenAI embeddings API. Implements the Embedder protocol."""

    def __init__(
        self,
        model: str = settings.embedding_model,
        dim: int = settings.embedding_dim,
        name: str = "openai",
    ):
        self.model = model
        self.dim = dim
        self.name = name
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, one vector per input in input order.

        Raises:
            ValueError: a returned vector does not match this embedder's `dim`
                (would otherwise fail at DB insert time).
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            response = self._client.embeddings.create(model=self.model, input=batch)
            # Sort by .index so order is guaranteed regardless of response ordering.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append(item.embedding)

        for i, vector in enumerate(vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"Embedding {i} has {len(vector)} dims, expected "
                    f"{self.dim} for model {self.model}."
                )

        return vectors

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


# Default instance + compat wrappers so existing callers keep working.
_DEFAULT_EMBEDDER = OpenAIEmbedder()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _DEFAULT_EMBEDDER.embed_texts(texts)


def embed_text(text: str) -> list[float]:
    return _DEFAULT_EMBEDDER.embed_text(text)


if __name__ == "__main__":
    # Tiny live smoke test (makes a real, billed API call): embed three short
    # texts and show that the two related ones are closer than the unrelated one.
    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    samples = [
        "Cannabis sativa was domesticated in East Asia.",
        "The domestication of cannabis happened in Asia.",
        "The stock market fell sharply on Tuesday.",
    ]
    vecs = embed_texts(samples)
    print(f"embedded {len(vecs)} texts; dims={len(vecs[0])} (model {settings.embedding_model})")
    print(f"cosine(related)   = {cosine(vecs[0], vecs[1]):.3f}")
    print(f"cosine(unrelated) = {cosine(vecs[0], vecs[2]):.3f}")
