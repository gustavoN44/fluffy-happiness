"""Embedding via Voyage AI — the cross-provider "quality contender" (Phase 3).

Implements the Embedder protocol with `voyage-3-large`. Two points of difference
from the OpenAI embedder:

  - Voyage embeds asymmetrically: documents and queries use different `input_type`s
    ("document" vs "query"), which improves retrieval. This maps onto the interface
    exactly — embed_texts() (chunks, at ingest) uses "document", embed_text() (the
    query, at retrieval) uses "query".
  - Output dimensionality is configurable; we use 1024 (the model default), lower
    than 3-small's 1536 — table-per-config stores each at its own dim.
"""

import voyageai

from app.config import settings

_BATCH_SIZE = 100


class VoyageEmbedder:
    """Embeds via Voyage AI. Implements the Embedder protocol."""

    def __init__(self, model: str = "voyage-3-large", dim: int = 1024, name: str = "voyage"):
        if not settings.voyage_api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set — required for the Voyage embedder. "
                "Add it to .env."
            )
        self.model = model
        self.dim = dim
        self.name = name
        self._client = voyageai.Client(api_key=settings.voyage_api_key)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            result = self._client.embed(
                batch,
                model=self.model,
                input_type=input_type,
                output_dimension=self.dim,
            )
            vectors.extend(result.embeddings)

        for i, vector in enumerate(vectors):
            if len(vector) != self.dim:
                raise ValueError(
                    f"Embedding {i} has {len(vector)} dims, expected "
                    f"{self.dim} for model {self.model}."
                )
        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (chunks) — input_type='document'."""
        return self._embed(texts, input_type="document") if texts else []

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query — input_type='query'."""
        return self._embed([text], input_type="query")[0]


if __name__ == "__main__":
    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    emb = VoyageEmbedder()
    docs = emb.embed_texts([
        "Cannabis sativa was domesticated in East Asia.",
        "The stock market fell sharply on Tuesday.",
    ])
    q = emb.embed_text("Where was cannabis domesticated?")
    print(f"model={emb.model} dim={emb.dim}  (doc dims={len(docs[0])})")
    print(f"cosine(query, related doc)   = {cosine(q, docs[0]):.3f}")
    print(f"cosine(query, unrelated doc) = {cosine(q, docs[1]):.3f}")
