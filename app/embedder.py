"""Embedding: turn chunk text into vectors via OpenAI text-embedding-3-small.

Deliberately pure: text in, vectors out, same order. It knows nothing about
chunks, the database, or retrieval — the store step zips these vectors back onto
chunks. Model and dimensionality come from app.config (Phase 1 baseline,
swappable in Phase 3).
"""

from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

# The embeddings endpoint accepts many inputs per request; cap batch size so a
# large document is sent in a few calls rather than one oversized request.
_BATCH_SIZE = 100


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input in input order.

    Raises:
        ValueError: a returned vector does not match the configured
            dimensionality (would otherwise fail at DB insert time).
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        # Sort by .index so order is guaranteed regardless of response ordering.
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(item.embedding)

    for i, vector in enumerate(vectors):
        if len(vector) != settings.embedding_dim:
            raise ValueError(
                f"Embedding {i} has {len(vector)} dims, expected "
                f"{settings.embedding_dim} for model {settings.embedding_model}."
            )

    return vectors


def embed_text(text: str) -> list[float]:
    """Embed a single text (e.g. a query). Convenience wrapper over embed_texts."""
    return embed_texts([text])[0]


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
