"""Semantic chunking: split where the topic shifts, not at fixed sizes.

Implements the Chunker protocol. The strategy (Greg Kamradt / LlamaIndex style):
embed each sentence (with a small neighbour window for stability), measure the
cosine distance between consecutive sentences, and start a new chunk wherever that
distance spikes above a percentile threshold — i.e. where the topic changes.

The breakpoint embedder is FIXED (text-embedding-3-small), independent of whatever
embedder a config later stores the chunks with, so "Semantic" produces the same
chunks across the Phase 5 embedding axis — one variable at a time (DECISIONS.md D6).

Distances and percentiles are computed in pure Python (no numpy dependency), which
is plenty fast at a few hundred sentences.
"""

import math
import re

from app.chunker import RecursiveChunker, _ntokens
from app.embedder import OpenAIEmbedder
from app.interfaces import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb) if na and nb else 1.0


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in 0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class SemanticChunker:
    """Topic-shift chunker. Implements the Chunker protocol.

    breakpoint_percentile: a sentence boundary becomes a chunk break when its
        distance exceeds this percentile of all boundary distances (higher = fewer,
        larger chunks).
    max_tokens: hard cap; a semantic group larger than this is sub-split (no
        overlap) so chunks stay within embedding limits and comparable to recursive.
    buffer_size: sentences of context on each side when embedding, for stability.
    """

    def __init__(
        self,
        breakpoint_percentile: float = 95,
        max_tokens: int = 512,
        buffer_size: int = 1,
        breakpoint_model: str = "text-embedding-3-small",
    ):
        self.breakpoint_percentile = breakpoint_percentile
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.name = f"semantic-p{breakpoint_percentile}"
        # Fixed, decoupled from the storage embedder.
        self._embedder = OpenAIEmbedder(model=breakpoint_model)

    def chunk(self, text: str) -> list[Chunk]:
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            groups = [text.strip()] if text.strip() else []
        else:
            windows = self._with_buffer(sentences)
            vectors = self._embedder.embed_texts(windows)
            distances = [
                _cosine_distance(vectors[i], vectors[i + 1])
                for i in range(len(vectors) - 1)
            ]
            breakpoints = self._breakpoints(distances)
            groups = self._group(sentences, breakpoints)

        pieces: list[str] = []
        for group in groups:
            pieces.extend(self._cap(group))
        return [Chunk(text=p, chunk_index=i) for i, p in enumerate(pieces)]

    def _split_sentences(self, text: str) -> list[str]:
        # Collapse whitespace first so PDF line-wrapping doesn't fragment sentences.
        normalized = re.sub(r"\s+", " ", text).strip()
        return [s.strip() for s in _SENTENCE_RE.split(normalized) if s.strip()]

    def _with_buffer(self, sentences: list[str]) -> list[str]:
        """Combine each sentence with `buffer_size` neighbours on each side."""
        n = len(sentences)
        return [
            " ".join(sentences[max(0, i - self.buffer_size) : min(n, i + self.buffer_size + 1)])
            for i in range(n)
        ]

    def _breakpoints(self, distances: list[float]) -> set[int]:
        if not distances:
            return set()
        threshold = _percentile(distances, self.breakpoint_percentile)
        return {i for i, d in enumerate(distances) if d > threshold}

    def _group(self, sentences: list[str], breakpoints: set[int]) -> list[str]:
        """Join sentences into groups, breaking after each breakpoint index."""
        groups: list[str] = []
        start = 0
        for i in range(len(sentences) - 1):
            if i in breakpoints:
                groups.append(" ".join(sentences[start : i + 1]))
                start = i + 1
        groups.append(" ".join(sentences[start:]))
        return [g.strip() for g in groups if g.strip()]

    def _cap(self, text: str) -> list[str]:
        """Sub-split a group that exceeds max_tokens (no overlap)."""
        if _ntokens(text) <= self.max_tokens:
            return [text]
        sub = RecursiveChunker(chunk_size=self.max_tokens, overlap=0)
        return [c.text for c in sub.chunk(text)]


if __name__ == "__main__":
    import sys

    from app.loader import load_document

    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.semantic_chunker <path-to-document>")

    chunks = SemanticChunker().chunk(load_document(sys.argv[1]))
    sizes = [_ntokens(c.text) for c in chunks]
    print(f"{len(chunks)} semantic chunks from {sys.argv[1]}")
    print(f"  tokens/chunk: min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.0f}")
    print("--- chunk 0 ---")
    print(chunks[0].text[:280])
