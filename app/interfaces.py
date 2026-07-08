"""Swappable component interfaces for the pipeline (Phase 3).

`Chunker` and `Embedder` are structural `Protocol`s: any object with the right
shape satisfies them, no inheritance required. This is what lets chunking
strategies and embedding models be swapped by configuration (via app.pipeline)
without the rest of the pipeline knowing which concrete implementation it holds.

`Chunk` lives here as the shared unit both the chunker (produces) and the store
(consumes) depend on, so neither has to import the other.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Chunk:
    """A single chunk: its text and its position in the document (0-based)."""

    text: str
    chunk_index: int


@runtime_checkable
class Chunker(Protocol):
    """Splits a document's raw text into overlapping, index-tagged chunks."""

    name: str

    def chunk(self, text: str) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. `dim` must match the DB vector column for its config."""

    name: str
    model: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_text(self, text: str) -> list[float]: ...
