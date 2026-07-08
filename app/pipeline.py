"""Pipeline configuration: build a run's chunker + embedder from config (Phase 3).

A `RunConfig` names a chunking strategy and an embedding model (plus their params)
and builds the concrete components via small registries. This is what makes the
Phase 5 experiment matrix a config-only sweep — no pipeline code changes per cell.

Each config has a stable `config_id` (a hash of its identifying params) used to
name its own vector table in Step 2 (table-per-config, DECISIONS.md D6), so
multiple configs coexist in the DB.
"""

import hashlib
from dataclasses import dataclass, field

from app.chunker import RecursiveChunker
from app.embedder import OpenAIEmbedder
from app.interfaces import Chunker, Embedder

# name -> implementation. Adding a strategy/model is a one-line registry entry.
CHUNKER_REGISTRY: dict[str, type] = {
    "recursive": RecursiveChunker,
}
EMBEDDER_REGISTRY: dict[str, type] = {
    "openai": OpenAIEmbedder,
}


@dataclass
class RunConfig:
    """A complete pipeline configuration: which chunker, which embedder, and K.

    chunker_params / embedder_params are passed straight to the registered class
    constructors, so e.g. Recursive-256 is chunker_params={"chunk_size": 256}.
    """

    chunker: str = "recursive"
    chunker_params: dict = field(default_factory=lambda: {"chunk_size": 512})
    embedder: str = "openai"
    embedder_params: dict = field(
        default_factory=lambda: {"model": "text-embedding-3-small", "dim": 1536}
    )
    retrieval_k: int = 5

    def build_chunker(self) -> Chunker:
        return CHUNKER_REGISTRY[self.chunker](**self.chunker_params)

    def build_embedder(self) -> Embedder:
        return EMBEDDER_REGISTRY[self.embedder](**self.embedder_params)

    @property
    def config_id(self) -> str:
        """Stable short hash identifying this config — used as the table suffix."""
        key = (
            f"{self.chunker}|{sorted(self.chunker_params.items())}|"
            f"{self.embedder}|{sorted(self.embedder_params.items())}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:10]

    @property
    def label(self) -> str:
        """Human-readable identifier for reports/logs."""
        size = self.chunker_params.get("chunk_size", "")
        model = self.embedder_params.get("model", self.embedder)
        return f"{self.chunker}{size}__{model}"


# The Phase 1 baseline expressed as a config: recursive 512/15%, 3-small, K=5.
BASELINE = RunConfig()
