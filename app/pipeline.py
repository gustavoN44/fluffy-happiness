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
from app.semantic_chunker import SemanticChunker
from app.voyage_embedder import VoyageEmbedder

# name -> implementation. Adding a strategy/model is a one-line registry entry.
CHUNKER_REGISTRY: dict[str, type] = {
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}
EMBEDDER_REGISTRY: dict[str, type] = {
    "openai": OpenAIEmbedder,
    "voyage": VoyageEmbedder,
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
    # Read-time only: "dense" (vector) or "hybrid" (dense + BM25 fused via RRF).
    # Deliberately NOT part of config_id — a hybrid config reads the same table its
    # dense twin populated; only the retrieval strategy differs.
    retrieval_mode: str = "dense"

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
# Deliberately left as the RunConfig defaults even after the Phase 5 matrix picked a
# different winner: BASELINE means "what we measure against", and re-pointing it would
# silently invalidate every stored result and comparison that cites it.
BASELINE = RunConfig()

# Phase 5 matrix winner, and what the API actually serves. Beat the baseline on all
# three dimensions at once: +0.102 composite quality (+12.6%), 42% lower cost per
# query, 17% lower latency. See DECISIONS.md D9 for the scoring rule and FINDINGS.md
# F3/F4 for why this pairing wins — and F6 for the honest limits of that claim
# (it is not statistically separable from the other two Voyage cells).
#
# retrieval_mode="hybrid" comes from the Phase 5 stretch comparison (D10): it wins on
# all nine measured metrics over its dense twin, drives Recall@5 to 1.000, and costs
# +$0.000003 per query with no latency penalty. Because retrieval_mode is excluded
# from config_id (D6), this reads the SAME table the dense config populated — the
# switch needs no re-ingest.
PRODUCTION = RunConfig(
    chunker="recursive",
    chunker_params={"chunk_size": 256},
    embedder="voyage",
    embedder_params={"model": "voyage-4-large", "dim": 1024},
    retrieval_mode="hybrid",
)


# Named configs addressable from the command line, so scripts and CI can target the
# served config by name instead of hardcoding BASELINE. Keeping this next to the
# definitions means a new named config is reachable everywhere the moment it exists.
NAMED_CONFIGS: dict[str, RunConfig] = {
    "baseline": BASELINE,
    "production": PRODUCTION,
}


def resolve_config(name: str) -> RunConfig:
    """Look up a named config, failing with the valid options rather than a KeyError."""
    try:
        return NAMED_CONFIGS[name]
    except KeyError:
        raise SystemExit(
            f"unknown config {name!r}; choose one of: {', '.join(NAMED_CONFIGS)}"
        ) from None


def add_config_arg(parser) -> None:
    """Attach a --config flag to an argparse parser (shared by the CLI entrypoints)."""
    parser.add_argument(
        "--config", default="baseline", choices=sorted(NAMED_CONFIGS),
        help="named pipeline config to use (default: baseline)",
    )
