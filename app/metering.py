"""Cost and latency metering (Phase 5).

Makes the roadmap's "quality per dollar and per second" rule measurable. Components
(embedders, generator) report real API token usage and phase timings into a scoped
Meter; nothing is recorded unless a meter is active, so instrumentation is a no-op
outside measurement and no Protocol had to change.

    with meter() as m:
        chunks = retrieve(question, config)
        answer = generate_answer(question, chunks)
    m.tokens      # {model: {"input": n, "output": n}}
    m.seconds     # {"query_embed": s, "search": s, "generation": s}
    m.cost_usd    # priced via PRICES_USD_PER_MTOK

Design note: results store RAW TOKEN COUNTS, and cost is derived from the price
table. If a price is wrong or changes, the whole matrix is re-priced by editing the
table — no API calls are re-run.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# PRICE TABLE — USD per 1M tokens.
#
# VERIFIED 2026-08-25 against the providers' own pricing pages:
#   OpenAI  https://developers.openai.com/api/docs/pricing
#   Voyage  https://docs.voyageai.com/docs/pricing
# Re-check before publishing; prices change. Because token counts are stored raw,
# correcting a value here re-prices existing results without re-running anything.
#
# NOTE: the voyage-3 family is now LEGACY ("older models") and carries NO free-token
# allowance. The current voyage-4 generation is both cheaper and includes the first
# 200M tokens free — see DECISIONS.md / FINDINGS.md for the contender choice.
# ---------------------------------------------------------------------------
PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # OpenAI embeddings (input-only; embeddings emit no output tokens)
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    # Voyage — current generation (200M free tokens per account)
    "voyage-4-large":         {"input": 0.12, "output": 0.0},
    "voyage-4":               {"input": 0.06, "output": 0.0},
    "voyage-4-lite":          {"input": 0.02, "output": 0.0},
    # Voyage — legacy generation (no free tokens)
    "voyage-3-large":         {"input": 0.18, "output": 0.0},
    "voyage-3":               {"input": 0.06, "output": 0.0},
    "voyage-3-lite":          {"input": 0.02, "output": 0.0},
    # Generation
    "gpt-4o-mini":            {"input": 0.15, "output": 0.60},
}

# Latency phases, in pipeline order.
PHASES = ("query_embed", "search", "generation")


@dataclass
class Meter:
    """Accumulates token usage and phase timings for one measured scope."""

    tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def record_usage(self, model: str, input_tokens: int, output_tokens: int = 0) -> None:
        entry = self.tokens.setdefault(model, {"input": 0, "output": 0})
        entry["input"] += int(input_tokens)
        entry["output"] += int(output_tokens)

    def record_time(self, phase: str, elapsed: float) -> None:
        # Summed, not appended: a hybrid query times `search` twice (vector +
        # BM25/fusion) and both belong to that query's search cost.
        self.seconds[phase] = self.seconds.get(phase, 0.0) + elapsed

    @property
    def unpriced_models(self) -> list[str]:
        """Models used that have no price-table entry (their cost is excluded)."""
        return [m for m in self.tokens if m not in PRICES_USD_PER_MTOK]

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for model, counts in self.tokens.items():
            price = PRICES_USD_PER_MTOK.get(model)
            if price is None:
                continue
            total += counts["input"] / 1e6 * price["input"]
            total += counts["output"] / 1e6 * price["output"]
        return total

    @property
    def total_tokens(self) -> int:
        return sum(c["input"] + c["output"] for c in self.tokens.values())

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    def summary(self) -> dict:
        """Plain-dict form for result files."""
        return {
            "tokens": self.tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "seconds": self.seconds,
            "total_seconds": self.total_seconds,
        }


_active: ContextVar[Meter | None] = ContextVar("active_meter", default=None)


@contextmanager
def meter():
    """Scope a measurement. Components report into the innermost active meter."""
    m = Meter()
    token = _active.set(m)
    try:
        yield m
    finally:
        _active.reset(token)


def record_usage(model: str, input_tokens: int, output_tokens: int = 0) -> None:
    """Report API token usage. No-op when no meter is active."""
    m = _active.get()
    if m is not None:
        m.record_usage(model, input_tokens, output_tokens)


@contextmanager
def timed(phase: str):
    """Time a pipeline phase into the active meter. No-op when none is active."""
    start = time.perf_counter()
    try:
        yield
    finally:
        m = _active.get()
        if m is not None:
            m.record_time(phase, time.perf_counter() - start)
