"""Fast unit tests: no API calls, no database, no network.

This is the suite CI runs on every push. The constraint is deliberate — the
evaluation harness needs API keys, costs money, and was observed failing on quota
limits during Phase 5; a CI job that goes red for reasons unrelated to the change
being pushed trains people to ignore CI. So the push gate covers only logic that
can be verified deterministically and for free, and the API-dependent evaluation
runs on demand (see .github/workflows/evaluation.yml and DECISIONS.md D11).

What that leaves is still worth having: the pure logic here is where the two real
bugs of the project lived (the chunker overlap seam, D4) and where a silent change
would do the most damage (config_id, which decides which physical table a config
reads and writes).

Run:  .venv/bin/python -m pytest tests/test_units.py -q
"""

from dataclasses import replace
from itertools import pairwise

import pytest
import tiktoken

from app.chunker import RecursiveChunker
from app.interfaces import Chunk, Chunker, Embedder
from app.metering import PRICES_USD_PER_MTOK, Meter
from app.pipeline import BASELINE, CHUNKER_REGISTRY, EMBEDDER_REGISTRY, PRODUCTION, RunConfig
from app.rbac import ADMIN, PUBLIC, User, acl_condition, where_clause
from app.retriever import rrf_fuse

ENC = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
    return len(ENC.encode(text))


# Long enough to force many chunks, with varied paragraph and sentence lengths so
# the recursive splitter has to use more than one separator level.
SAMPLE = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"word{i}x{j}" for j in range(40)) for i in range(30)
)


# --------------------------------------------------------------------------- chunker


@pytest.mark.parametrize("size", [128, 256, 512])
def test_chunks_respect_size_budget(size):
    """No chunk may exceed chunk_size — the overlap seed is budgeted for, not added
    on top. Violating this would silently blow past the embedding model's limit."""
    chunks = RecursiveChunker(chunk_size=size).chunk(SAMPLE)
    assert chunks, "chunker produced nothing"
    oversized = [(c.chunk_index, ntok(c.text)) for c in chunks if ntok(c.text) > size]
    assert not oversized, f"chunks exceed {size} tokens: {oversized}"


def test_every_seam_has_overlap():
    """Regression test for a real bug: a piece-based retention pass dropped overlap
    at 12 of 30 boundaries. Overlap must exist at EVERY seam, not most of them."""
    chunks = RecursiveChunker(chunk_size=256).chunk(SAMPLE)
    assert len(chunks) > 2, "need several chunks for this to mean anything"
    for prev, curr in pairwise(chunks):
        prev_tail = set(ENC.encode(prev.text)[-64:])
        curr_head = set(ENC.encode(curr.text)[:64])
        assert prev_tail & curr_head, f"no overlap between chunk {prev.chunk_index} and {curr.chunk_index}"


def test_overlap_seam_does_not_corrupt_words():
    """Regression test for DECISIONS.md D4.

    Joining the overlap seed to the chunk body with .strip() deleted the boundary
    whitespace, fusing the last word of the seed to the first of the body ("in" +
    "early" -> "inearly"). It corrupted 33 of 34 chunks and was invisible until
    retrieval scores were inspected. The fix joins with a single space.
    """
    words = [f"alpha{i} beta{i} gamma{i}" for i in range(400)]
    chunks = RecursiveChunker(chunk_size=128).chunk(" ".join(words))
    for c in chunks:
        # Every token in the vocabulary is of the form <name><digits>; a fused seam
        # produces <digits><name>, e.g. "3alpha". That pattern can only be corruption.
        for bad in ("alpha", "beta", "gamma"):
            assert f"0{bad}" not in c.text and f"9{bad}" not in c.text, \
                f"seam corruption in chunk {c.chunk_index}: {c.text[:120]!r}"


def test_chunking_is_deterministic():
    """Same input, same config -> identical chunks. The matrix compares configs by
    re-ingesting; nondeterminism here would make cells incomparable."""
    a = RecursiveChunker(chunk_size=256).chunk(SAMPLE)
    b = RecursiveChunker(chunk_size=256).chunk(SAMPLE)
    assert [c.text for c in a] == [c.text for c in b]


def test_chunk_indices_are_sequential():
    chunks = RecursiveChunker(chunk_size=256).chunk(SAMPLE)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_text_produces_no_chunks():
    assert RecursiveChunker().chunk("") == []


def test_recursive_chunker_satisfies_protocol():
    assert isinstance(RecursiveChunker(), Chunker)


# -------------------------------------------------------------------------- pipeline


def test_config_id_excludes_retrieval_mode():
    """DECISIONS.md D6/D10. retrieval_mode is a READ-time strategy: dense and hybrid
    must resolve to the same physical table, or switching modes would demand a
    re-ingest and the Phase 5 hybrid comparison would not have been free."""
    dense = replace(PRODUCTION, retrieval_mode="dense")
    hybrid = replace(PRODUCTION, retrieval_mode="hybrid")
    assert dense.config_id == hybrid.config_id


def test_config_id_changes_with_chunker_and_embedder():
    """The converse: anything that changes the stored vectors MUST change the table."""
    base = BASELINE.config_id
    assert replace(BASELINE, chunker_params={"chunk_size": 256}).config_id != base
    assert replace(BASELINE, embedder="voyage",
                   embedder_params={"model": "voyage-4-large", "dim": 1024}).config_id != base


def test_config_id_is_stable_across_runs():
    """config_id is a hash used as a table name. If it ever drifts, existing data
    becomes unreachable rather than merely stale."""
    assert BASELINE.config_id == "33fd8fca2f"
    assert PRODUCTION.config_id == "41024635fe"


def test_config_id_is_insensitive_to_param_ordering():
    a = RunConfig(embedder_params={"model": "text-embedding-3-small", "dim": 1536})
    b = RunConfig(embedder_params={"dim": 1536, "model": "text-embedding-3-small"})
    assert a.config_id == b.config_id


def test_production_is_the_measured_winner():
    """Guards the Phase 5 conclusion against an accidental edit."""
    assert PRODUCTION.chunker == "recursive"
    assert PRODUCTION.chunker_params["chunk_size"] == 256
    assert PRODUCTION.embedder == "voyage"
    assert PRODUCTION.retrieval_mode == "hybrid"


def test_baseline_remains_the_phase1_config():
    """BASELINE means 'what we measure against'. Re-pointing it would invalidate
    every stored result that cites it."""
    assert BASELINE.chunker_params["chunk_size"] == 512
    assert BASELINE.embedder_params["model"] == "text-embedding-3-small"
    assert BASELINE.retrieval_mode == "dense"


def test_registries_build_declared_components():
    """The factory must produce objects satisfying the Protocols — this is what lets
    the matrix swap components by config alone."""
    chunker = RunConfig().build_chunker()
    assert isinstance(chunker, Chunker)
    for name in CHUNKER_REGISTRY:
        assert name in ("recursive", "semantic"), f"unexpected chunker {name}"
    assert set(EMBEDDER_REGISTRY) == {"openai", "voyage"}


def test_unknown_component_fails_loudly():
    with pytest.raises(KeyError):
        RunConfig(chunker="does-not-exist").build_chunker()


def test_label_is_human_readable():
    assert BASELINE.label == "recursive512__text-embedding-3-small"
    assert PRODUCTION.label == "recursive256__voyage-4-large"


# ------------------------------------------------------------------------------ rbac


def test_unrestricted_call_has_no_filter():
    """user=None is the trusted/internal path (the eval harness). It must produce an
    EMPTY where clause, not a permissive one that could be silently wrong."""
    assert acl_condition(None) is None
    clause, params = where_clause(None)
    assert clause.as_string(None).strip() == ""
    assert params == []


def test_user_produces_acl_filter():
    clause, params = where_clause(PUBLIC)
    rendered = clause.as_string(None)
    assert "jsonb_exists_any" in rendered
    assert "allowed_roles" in rendered
    assert params == [["public"]]


def test_admin_carries_both_roles():
    """ADMIN keeps 'public' so an admin sees public documents too — a positive
    control the negative test depends on."""
    assert "admin" in ADMIN.roles and "public" in ADMIN.roles


def test_default_user_is_public_not_privileged():
    """A User constructed with no roles must default to the LEAST privilege."""
    assert User().roles == ["public"]


def test_roles_are_passed_as_parameters_not_interpolated():
    """SQL injection guard: roles must arrive as bound params, never inlined."""
    _, params = where_clause(User(id="x", roles=["admin'; DROP TABLE chunks; --"]))
    assert params == [["admin'; DROP TABLE chunks; --"]]


# ------------------------------------------------------------------------------- rrf


class FakeHit:
    """Minimal stand-in for a retrieved chunk — RRF only reads these fields."""

    def __init__(self, idx, distance=None, similarity=None):
        self.source, self.chunk_index = "doc.pdf", idx
        self.content, self.distance, self.similarity = f"chunk {idx}", distance, similarity


def test_rrf_rewards_agreement_between_retrievers():
    """A chunk both retrievers rank first must outrank one that only appears in a
    single list. This is the entire point of fusing."""
    dense = [FakeHit(1), FakeHit(2)]
    lexical = [FakeHit(1), FakeHit(3)]
    out = rrf_fuse(dense, lexical, k=3)
    assert out[0].chunk_index == 1


def test_rrf_score_matches_the_formula():
    """score = Σ 1/(60 + rank), summed over the lists a chunk appears in."""
    out = rrf_fuse([FakeHit(1)], [FakeHit(1)], k=1)
    assert out[0].score == pytest.approx(2 * (1 / 61))


def test_rrf_surfaces_chunks_dense_missed():
    """The q04 case: a chunk dense never returned can still reach the top via BM25.
    Without this, Recall@5 could not have reached 1.000."""
    dense = [FakeHit(i) for i in range(1, 6)]
    out = rrf_fuse(dense, [FakeHit(99)], k=6)
    assert 99 in [c.chunk_index for c in out]


def test_rrf_respects_k():
    out = rrf_fuse([FakeHit(i) for i in range(10)], [FakeHit(i) for i in range(10, 20)], k=4)
    assert len(out) == 4


def test_rrf_handles_empty_lists():
    assert rrf_fuse([], [], k=5) == []
    assert len(rrf_fuse([FakeHit(1)], [], k=5)) == 1


def test_rrf_preserves_dense_fields_and_nulls_lexical_only():
    """Dense-only diagnostics must survive fusion when dense contributed the chunk,
    and be None when only BM25 did — the API surfaces both."""
    out = rrf_fuse([FakeHit(1, distance=0.2, similarity=0.8)], [FakeHit(2)], k=2)
    by_idx = {c.chunk_index: c for c in out}
    assert by_idx[1].similarity == 0.8 and by_idx[1].distance == 0.2
    assert by_idx[2].similarity is None and by_idx[2].distance is None


def test_rrf_output_is_sorted_by_score():
    out = rrf_fuse([FakeHit(i) for i in range(5)], [FakeHit(i) for i in range(3, 8)], k=8)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)


# -------------------------------------------------------------------------- metering


def test_cost_is_computed_from_the_price_table():
    m = Meter()
    m.record_usage("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    assert m.cost_usd == pytest.approx(PRICES_USD_PER_MTOK["gpt-4o-mini"]["input"])


def test_input_and_output_are_priced_separately():
    """Output tokens cost ~4x input for gpt-4o-mini; conflating them would understate
    cost in exactly the direction that flatters the system."""
    p = PRICES_USD_PER_MTOK["gpt-4o-mini"]
    assert p["output"] > p["input"]
    m = Meter()
    m.record_usage("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert m.cost_usd == pytest.approx(p["input"] + p["output"])


def test_usage_accumulates_across_calls():
    m = Meter()
    m.record_usage("gpt-4o-mini", 100, 10)
    m.record_usage("gpt-4o-mini", 200, 20)
    assert m.summary()["total_tokens"] == 330


def test_unpriced_model_is_reported_not_silently_zero():
    """An unknown model must be visible as unpriced. Silently costing $0 would make
    a config look free and corrupt the quality-per-dollar comparison."""
    m = Meter()
    m.record_usage("some-new-model-2027", 1000)
    assert "some-new-model-2027" in m.unpriced_models


def test_every_model_the_matrix_used_is_priced():
    """Guard against a config referencing a model with no price entry."""
    for model in ("text-embedding-3-small", "voyage-4-large", "gpt-4o-mini"):
        assert model in PRICES_USD_PER_MTOK


def test_summary_reports_tokens_per_model():
    m = Meter()
    m.record_usage("gpt-4o-mini", 100, 10)
    m.record_usage("text-embedding-3-small", 50)
    s = m.summary()
    assert set(s["tokens"]) == {"gpt-4o-mini", "text-embedding-3-small"}
    assert s["tokens"]["gpt-4o-mini"]["input"] == 100
