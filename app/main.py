"""FastAPI app: expose the RAG pipeline over HTTP.

This is the Phase 1 exit point — POST a question, get back a grounded answer plus
the source passages and their relevance scores (the README's transparency
requirement). Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, Header
from psycopg import errors, sql
from pydantic import BaseModel, Field

from app.db import config_table, connect
from app.generator import generate_answer
from app.metering import meter
from app.pipeline import PRODUCTION
from app.rbac import User
from app.retriever import DEFAULT_K, retrieve

app = FastAPI(title="RAG Evaluation System", version="0.1.0")

# The API serves a single "active" config. Phase 5 promoted this from BASELINE to the
# matrix winner (recursive-256 x voyage-4-large) — the point of the whole exercise was
# that the served config is chosen by measurement, not by whatever was built first.
# Swapping it is a one-line change because Phase 3 made configs interchangeable.
ACTIVE_CONFIG = PRODUCTION


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(DEFAULT_K, ge=1, le=50)


class Source(BaseModel):
    source: str
    chunk_index: int
    score: float                       # ranking score (cosine similarity or RRF)
    similarity: float | None = None    # dense only
    distance: float | None = None      # dense only
    content: str


class Metrics(BaseModel):
    """Real measured cost and latency for THIS request.

    Surfaced deliberately: this project is about measuring quality per dollar and per
    second, and an interface that hides those numbers argues the opposite of its own
    thesis. Values come from the same meter the evaluation harness uses, so what the
    UI shows and what the matrix recorded are the same instrument — not an estimate.
    """

    cost_usd: float
    total_tokens: int
    seconds: dict[str, float]        # per phase: query_embed, search, generation
    total_seconds: float
    config: str                      # which RunConfig answered
    retrieval_mode: str              # dense | hybrid


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    metrics: Metrics


class CorpusInfo(BaseModel):
    """What is actually ingested and answerable right now.

    Exists because the browser cannot know this: the frontend ships a fixed set of
    example questions, but whether the documents those questions are ABOUT have been
    ingested is a fact only the server can check. Without it, an example button can
    produce a confusing refusal that reads as a broken system.
    """

    config: str
    retrieval_mode: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only — deliberately does NOT touch the database.

    docker-compose polls this as the api service's healthcheck. Adding a DB query
    would conflate "the process is up" with "its dependency is up", so a brief
    database blip would mark the API itself unhealthy. Corpus state lives on /corpus.
    """
    return {"status": "ok"}


@app.get("/corpus", response_model=CorpusInfo)
def corpus() -> CorpusInfo:
    """Distinct document sources present in the served config's table."""
    sources: list[str] = []
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT DISTINCT metadata->>'source' FROM {} ORDER BY 1").format(
                    config_table(ACTIVE_CONFIG.config_id)
                )
            )
            sources = [row[0] for row in cur.fetchall() if row[0]]
    except errors.UndefinedTable:
        # Nothing ingested under this config yet — an empty corpus, not an error.
        pass
    return CorpusInfo(
        config=ACTIVE_CONFIG.label,
        retrieval_mode=ACTIVE_CONFIG.retrieval_mode,
        sources=sources,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    x_user_roles: str | None = Header(default=None),
) -> QueryResponse:
    # Identity comes from the auth layer; here, a header. No header => public.
    roles = [r.strip() for r in x_user_roles.split(",")] if x_user_roles else ["public"]
    user = User(id="api-caller", roles=roles)

    # The meter is ContextVar-scoped, so it captures usage recorded deep inside the
    # embedder and generator without either of them knowing a request is being measured.
    with meter() as m:
        chunks = retrieve(request.question, config=ACTIVE_CONFIG, k=request.k, user=user)
        answer = generate_answer(request.question, chunks)
    usage = m.summary()

    sources = [
        Source(
            source=c.source,
            chunk_index=c.chunk_index,
            score=c.score,
            similarity=c.similarity,
            distance=c.distance,
            content=c.content,
        )
        for c in chunks
    ]
    metrics = Metrics(
        cost_usd=usage["cost_usd"],
        total_tokens=usage["total_tokens"],
        seconds=usage["seconds"],
        total_seconds=usage["total_seconds"],
        config=ACTIVE_CONFIG.label,
        retrieval_mode=ACTIVE_CONFIG.retrieval_mode,
    )
    return QueryResponse(answer=answer, sources=sources, metrics=metrics)
