"""FastAPI app: expose the RAG pipeline over HTTP.

This is the Phase 1 exit point — POST a question, get back a grounded answer plus
the source passages and their relevance scores (the README's transparency
requirement). Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from app.generator import generate_answer
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


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    x_user_roles: str | None = Header(default=None),
) -> QueryResponse:
    # Identity comes from the auth layer; here, a header. No header => public.
    roles = [r.strip() for r in x_user_roles.split(",")] if x_user_roles else ["public"]
    user = User(id="api-caller", roles=roles)
    chunks = retrieve(request.question, config=ACTIVE_CONFIG, k=request.k, user=user)
    answer = generate_answer(request.question, chunks)
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
    return QueryResponse(answer=answer, sources=sources)
