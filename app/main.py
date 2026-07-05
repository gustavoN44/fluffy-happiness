"""FastAPI app: expose the RAG pipeline over HTTP.

This is the Phase 1 exit point — POST a question, get back a grounded answer plus
the source passages and their relevance scores (the README's transparency
requirement). Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.generator import generate_answer
from app.retriever import DEFAULT_K, retrieve

app = FastAPI(title="RAG Evaluation System", version="0.1.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(DEFAULT_K, ge=1, le=50)


class Source(BaseModel):
    source: str
    chunk_index: int
    similarity: float
    distance: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chunks = retrieve(request.question, k=request.k)
    answer = generate_answer(request.question, chunks)
    sources = [
        Source(
            source=c.source,
            chunk_index=c.chunk_index,
            similarity=c.similarity,
            distance=c.distance,
            content=c.content,
        )
        for c in chunks
    ]
    return QueryResponse(answer=answer, sources=sources)
