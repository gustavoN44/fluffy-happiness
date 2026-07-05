"""Generate: produce an answer grounded strictly in the retrieved chunks.

"Grounded" is the whole point — the model must answer from the provided context
only, and say so when the context doesn't contain the answer, rather than falling
back on its own parametric knowledge. This is exactly what the Phase 2
faithfulness metric will later score.
"""

from openai import OpenAI

from app.config import settings
from app.retriever import RetrievedChunk

_client = OpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question using "
    "ONLY the provided context passages. Do not use any prior knowledge. If the "
    "answer is not contained in the context, reply exactly: \"I don't know based "
    "on the provided context.\" Be concise and do not invent sources or facts."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    return "\n\n".join(
        f"[{i + 1}] (source: {c.source}#{c.chunk_index})\n{c.content.strip()}"
        for i, c in enumerate(chunks)
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer to `question` from the given context chunks."""
    if not chunks:
        return "I don't know based on the provided context."

    user_message = (
        f"Context passages:\n{_build_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    response = _client.chat.completions.create(
        model=settings.generation_model,
        temperature=settings.generation_temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    import sys

    from app.retriever import retrieve

    question = " ".join(sys.argv[1:]) or "Where was cannabis first domesticated?"
    chunks = retrieve(question)
    answer = generate_answer(question, chunks)
    print(f"Q: {question}\n\nA: {answer}\n")
    print("grounded on:")
    for c in chunks:
        print(f"  - {c.source}#{c.chunk_index} (sim={c.similarity:.3f})")
