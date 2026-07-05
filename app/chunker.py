"""Chunking: split one document's raw text into overlapping chunks.

Phase 1 uses a hardcoded recursive splitter measured in *tokens* (not
characters), because cost and the embedding model's limits are denominated in
tokens and the project optimizes for quality-per-token. The config below is
fixed on purpose — chunking becomes config-swappable in Phase 3, not before.

Strategy: "recursive" means we try to split on the coarsest natural boundary
first (paragraph -> line -> sentence -> word -> character) and only descend to a
finer boundary when a piece is still too large. Pieces are then merged back up
to the size budget, carrying a token overlap between consecutive chunks so a
fact straddling a boundary survives in retrieval.
"""

from dataclasses import dataclass

import tiktoken

# text-embedding-3-small tokenizes with cl100k_base; count in those same tokens.
_ENCODING = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 512          # max tokens per chunk (Phase 1 baseline)
OVERLAP = round(CHUNK_SIZE * 0.15)  # 15% -> 77 tokens carried between chunks
# Base ("new content") budget per chunk before overlap is prepended. Consecutive
# chunks advance by this stride; each then repeats the previous chunk's last
# OVERLAP tokens, so STRIDE + OVERLAP == CHUNK_SIZE keeps finals within budget.
_STRIDE = CHUNK_SIZE - OVERLAP

# Tried in order: paragraph, line, sentence-ish, word, character. The empty
# string is the terminal fallback — split into individual characters so even
# text with no separators at all can be reduced under the size budget.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """A single chunk: its text and its position in the document (0-based)."""

    text: str
    chunk_index: int


def _ntokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _tail_tokens(text: str, n: int) -> str:
    """Return the text of the last `n` tokens of `text` (decoded back to a
    string). Used to seed the overlap region of the next chunk."""
    if n <= 0:
        return ""
    return _ENCODING.decode(_ENCODING.encode(text)[-n:])


def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split on `separator` but keep it attached to the preceding piece, so
    rejoining the pieces reproduces the original text exactly. An empty
    separator splits into individual characters."""
    if separator == "":
        return list(text)
    parts = text.split(separator)
    pieces = [p + separator for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p != ""]


def _merge(pieces: list[str]) -> list[str]:
    """Greedily pack already-separator-bearing pieces into non-overlapping base
    chunks of up to _STRIDE tokens. Overlap is added later, globally, by
    chunk_text. Pieces are concatenated with "" since each carries its separator."""
    chunks: list[str] = []
    current: list[str] = []
    total = 0

    for piece in pieces:
        plen = _ntokens(piece)
        if current and total + plen > _STRIDE:
            chunks.append("".join(current).strip())
            current = []
            total = 0
        current.append(piece)
        total += plen

    if current:
        chunks.append("".join(current).strip())

    return [c for c in chunks if c]


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    """Split `text` into non-overlapping base pieces that fit _STRIDE where
    possible, descending through `separators` only for pieces that remain too
    large."""
    # Pick the first separator that occurs in the text (last one is the fallback).
    separator = separators[-1]
    remaining: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    pieces = _split_keeping_separator(text, separator)

    final: list[str] = []
    mergeable: list[str] = []
    for piece in pieces:
        if _ntokens(piece) <= _STRIDE:
            mergeable.append(piece)
            continue
        # Piece is oversized: flush what we've accumulated, then break this
        # piece down further with the finer separators.
        if mergeable:
            final.extend(_merge(mergeable))
            mergeable = []
        if remaining:
            final.extend(_recursive_split(piece, remaining))
        else:
            final.append(piece)  # nothing finer to try; keep as-is
    if mergeable:
        final.extend(_merge(mergeable))

    return final


def _add_overlap(base: list[str]) -> list[str]:
    """Prepend each base chunk with the trailing OVERLAP tokens of its
    predecessor. Done as one global pass over the finished, ordered chunk list so
    every boundary gets overlap regardless of how the recursion segmented the
    text. base[i] <= _STRIDE and the seed <= OVERLAP, so finals stay <= CHUNK_SIZE.

    The seed and curr are both already .strip()-ed, so the whitespace that
    separated them in the source is gone; join with a single space to avoid
    colliding the last word of the seed with the first word of curr (e.g.
    "in" + "early" -> "inearly"). Original separators are whitespace, so one
    space faithfully reconstructs the boundary after whitespace-normalization."""
    if not base:
        return []
    result = [base[0]]
    for prev, curr in zip(base, base[1:]):
        seed = _tail_tokens(prev, OVERLAP)
        result.append((seed + " " + curr).strip())
    return result


def chunk_text(text: str) -> list[Chunk]:
    """Split a document's raw text into overlapping, index-tagged chunks."""
    base = _recursive_split(text, SEPARATORS)
    pieces = _add_overlap(base)
    return [Chunk(text=p, chunk_index=i) for i, p in enumerate(pieces)]


if __name__ == "__main__":
    import sys

    from app.loader import load_document

    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.chunker <path-to-document>")

    chunks = chunk_text(load_document(sys.argv[1]))
    sizes = [_ntokens(c.text) for c in chunks]
    print(f"{len(chunks)} chunks from {sys.argv[1]}")
    print(f"  tokens/chunk: min={min(sizes)} max={max(sizes)} "
          f"avg={sum(sizes) / len(sizes):.0f} (budget {CHUNK_SIZE}, overlap {OVERLAP})")
    print("--- chunk 0 ---")
    print(chunks[0].text[:280])
