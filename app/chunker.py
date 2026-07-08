"""Chunking: split a document's raw text into overlapping chunks.

The recursive, token-based strategy (Phase 1) now lives in `RecursiveChunker`,
parametrized by `chunk_size` / `overlap` so Recursive-512 and Recursive-256 are
just constructor arguments — this is what makes chunking config-swappable in
Phase 3. `chunk_text()` remains as a thin wrapper over a default instance so
existing callers are unaffected.

Strategy: "recursive" means we try to split on the coarsest natural boundary
first (paragraph -> line -> sentence -> word -> character) and only descend to a
finer boundary when a piece is still too large. Pieces are merged back up to a
stride budget, then a token overlap is prepended between consecutive chunks so a
fact straddling a boundary survives in retrieval. Counted in *tokens* (tiktoken),
because cost and the embedding model's limits are denominated in tokens.
"""

import tiktoken

from app.interfaces import Chunk  # shared type; re-exported for existing importers

# text-embedding-3-small tokenizes with cl100k_base; count in those same tokens.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Baseline references (default construction + eval reporting).
CHUNK_SIZE = 512                    # max tokens per chunk (Phase 1 baseline)
OVERLAP = round(CHUNK_SIZE * 0.15)  # 15% -> 77 tokens carried between chunks

# Tried in order: paragraph, line, sentence-ish, word, character. The empty
# string is the terminal fallback — split into individual characters so even
# text with no separators at all can be reduced under the size budget.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


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


class RecursiveChunker:
    """Recursive, token-based splitter. Implements the Chunker protocol.

    chunk_size / overlap are the only knobs; stride = chunk_size - overlap is the
    non-overlapping base budget, so finals stay <= chunk_size after the overlap
    seed is prepended. overlap defaults to 15% of chunk_size.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int | None = None):
        self.chunk_size = chunk_size
        self.overlap = overlap if overlap is not None else round(chunk_size * 0.15)
        self.stride = self.chunk_size - self.overlap
        self.name = f"recursive-{chunk_size}"

    def chunk(self, text: str) -> list[Chunk]:
        base = self._recursive_split(text, SEPARATORS)
        pieces = self._add_overlap(base)
        return [Chunk(text=p, chunk_index=i) for i, p in enumerate(pieces)]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily pack already-separator-bearing pieces into non-overlapping base
        chunks of up to `stride` tokens. Overlap is added later by _add_overlap."""
        chunks: list[str] = []
        current: list[str] = []
        total = 0

        for piece in pieces:
            plen = _ntokens(piece)
            if current and total + plen > self.stride:
                chunks.append("".join(current).strip())
                current = []
                total = 0
            current.append(piece)
            total += plen

        if current:
            chunks.append("".join(current).strip())

        return [c for c in chunks if c]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split `text` into non-overlapping base pieces that fit `stride` where
        possible, descending through `separators` only for oversized pieces."""
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
            if _ntokens(piece) <= self.stride:
                mergeable.append(piece)
                continue
            if mergeable:
                final.extend(self._merge(mergeable))
                mergeable = []
            if remaining:
                final.extend(self._recursive_split(piece, remaining))
            else:
                final.append(piece)  # nothing finer to try; keep as-is
        if mergeable:
            final.extend(self._merge(mergeable))

        return final

    def _add_overlap(self, base: list[str]) -> list[str]:
        """Prepend each base chunk with the trailing `overlap` tokens of its
        predecessor, as one global pass so every boundary gets overlap regardless
        of how the recursion segmented the text.

        The seed and curr are both already .strip()-ed, so the whitespace that
        separated them in the source is gone; join with a single space to avoid
        colliding the last word of the seed with the first word of curr (e.g.
        "in" + "early" -> "inearly"). See DECISIONS.md D4.
        """
        if not base:
            return []
        result = [base[0]]
        for prev, curr in zip(base, base[1:]):
            seed = _tail_tokens(prev, self.overlap)
            result.append((seed + " " + curr).strip())
        return result


# Default instance + compat wrapper so existing callers (store.py) keep working.
_DEFAULT_CHUNKER = RecursiveChunker(CHUNK_SIZE, OVERLAP)


def chunk_text(text: str) -> list[Chunk]:
    """Split a document's raw text into overlapping, index-tagged chunks using the
    baseline recursive config. Thin wrapper over the default RecursiveChunker."""
    return _DEFAULT_CHUNKER.chunk(text)


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
