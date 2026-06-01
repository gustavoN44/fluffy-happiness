"""Document loading: read one source document off disk, return raw text.

Phase 1 scope is deliberately small — a single document in, a single string of
text out. Chunking, embedding, and storage are later steps; this module knows
nothing about them. Supported formats are PDF (via pypdf) and plain text /
markdown; anything else raises rather than silently returning garbage.
"""

from pathlib import Path

from pypdf import PdfReader

# Extensions we treat as already-plain UTF-8 text.
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _load_pdf(path: Path) -> str:
    """Extract text from every page and join with blank lines between pages."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_document(path: str | Path) -> str:
    """Read a single document and return its raw text.

    Args:
        path: Path to a .pdf, .txt, or .md file.

    Returns:
        The document's full text as one string.

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        ValueError: the file extension is not a supported format, or the file
            contained no extractable text.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No document found at: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in _TEXT_SUFFIXES:
        text = _load_text(path)
    else:
        supported = ", ".join(sorted({".pdf", *_TEXT_SUFFIXES}))
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path.name}. "
            f"Supported: {supported}"
        )

    if not text.strip():
        raise ValueError(
            f"Loaded {path.name} but it contained no extractable text "
            "(empty file, or a scanned/image-only PDF)."
        )

    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.loader <path-to-document>")
    doc = load_document(sys.argv[1])
    print(f"Loaded {sys.argv[1]}: {len(doc)} chars, {len(doc.splitlines())} lines")
    print("--- first 300 chars ---")
    print(doc[:300])
