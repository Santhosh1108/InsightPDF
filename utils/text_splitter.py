"""Recursive, sentence-aware text splitting with page-level metadata."""
from dataclasses import dataclass
from typing import List, Optional

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]


@dataclass
class Chunk:
    text: str
    chunk_id: int
    page_number: Optional[int] = None
    source: Optional[str] = None


def _split_on_separator(text: str, sep: str) -> List[str]:
    parts = text.split(sep)
    # re-attach the separator to every piece except the last
    return [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # hard split as a last resort (e.g. one giant unbroken word)
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_seps = separators[0], separators[1:]
    pieces = _split_on_separator(text, sep)

    chunks, buffer = [], ""
    for piece in pieces:
        if len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            if buffer.strip():
                chunks.append(buffer)
            if len(piece) > chunk_size:
                chunks.extend(_recursive_split(piece, chunk_size, rest_seps))
                buffer = ""
            else:
                buffer = piece
    if buffer.strip():
        chunks.append(buffer)
    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    overlapped = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        overlapped.append(tail + curr)
    return overlapped


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Backward-compatible splitter: plain string in, list of strings out."""
    raw = _recursive_split(text.strip(), chunk_size, _SEPARATORS)
    return _add_overlap(raw, chunk_overlap)


def split_pages(
    pages,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    source: Optional[str] = None,
) -> List[Chunk]:
    """Split a list of PageText into Chunk objects that retain page numbers,
    so every answer can cite exactly where it came from.
    """
    chunks: List[Chunk] = []
    cid = 0
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(text=piece, chunk_id=cid, page_number=page.page_number, source=source)
            )
            cid += 1
    return chunks
