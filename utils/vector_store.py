"""FAISS-backed semantic search over document chunks."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


@dataclass
class SearchResult:
    text: str
    score: float
    page_number: Optional[int] = None
    source: Optional[str] = None


class VectorStore:
    """Thin wrapper around FAISS with cosine similarity (via normalized
    inner-product search) plus metadata for citations.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = _get_model(model_name)
        self.index: Optional[faiss.Index] = None
        self.chunks: List = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs.astype("float32")

    def create_index(self, chunks: List) -> None:
        """Build the index from either plain strings or Chunk objects."""
        if not chunks:
            raise ValueError("No text chunks to index — the document may be empty.")

        self.chunks = chunks
        texts = [c.text if hasattr(c, "text") else c for c in chunks]
        vectors = self._embed(texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        if self.index is None or not self.chunks:
            return []
        k = min(k, len(self.chunks))
        q_vec = self._embed([query])
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if hasattr(chunk, "text"):
                results.append(
                    SearchResult(
                        text=chunk.text,
                        score=float(score),
                        page_number=getattr(chunk, "page_number", None),
                        source=getattr(chunk, "source", None),
                    )
                )
            else:
                results.append(SearchResult(text=chunk, score=float(score)))
        return results

    def __len__(self) -> int:
        return len(self.chunks)
