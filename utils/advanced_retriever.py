"""Advanced multi-stage retrieval for PDF RAG.

Pipeline:
1. Clean/normalize the query.
2. Retrieve candidates with dense FAISS + sparse BM25.
3. Fuse rankings with Reciprocal Rank Fusion (RRF).
4. Cross-encoder rerank the strongest candidates.
5. Remove near-duplicate chunks.
6. Apply MMR so the final context is both relevant and diverse.
7. Preserve page/source metadata for grounded citations.

The expensive reranker is loaded lazily and can be disabled for low-latency mode.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

_MODEL_CACHE = {}
_RERANKER_CACHE = {}


def _get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _get_reranker(model_name: str) -> CrossEncoder:
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def normalize_query(query: str) -> str:
    """Normalize user input without destroying meaningful punctuation."""
    query = re.sub(r"\s+", " ", (query or "").strip())
    query = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", query)
    return query[:2000]


def _tokens(text: str) -> List[str]:
    return re.findall(r"(?u)\b\w[\w'-]*\b", text.lower())


@dataclass
class SearchResult:
    text: str
    score: float
    page_number: Optional[int] = None
    source: Optional[str] = None
    chunk_id: Optional[int] = None
    retrieval_score: float = 0.0
    rerank_score: Optional[float] = None


class VectorStore:
    """Hybrid dense+sparse retriever with optional cross-encoder reranking."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self.reranker_name = reranker_name
        self.model = _get_model(model_name)
        self.index: Optional[faiss.Index] = None
        self.chunks: List = []
        self._vectors: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=32,
        )
        return np.asarray(vecs, dtype="float32")

    def create_index(self, chunks: List) -> None:
        if not chunks:
            raise ValueError("No text chunks to index — the document may be empty.")

        self.chunks = chunks
        texts = [c.text if hasattr(c, "text") else str(c) for c in chunks]
        self._vectors = self._embed(texts)

        self.index = faiss.IndexFlatIP(self._vectors.shape[1])
        self.index.add(self._vectors)

        self._bm25_tokens = [_tokens(t) for t in texts]
        self._bm25 = BM25Okapi(self._bm25_tokens)

    def _dense_candidates(self, query: str, k: int):
        if self.index is None:
            return []
        k = min(k, len(self.chunks))
        q_vec = self._embed([query])
        scores, indices = self.index.search(q_vec, k)
        return [
            (int(idx), float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def _sparse_candidates(self, query: str, k: int):
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokens(query))
        k = min(k, len(scores))
        # argpartition avoids a full sort for large document sets.
        idxs = np.argpartition(scores, -k)[-k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0]

    @staticmethod
    def _rrf(rank: int, constant: int = 60) -> float:
        return 1.0 / (constant + rank)

    def _hybrid_candidates(self, query: str, candidate_k: int):
        dense = self._dense_candidates(query, candidate_k)
        sparse = self._sparse_candidates(query, candidate_k)

        fused = {}
        for rank, (idx, _) in enumerate(dense, start=1):
            fused.setdefault(idx, {"rrf": 0.0, "dense": 0.0, "sparse": 0.0})
            fused[idx]["rrf"] += self._rrf(rank)
            fused[idx]["dense"] = dense[rank - 1][1]

        for rank, (idx, _) in enumerate(sparse, start=1):
            fused.setdefault(idx, {"rrf": 0.0, "dense": 0.0, "sparse": 0.0})
            fused[idx]["rrf"] += self._rrf(rank)
            fused[idx]["sparse"] = sparse[rank - 1][1]

        return sorted(fused.items(), key=lambda x: x[1]["rrf"], reverse=True)

    def _deduplicate(self, indices: List[int], similarity_threshold: float = 0.92) -> List[int]:
        if not indices or self._vectors is None:
            return indices

        selected = []
        for idx in indices:
            if all(float(np.dot(self._vectors[idx], self._vectors[j])) < similarity_threshold for j in selected):
                selected.append(idx)
        return selected

    def _mmr(
        self,
        query_vec: np.ndarray,
        candidates: List[int],
        scores: dict,
        k: int,
        lambda_mult: float,
    ) -> List[int]:
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)
        while remaining and len(selected) < k:
            best_idx = None
            best_value = -float("inf")
            for idx in remaining:
                relevance = scores[idx]
                redundancy = max(
                    (float(np.dot(self._vectors[idx], self._vectors[j])) for j in selected),
                    default=0.0,
                )
                value = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
                if value > best_value:
                    best_value = value
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: Optional[int] = None,
        rerank: bool = True,
        mmr_lambda: float = 0.72,
        min_rerank_score: float = -1.5,
    ) -> List[SearchResult]:
        if self.index is None or not self.chunks:
            return []

        query = normalize_query(query)
        if not query:
            return []

        candidate_k = candidate_k or max(k * 5, 20)
        candidate_k = min(candidate_k, len(self.chunks))
        fused = self._hybrid_candidates(query, candidate_k)
        if not fused:
            return []

        # Keep a bounded reranking pool so cross-encoder latency stays predictable.
        pool = [idx for idx, _ in fused[:candidate_k]]
        pool = self._deduplicate(pool)

        rerank_scores = {}
        if rerank and pool:
            reranker = _get_reranker(self.reranker_name)
            pairs = [(query, self.chunks[idx].text if hasattr(self.chunks[idx], "text") else str(self.chunks[idx])) for idx in pool]
            scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            rerank_scores = {idx: float(score) for idx, score in zip(pool, scores)}
            pool = [idx for idx in pool if rerank_scores[idx] >= min_rerank_score]
            pool.sort(key=lambda idx: rerank_scores[idx], reverse=True)

        if not pool:
            return []

        query_vec = self._embed([query])[0]
        base_scores = {}
        for idx, info in fused:
            if idx in pool:
                # Blend semantic reranker with the fused retrieval signal.
                base_scores[idx] = (
                    0.82 * rerank_scores.get(idx, info["rrf"])
                    + 0.18 * info["rrf"] * 100
                )

        selected = self._mmr(
            query_vec,
            pool,
            base_scores,
            min(k, len(pool)),
            mmr_lambda,
        )

        results = []
        for idx in selected:
            chunk = self.chunks[idx]
            retrieval_score = base_scores.get(idx, 0.0)
            results.append(
                SearchResult(
                    text=chunk.text if hasattr(chunk, "text") else str(chunk),
                    score=float(rerank_scores.get(idx, retrieval_score)),
                    page_number=getattr(chunk, "page_number", None),
                    source=getattr(chunk, "source", None),
                    chunk_id=getattr(chunk, "chunk_id", idx),
                    retrieval_score=float(retrieval_score),
                    rerank_score=rerank_scores.get(idx),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self.chunks)
