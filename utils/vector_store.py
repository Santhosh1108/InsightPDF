"""Backward-compatible entry point for the advanced RAG retriever."""
from .advanced_retriever import SearchResult, VectorStore

__all__ = ["VectorStore", "SearchResult"]
