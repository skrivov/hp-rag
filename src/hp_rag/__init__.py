"""Hyperlink-driven RAG components."""

from .storage import SQLiteHyperlinkStore, SQLiteHyperlinkConfig

try:
    from .retriever import HyperlinkRetriever, HyperlinkRetrieverConfig

    __all_retriever__ = ["HyperlinkRetriever", "HyperlinkRetrieverConfig"]
except ModuleNotFoundError:
    HyperlinkRetriever = HyperlinkRetrieverConfig = None  # type: ignore[assignment]
    __all_retriever__ = []

__all__ = [
    "SQLiteHyperlinkStore",
    "SQLiteHyperlinkConfig",
] + __all_retriever__
