from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Tuple

from src.models.context import RetrievedContext


@dataclass(slots=True)
class RetrievalOutput:
    """Normalized response returned by every retriever."""

    retriever_id: str
    query: str
    top_k: int
    contexts: List[RetrievedContext]
    metadata: Dict[str, Any]


class BaseRetriever(ABC):
    """Common interface for all retrieval strategies."""

    def __init__(self, retriever_id: str) -> None:
        self.retriever_id = retriever_id

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalOutput:
        """Return the top_k contexts for a query with normalized metadata."""

        start = perf_counter()
        contexts, info = self._retrieve(query, top_k=top_k)
        duration_ms = (perf_counter() - start) * 1000
        metadata = dict(info or {})
        metadata.setdefault("duration_ms", duration_ms)
        metadata.setdefault("retrieved", len(contexts))
        return RetrievalOutput(
            retriever_id=self.retriever_id,
            query=query,
            top_k=top_k,
            contexts=contexts,
            metadata=metadata,
        )

    @abstractmethod
    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Tuple[List[RetrievedContext], Dict[str, Any]]:
        """Subclass implementation returning contexts and optional metadata."""


__all__ = ["BaseRetriever", "RetrievalOutput"]
