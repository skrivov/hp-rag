from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.interfaces import BaseRetriever
from src.models.context import RetrievedContext
from src.rag.storage import FaissVectorConfig, FaissVectorStore


@dataclass(slots=True)
class VectorRetrieverConfig:
    """Configuration for llama-index + FAISS retrieval."""

    faiss_config: FaissVectorConfig
    similarity_top_k: int = 5
    retriever_id: str = "vector"


class VectorRetriever(BaseRetriever):
    """Semantic retriever backed by FAISS via llama-index."""

    def __init__(self, config: VectorRetrieverConfig) -> None:
        self.config = config
        super().__init__(config.retriever_id)
        self._store = FaissVectorStore(config.faiss_config)

    def _retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Tuple[List[RetrievedContext], Dict[str, float]]:
        retriever = self._store.as_retriever(
            similarity_top_k=min(top_k, self.config.similarity_top_k)
        )
        hits = retriever.retrieve(query)

        contexts: List[RetrievedContext] = []
        for hit in hits:
            node = getattr(hit, "node", hit)
            metadata = dict(getattr(node, "metadata", {}) or {})
            text = (
                node.get_content(metadata_mode="all")
                if hasattr(node, "get_content")
                else getattr(node, "text", "")
            )
            path = metadata.get("path", "")
            title = metadata.get("parent_title", metadata.get("title", path))
            score = getattr(hit, "score", 0.0) or 0.0
            metadata.update({"retrieval_score": score})
            contexts.append(
                RetrievedContext(
                    path=path,
                    title=title,
                    text=text,
                    score=score,
                    metadata=metadata if metadata else None,
                )
            )

        contexts = contexts[:top_k]
        return contexts, {"raw_hits": len(hits)}


__all__ = ["VectorRetriever", "VectorRetrieverConfig"]

