from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Protocol, Sequence

from dotenv import load_dotenv


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split(" ") if token}


def precision_recall(
    retrieved: Iterable[str],
    references: Iterable[str],
) -> tuple[float, float, int, int]:
    retrieved_list = list(retrieved)
    references_list = list(references)
    if not retrieved_list:
        return 0.0, 0.0, 0, 0
    if not references_list:
        return 0.0, 0.0, 0, len(retrieved_list)

    matches_precision = sum(1 for cand in retrieved_list if cand in references_list)
    matches_recall = sum(1 for ref in references_list if ref in retrieved_list)

    precision = matches_precision / len(retrieved_list)
    recall = matches_recall / len(references_list)
    return precision, recall, matches_precision, len(retrieved_list)


class AnswerSimilarityStrategy(ABC):
    """Strategy contract for computing similarity between expected and actual answers."""

    @abstractmethod
    def compute(self, expected: str, actual: str) -> float:
        """Return similarity score in range [0,1]."""


class TokenF1Strategy(AnswerSimilarityStrategy):
    """Token-set F1 similarity."""

    def compute(self, expected: str, actual: str) -> float:
        expected_tokens = token_set(expected)
        actual_tokens = token_set(actual)

        if not expected_tokens:
            return 0.0
        if not actual_tokens:
            return 0.0

        overlap = expected_tokens & actual_tokens
        precision = len(overlap) / len(actual_tokens)
        recall = len(overlap) / len(expected_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class EmbeddingProvider(Protocol):
    """Minimal protocol for embedding providers."""

    def embed(self, text: str) -> Sequence[float]:
        ...


class OpenAIEmbeddingProvider:
    """Embedding provider backed by llama-index OpenAI embeddings."""

    def __init__(
        self,
        *,
        model: str | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        self._embedder = embedder

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            load_dotenv()
            from llama_index.embeddings.openai import OpenAIEmbedding

            self._embedder = OpenAIEmbedding(model=self.model)
        return self._embedder

    def embed(self, text: str) -> Sequence[float]:
        embedder = self._ensure_embedder()
        vector = embedder.get_text_embedding(text)
        return vector


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(min(dot / (norm_a * norm_b), 1.0), -1.0)


class EmbeddingCosineStrategy(AnswerSimilarityStrategy):
    """Similarity strategy based on cosine distance of embeddings."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider | None = None,
        model: str | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.provider = provider or OpenAIEmbeddingProvider(model=model, embedder=embedder)

    def compute(self, expected: str, actual: str) -> float:
        if not expected.strip() or not actual.strip():
            return 0.0
        expected_vector = self.provider.embed(expected)
        actual_vector = self.provider.embed(actual)
        score = cosine_similarity(expected_vector, actual_vector)
        # Map [-1, 1] cosine to [0, 1] similarity
        return (score + 1.0) / 2.0


class LLMJudgementStrategy(AnswerSimilarityStrategy):
    """Similarity strategy using a DeepEval LLM metric."""

    def __init__(
        self,
        *,
        model: str | None = None,
        threshold: float = 0.5,
        scorer: Any | None = None,
        scorer_factory: Callable[[], Any] | None = None,
        **metric_kwargs: Any,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.threshold = threshold
        self.metric_kwargs = dict(metric_kwargs)
        self._scorer = scorer
        self._scorer_factory = scorer_factory

    def _ensure_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        if self._scorer_factory is not None:
            self._scorer = self._scorer_factory()
            return self._scorer

        load_dotenv()
        from deepeval.metrics.answer_correctness import AnswerCorrectnessMetric as DeepEvalAnswerCorrectnessMetric

        self._scorer = DeepEvalAnswerCorrectnessMetric(
            model=self.model,
            threshold=self.threshold,
            **self.metric_kwargs,
        )
        return self._scorer

    def compute(self, expected: str, actual: str) -> float:
        scorer = self._ensure_scorer()
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input="",
            actual_output=actual,
            expected_output=expected,
        )
        score = scorer.measure(test_case)
        return float(score or 0.0)

