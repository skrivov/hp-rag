from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Type

from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _token_set(text: str) -> set[str]:
    return {token for token in _normalize(text).split(" ") if token}


def _normalize_contexts(items: Iterable[str | None]) -> List[str]:
    return [_normalize(item) for item in items if item]


def _matches(candidate: str, references: Iterable[str]) -> bool:
    return any(ref and (ref in candidate or candidate in ref) for ref in references)


def _precision_recall(
    retrieved: Iterable[str],
    references: Iterable[str],
) -> tuple[float, float, int, int]:
    retrieved_list = list(retrieved)
    references_list = list(references)
    if not retrieved_list:
        return 0.0, 0.0, 0, len(references_list)
    if not references_list:
        return 0.0, 0.0, 0, 0

    matches_precision = sum(1 for ctx in retrieved_list if _matches(ctx, references_list))
    matches_recall = sum(1 for ref in references_list if _matches(ref, retrieved_list))

    precision = matches_precision / len(retrieved_list)
    recall = matches_recall / len(references_list) if references_list else 0.0
    return precision, recall, matches_precision, len(retrieved_list)


class AnswerCorrectnessMetric(BaseMetric):
    """Lightweight lexical F1 score between expected and actual answers."""

    async_mode = False
    include_reason = False
    verbose_mode = False

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_tokens = _token_set(test_case.actual_output or "")
        expected_tokens = _token_set(test_case.expected_output or "")

        if not expected_tokens:
            self.score = 0.0
            self.reason = "No expected answer provided."
            return self.score

        if not actual_tokens:
            self.score = 0.0
            self.reason = "Model produced an empty answer."
            return self.score

        overlap = actual_tokens & expected_tokens
        precision = len(overlap) / len(actual_tokens)
        recall = len(overlap) / len(expected_tokens)
        if precision + recall == 0:
            self.score = 0.0
        else:
            self.score = 2 * precision * recall / (precision + recall)

        self.reason = (
            f"precision={precision:.2f}, recall={recall:.2f}, "
            f"overlap_tokens={len(overlap)}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return (self.score or 0.0) >= self.threshold

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Answer Correctness"


class ContextPrecisionMetric(BaseMetric):
    """Portion of retrieved contexts that overlap the reference contexts."""

    async_mode = False
    include_reason = False
    verbose_mode = False

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        retrieved = _normalize_contexts(test_case.retrieval_context or [])
        references = _normalize_contexts(test_case.context or [])

        if not retrieved:
            self.score = 0.0
            self.reason = "No contexts retrieved."
            return self.score

        matches = sum(1 for ctx in retrieved if _matches(ctx, references))
        self.score = matches / len(retrieved)
        self.reason = f"{matches} of {len(retrieved)} contexts matched references"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return (self.score or 0.0) >= self.threshold

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Context Precision"


class ContextRecallMetric(BaseMetric):
    """Portion of reference contexts that are retrieved."""

    async_mode = False
    include_reason = False
    verbose_mode = False

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        references = _normalize_contexts(test_case.context or [])
        retrieved = _normalize_contexts(test_case.retrieval_context or [])

        if not references:
            self.score = 0.0
            self.reason = "No reference contexts supplied."
            return self.score

        matched = sum(1 for ref in references if _matches(ref, retrieved))
        self.score = matched / len(references)
        self.reason = f"Retrieved {matched} of {len(references)} expected contexts"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return (self.score or 0.0) >= self.threshold

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Context Recall"


class ContextF1Metric(BaseMetric):
    """Harmonic mean of context precision and recall."""

    async_mode = False
    include_reason = False
    verbose_mode = False

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        retrieved = _normalize_contexts(test_case.retrieval_context or [])
        references = _normalize_contexts(test_case.context or [])

        if not retrieved or not references:
            self.score = 0.0
            self.reason = "Insufficient contexts to compute F1."
            return self.score

        precision, recall, matches, total_retrieved = _precision_recall(retrieved, references)
        if precision + recall == 0:
            self.score = 0.0
        else:
            self.score = 2 * precision * recall / (precision + recall)
        self.reason = (
            f"precision={precision:.2f}, recall={recall:.2f}, "
            f"matched={matches}/{total_retrieved}"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        if self.error is not None:
            return False
        return (self.score or 0.0) >= self.threshold

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Context F1"


@dataclass(slots=True)
class MetricDefinition:
    """Defines a metric backed by DeepEval."""

    name: str
    kwargs: Dict[str, Any]


DEFAULT_METRICS = [
    MetricDefinition(name="answer_correctness", kwargs={"threshold": 0.5}),
    MetricDefinition(name="context_precision", kwargs={"threshold": 0.5}),
    MetricDefinition(name="context_recall", kwargs={"threshold": 0.5}),
    MetricDefinition(name="context_f1", kwargs={"threshold": 0.5}),
]


_METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "answer_correctness": AnswerCorrectnessMetric,
    "context_precision": ContextPrecisionMetric,
    "context_recall": ContextRecallMetric,
    "context_f1": ContextF1Metric,
}


def instantiate_metrics(definitions: Sequence[MetricDefinition]) -> List[BaseMetric]:
    """Instantiate DeepEval metric objects from lightweight definitions."""

    metrics: List[BaseMetric] = []
    for definition in definitions:
        metric_cls = _METRIC_REGISTRY.get(definition.name)
        if metric_cls is None:
            raise KeyError(f"Unknown metric '{definition.name}'")
        metrics.append(metric_cls(**definition.kwargs))
    return metrics


__all__ = [
    "AnswerCorrectnessMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "ContextF1Metric",
    "MetricDefinition",
    "DEFAULT_METRICS",
    "instantiate_metrics",
]
