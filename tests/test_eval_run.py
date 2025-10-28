import json
import pytest
from pathlib import Path

import src.ingest  # noqa: F401  # ensure package initializes before storage imports
from src.eval.run import EvaluationConfig, run_suite
from src.eval.metrics import MetricDefinition
from src.orchestration.query_runner import GeneratedAnswer, QueryRunner, QueryRunnerConfig
from src.models.context import RetrievedContext
from src.interfaces import BaseRetriever


class CountingRetriever(BaseRetriever):
    def __init__(self) -> None:
        super().__init__("stub")
        self.queries: list[str] = []

    def _retrieve(self, query: str, *, top_k: int):
        self.queries.append(query)
        context = RetrievedContext(
            path="doc/path",
            title="Title",
            text="alpha beta",
            score=0.9,
        )
        return [context], {"duration_ms": 12.0}


class StubEmbeddingProvider:
    def embed(self, text: str):
        length = max(1, len(text.split()))
        return [float(length), 0.0, 1.0]


class StubLLMScorer:
    def measure(self, test_case, *args, **kwargs):
        expected = (test_case.expected_output or "").strip().lower()
        actual = (test_case.actual_output or "").strip().lower()
        return 1.0 if expected and expected == actual else 0.0


def test_run_suite_aggregates_metrics(tmp_path):
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        '{"query": "first", "expected_answer": "alpha beta", "reference_contexts": ["alpha beta"]}\n'
        '{"query": "second", "expected_answer": "alpha beta", "reference_contexts": ["alpha beta"]}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    retriever = CountingRetriever()
    config = EvaluationConfig(
        suite_name="baseline",
        questions_path=str(dataset_path),
        output_path=str(output_path),
        retriever_ids=["stub"],
        top_k=1,
        metrics=[
            MetricDefinition("answer_token_f1", {"threshold": 0.5}),
            MetricDefinition(
                "answer_embedding_similarity",
                {"threshold": 0.5, "provider": StubEmbeddingProvider()},
            ),
            MetricDefinition(
                "answer_llm_correctness",
                {"threshold": 0.5, "scorer_factory": StubLLMScorer},
            ),
            MetricDefinition("context_precision", {"threshold": 0.5}),
            MetricDefinition("context_recall", {"threshold": 0.5}),
            MetricDefinition("context_f1", {"threshold": 0.5}),
        ],
    )

    result = run_suite(config, {"stub": retriever})

    assert retriever.queries == ["first", "second"]
    assert result.metrics["stub/avg_contexts"] == 1.0
    assert result.metrics["stub/avg_tokens"] == 2.0
    assert result.metrics["stub/avg_latency_ms"] == 12.0
    assert result.metrics["stub/answer_token_f1"] == 1.0
    assert result.metrics["stub/answer_embedding_similarity"] == pytest.approx(1.0, rel=1e-6)
    assert result.metrics["stub/answer_llm_correctness"] == 1.0
    assert result.metrics["stub/context_precision"] == 1.0
    assert result.metrics["stub/context_recall"] == 1.0
    assert result.metrics["stub/context_f1"] == 1.0

    saved = json.loads(Path(output_path).read_text())
    assert saved["suite_name"] == "baseline"


class FixedAnswerRunner(QueryRunner):
    def __init__(self, retriever, answer: str) -> None:
        super().__init__(retriever, QueryRunnerConfig(), llm=None)
        self.answer = answer
        self.calls: list[str] = []

    def generate_answer(self, query, contexts):  # type: ignore[override]
        self.calls.append(query)
        return GeneratedAnswer(prompt="fixed", answer=self.answer)


def test_run_suite_query_runner_factory(tmp_path):
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text(
        '{"query": "who", "expected_answer": "alpha", "reference_contexts": ["alpha"]}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"

    retriever = CountingRetriever()

    created: list[FixedAnswerRunner] = []

    def factory(rid, retriever_obj):
        runner = FixedAnswerRunner(retriever_obj, answer="alpha")
        created.append(runner)
        return runner

    config = EvaluationConfig(
        suite_name="factory",
        questions_path=str(dataset_path),
        output_path=str(output_path),
        retriever_ids=["stub"],
        top_k=1,
        metrics=[
            MetricDefinition("answer_token_f1", {"threshold": 0.5}),
            MetricDefinition(
                "answer_embedding_similarity",
                {"threshold": 0.5, "provider": StubEmbeddingProvider()},
            ),
            MetricDefinition(
                "answer_llm_correctness",
                {"threshold": 0.5, "scorer_factory": StubLLMScorer},
            ),
            MetricDefinition("context_precision", {"threshold": 0.5}),
            MetricDefinition("context_recall", {"threshold": 0.5}),
            MetricDefinition("context_f1", {"threshold": 0.5}),
        ],
    )

    result = run_suite(config, {"stub": retriever}, query_runner_factory=factory)

    assert retriever.queries == ["who"]
    assert created and created[0].calls == ["who"]
    assert result.metrics["stub/answer_token_f1"] == 1.0
    assert result.metrics["stub/answer_embedding_similarity"] == pytest.approx(1.0, rel=1e-6)
    assert result.metrics["stub/answer_llm_correctness"] == 1.0
    assert result.metrics["stub/context_precision"] == 1.0
    assert result.metrics["stub/context_recall"] == 1.0
    assert result.metrics["stub/context_f1"] == 1.0
