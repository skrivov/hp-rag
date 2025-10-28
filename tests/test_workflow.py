from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.eval.metrics import MetricDefinition
from src.orchestration.datasets import DatasetSpec, register_dataset, unregister_dataset
from src.orchestration.workflow import BenchmarkWorkflow


class StaticSelectorLLM:
    def complete(self, prompt: str):  # pragma: no cover - trivial wrapper
        return SimpleNamespace(text='["doc1/passage-1"]')


class WorkflowEmbeddingProvider:
    def embed(self, text: str):
        return [float(len(text)), 1.0]


class WorkflowLLMScorer:
    def measure(self, test_case, *args, **kwargs):
        return 1.0 if test_case.expected_output == test_case.actual_output else 0.0


def test_workflow_full_cycle(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="synthetic-mini",
        download_url=None,
        corpus_relative_path="corpus.jsonl",
        questions_relative_path="questions.jsonl",
        adapter_name="beir",
        description="Synthetic dataset for workflow tests",
    )
    register_dataset(spec, override=True)
    try:
        data_root = tmp_path / "data"
        artifacts_root = tmp_path / "artifacts"
        workflow = BenchmarkWorkflow(data_root=data_root, artifact_root=artifacts_root)

        dataset_dir = workflow.download(spec.name)
        corpus_path = dataset_dir / "corpus.jsonl"
        corpus_path.write_text(
            json.dumps({"_id": "doc1", "title": "Doc", "text": "Alpha beta.\n\nGamma delta."}) + "\n",
            encoding="utf-8",
        )
        questions_path = dataset_dir / "questions.jsonl"
        questions_path.write_text(
            json.dumps(
                {
                    "query": "Where is alpha?",
                    "expected_answer": "Alpha beta.",
                    "reference_contexts": ["Alpha beta."],
                    "actual_answer": "Alpha beta.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        ingest_stats = workflow.ingest(spec.name, clean_stores=True)
        artifacts_dir = workflow.artifacts_dir(spec.name)
        sqlite_file = artifacts_dir / "hyperlink.db"
        assert sqlite_file.exists()
        assert ingest_stats["documents"] == 1

        metrics = workflow.evaluate(
            spec.name,
            suite_name="synthetic",
            include_vector=False,
            selector_llm=StaticSelectorLLM(),
            top_k=1,
            metrics=[
                MetricDefinition("answer_token_f1", {"threshold": 0.5}),
                MetricDefinition(
                    "answer_embedding_similarity",
                    {"threshold": 0.5, "provider": WorkflowEmbeddingProvider()},
                ),
                MetricDefinition(
                    "answer_llm_correctness",
                    {"threshold": 0.5, "scorer_factory": WorkflowLLMScorer},
                ),
                MetricDefinition("context_precision", {"threshold": 0.5}),
                MetricDefinition("context_recall", {"threshold": 0.5}),
                MetricDefinition("context_f1", {"threshold": 0.5}),
            ],
        )
        assert "hyperlink/answer_token_f1" in metrics

        workflow.remove_dataset(spec.name, remove_artifacts=True)
        assert not workflow.dataset_dir(spec.name).exists()
        assert not workflow.artifacts_dir(spec.name).exists()
    finally:
        unregister_dataset(spec.name)
