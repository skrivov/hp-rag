from __future__ import annotations

import json
from pathlib import Path

from src.ingest.adapters.beir import BeirCorpusAdapter
from src.ingest.adapters.hotpotqa import HotpotQAAdapter
from src.ingest.adapters.miracl import MiraclAdapter
from src.ingest.adapters.squad import SquadAdapter
from src.ingest.pipeline import IngestionPipeline
from src.ingest.simple_chunker import ParagraphChunker
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore


def test_beir_adapter_reads_jsonl(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "doc1", "title": "Doc 1", "text": "Paragraph one.\n\nSecond."})
        + "\n",
        encoding="utf-8",
    )
    adapter = BeirCorpusAdapter(corpus)
    roots = list(adapter.iter_section_roots())
    assert roots[0].title == "Doc 1"
    assert len(roots[0].children) == 2


def test_squad_adapter_parses_context(tmp_path: Path) -> None:
    payload = {
        "data": [
            {
                "title": "Article",
                "paragraphs": [
                    {
                        "context": "Context paragraph.",
                        "qas": [],
                    }
                ],
            }
        ]
    }
    dataset = tmp_path / "squad.json"
    dataset.write_text(json.dumps(payload), encoding="utf-8")
    adapter = SquadAdapter(dataset)
    roots = list(adapter.iter_section_roots())
    assert roots[0].title == "Article"
    assert roots[0].children[0].body == "Context paragraph."


def test_hotpotqa_adapter_handles_jsonl(tmp_path: Path) -> None:
    payload = {
        "_id": "sample",
        "context": [["Page", ["Sentence one.", "Sentence two."]]],
    }
    path = tmp_path / "hotpot.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")
    adapter = HotpotQAAdapter(path)
    roots = list(adapter.iter_section_roots())
    assert roots[0].title == "Page"
    assert "Sentence one." in roots[0].children[0].body


def test_miracl_adapter_reads_tsv(tmp_path: Path) -> None:
    corpus = tmp_path / "en.tsv"
    corpus.write_text("doc1\turl\tTitle\tDocument text", encoding="utf-8")
    adapter = MiraclAdapter(corpus)
    roots = list(adapter.iter_section_roots())
    assert roots[0].title == "Title"
    assert roots[0].children[0].body == "Document text"


def test_ingestion_pipeline_accepts_adapter_sections(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"_id": "doc1", "text": "Content."}) + "\n", encoding="utf-8")
    adapter = BeirCorpusAdapter(corpus)
    sqlite_path = tmp_path / "store.db"
    pipeline = IngestionPipeline(
        toc_builder=None,
        chunker=ParagraphChunker(),
        sqlite_store=SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=sqlite_path)),
        vector_store=None,
    )
    result = pipeline.ingest_sections(adapter.iter_section_roots())
    assert result.documents_processed == 1
    assert result.sections_written >= 1
