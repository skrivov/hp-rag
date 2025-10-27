from pathlib import Path

from src.ingest.markdown import MarkdownTOCBuilder, MarkdownTOCBuilderConfig


def test_markdown_toc_builder_creates_hierarchy(tmp_path):
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# Doc Title\n\nIntro text.\n\n## Section One\n\nParagraph one.\n\n### Subsection\n\nDetails here.\n",
        encoding="utf-8",
    )

    builder = MarkdownTOCBuilder(MarkdownTOCBuilderConfig())
    root = builder.build(Path(doc_path))

    assert root.document_id == "sample"
    assert root.title == "sample"
    assert len(root.children) == 1

    section = root.children[0]
    assert section.title == "Doc Title"
    assert section.path.endswith("doc-title")
    assert "Intro text." in section.body
    assert len(section.children) == 1

    subsection = section.children[0]
    assert subsection.title == "Section One"
    assert subsection.level == 2
    assert subsection.body == "Paragraph one."
    assert subsection.children[0].body == "Details here."
