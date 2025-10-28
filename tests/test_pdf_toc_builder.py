from __future__ import annotations

from pathlib import Path

import fitz

from src.ingest.pdf import PyMuPDFTOCBuilder, PyMuPDFTOCBuilderConfig


def _make_sample_pdf(path: Path) -> None:
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "Section 1: Overview\nThis contract outlines responsibilities and scope.",
    )

    second_page = document.new_page()
    second_page.insert_text(
        (72, 72),
        "Section 2: Payment Terms\nCompensation will be remitted within 30 days.",
    )

    document.set_toc(
        [
            [1, "Overview", 1],
            [1, "Payment Terms", 2],
        ]
    )

    document.save(path)
    document.close()


def test_pdf_builder_creates_section_tree(tmp_path):
    pdf_path = tmp_path / "contract.pdf"
    _make_sample_pdf(pdf_path)

    builder = PyMuPDFTOCBuilder(PyMuPDFTOCBuilderConfig())
    root = builder.build(pdf_path)

    assert root.document_id == "contract"
    assert len(root.children) == 2

    first_section = root.children[0]
    second_section = root.children[1]

    assert first_section.title == "Overview"
    assert "scope" in first_section.body.lower()
    assert second_section.title == "Payment Terms"
    assert "compensation" in second_section.body.lower()
