from pathlib import Path

from src.models.section import SectionNode
from src.models.tenant import TenantRecord
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore


def build_sample_sections() -> SectionNode:
    root = SectionNode(
        document_id="doc",
        path="doc",
        title="Doc",
        body="Root body",
        level=0,
    )
    alpha = SectionNode(
        document_id="doc",
        path="doc/alpha",
        title="Alpha",
        body="Alpha body with keyword",
        level=1,
    )
    beta = SectionNode(
        document_id="doc",
        path="doc/beta",
        title="Beta",
        body="Beta body",
        level=1,
    )
    root.add_child(alpha)
    root.add_child(beta)
    return root


def test_sqlite_store_upsert_and_search(tmp_path):
    db_path = tmp_path / "hyperlink.db"
    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=Path(db_path)))
    store.initialize()

    root = build_sample_sections()
    store.upsert_sections([root])

    rows = store.search("keyword")
    assert len(rows) == 1
    assert rows[0]["path"] == "doc/alpha"

    beta = store.fetch_by_path("doc/beta")
    assert beta is not None
    assert beta["title"] == "Beta"

    fetched = store.fetch_sections(["doc/beta", "doc/alpha"])
    assert [row["path"] for row in fetched] == ["doc/alpha", "doc/beta"]

    neighbors = store.fetch_neighbors("doc/beta", window=1)
    paths = [row["path"] for row in neighbors]
    assert "doc/alpha" in paths and "doc/beta" in paths

    limited = store.iter_sections(max_level=1)
    assert any(row["path"] == "doc" for row in limited)

    store.close()


def test_tenant_registration_and_filtering(tmp_path):
    db_path = tmp_path / "hyperlink.db"
    store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=db_path))
    store.initialize()

    root = build_sample_sections()
    store.upsert_sections([root])

    tenant = TenantRecord(tenant_id="apex-robotics", name="Apex Robotics, Inc.", role="Buyer", aliases=["Buyer"])
    store.register_document_tenants("doc", [tenant])

    tenants = store.list_tenants()
    assert tenants and tenants[0]["tenant_id"] == "apex-robotics"
    assert store.list_documents_for_tenant("apex-robotics") == ["doc"]

    filtered = store.iter_sections(tenant_id="apex-robotics")
    assert filtered  # sections still returned
    for row in filtered:
        assert row["path"].startswith("doc")

    store.close()
