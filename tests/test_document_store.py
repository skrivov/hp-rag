from __future__ import annotations

from src.models.document import DocumentStatus, PluginStatus
from src.models.section import SectionNode
from src.models.tenant import TenantRecord
from src.hp_rag.storage import SQLiteHyperlinkConfig, SQLiteHyperlinkStore
from src.server.documents.store import DocumentStore


def test_document_store_crud(tmp_path):
    db_path = tmp_path / "docs.db"
    store = DocumentStore(db_path)
    sqlite_store = SQLiteHyperlinkStore(SQLiteHyperlinkConfig(db_path=db_path))
    sqlite_store.initialize()

    store.upsert_document(
        document_id="doc-1",
        name="Doc One",
        source_path=tmp_path / "doc-1.md",
        mime_type="text/markdown",
        size_bytes=123,
        status=DocumentStatus.QUEUED,
    )
    store.set_plugin_status("doc-1", "hp-rag", PluginStatus.QUEUED)

    docs = store.list_documents()
    assert len(docs) == 1
    doc = docs[0]
    assert doc.name == "Doc One"
    assert doc.status == DocumentStatus.QUEUED
    assert doc.plugin_states[0].plugin_name == "hp-rag"
    assert doc.tenants == []

    store.update_status("doc-1", DocumentStatus.READY)
    store.set_plugin_status("doc-1", "hp-rag", PluginStatus.READY, stats={"sections": 5})

    fetched = store.get_document("doc-1")
    assert fetched is not None
    assert fetched.status == DocumentStatus.READY
    assert fetched.plugin_states[0].stats["sections"] == 5
    assert fetched.tenants == []

    root = SectionNode(
        document_id="doc-1",
        path="doc-1",
        title="Doc One",
        body="",
        level=0,
    )
    sqlite_store.upsert_sections([root])
    sqlite_store.register_document_tenants(
        "doc-1",
        [TenantRecord(tenant_id="apex-robotics", name="Apex Robotics, Inc.", role="Buyer")],
    )

    tenant_docs = store.list_documents(tenant_id="apex-robotics")
    assert len(tenant_docs) == 1
    assert tenant_docs[0].tenants and tenant_docs[0].tenants[0].tenant_id == "apex-robotics"

    store.delete_all()
    assert store.list_documents() == []
