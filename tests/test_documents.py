from unittest.mock import Mock

from googleapiclient.errors import HttpError
import pytest

from _core.api.routes import documents as documents_route
from _core.ingest.chunking import parse_structured_document
from _core.ingest.documents import upsert_documents_structured
from tests.conftest import (
    CHUNK_DOCUMENT_URL,
    FAQ_DOCUMENT_ID,
    FAQ_DOCUMENT_URL,
    FAQ_STRUCTURED_TEXT,
    FAQ_TEXT,
    FAQ_TITLE,
)


def test_google_doc_ingest_openapi_examples_use_distinct_docs():
    from _core.api.schemas import GoogleDocIngest, GoogleDocStructuredIngest

    chunk_schema = GoogleDocIngest.model_json_schema()
    structured_schema = GoogleDocStructuredIngest.model_json_schema()

    chunk_examples = chunk_schema["examples"]
    structured_examples = structured_schema["examples"]

    assert chunk_examples[0]["document_url"] == CHUNK_DOCUMENT_URL
    assert structured_examples[0]["document_url"] == FAQ_DOCUMENT_URL
    assert chunk_examples[0]["document_url"] != structured_examples[0]["document_url"]


def test_ingest_example_google_doc(client, monkeypatch):
    monkeypatch.setattr(
        documents_route,
        "get_doc_sections",
        lambda document_id: (FAQ_TITLE, [{"text": FAQ_TEXT}]),
    )

    captured = {}

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {
            "document_id": kwargs["document_id"],
            "filename": kwargs["filename"],
            "summary": kwargs["summary"],
            "chunks": 1,
            "chunk_chars": kwargs["chunk_chars"] or 3200,
            "has_embedding": True,
        }

    monkeypatch.setattr(documents_route, "upsert_document", fake_upsert)

    response = client.post(
        "/documents/google-doc",
        json={
            "document_url": FAQ_DOCUMENT_URL,
            "summary": "FAQ covering shipping, returns, and customer support",
            "chunk_chars": 1300,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == FAQ_DOCUMENT_ID
    assert body["filename"] == FAQ_TITLE
    assert body["chunk_chars"] == 1300
    assert body["has_embedding"] is True
    assert captured["document_id"] == FAQ_DOCUMENT_ID
    assert captured["content"] == FAQ_TEXT
    assert captured["chunk_chars"] == 1300
    assert captured["sections"] == [{"text": FAQ_TEXT}]


def test_ingest_google_doc_treats_blank_chunk_chars_as_omitted(client, monkeypatch):
    monkeypatch.setattr(
        documents_route,
        "get_doc_sections",
        lambda document_id: (FAQ_TITLE, [{"text": FAQ_TEXT}]),
    )

    captured = {}

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {
            "document_id": kwargs["document_id"],
            "filename": kwargs["filename"],
            "summary": kwargs["summary"],
            "chunks": 1,
            "chunk_chars": 3200,
            "has_embedding": True,
        }

    monkeypatch.setattr(documents_route, "upsert_document", fake_upsert)

    response = client.post(
        "/documents/google-doc",
        json={
            "document_url": FAQ_DOCUMENT_URL,
            "summary": "",
            "chunk_chars": "",
        },
    )

    assert response.status_code == 200
    assert captured["chunk_chars"] is None
    assert captured["summary"] is None


def test_ingest_google_doc_rejects_non_docs_url(client):
    response = client.post(
        "/documents/google-doc",
        json={"document_url": "https://example.com/not-a-doc"},
    )
    assert response.status_code == 400
    assert "Google Doc URL" in response.json()["detail"]


def test_ingest_google_doc_not_found(client, monkeypatch):
    resp = Mock()
    resp.status = 404
    monkeypatch.setattr(
        documents_route,
        "get_doc_sections",
        Mock(side_effect=HttpError(resp=resp, content=b"not found")),
    )

    response = client.post(
        "/documents/google-doc",
        json={"document_url": FAQ_DOCUMENT_URL},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Google Doc not found"


def test_ingest_google_doc_empty_text(client, monkeypatch):
    monkeypatch.setattr(
        documents_route,
        "get_doc_sections",
        lambda document_id: (FAQ_TITLE, [{"text": "   "}]),
    )

    response = client.post(
        "/documents/google-doc",
        json={"document_url": FAQ_DOCUMENT_URL},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Google Doc has no text to embed"


def test_ingest_structured_google_doc(client, monkeypatch):
    monkeypatch.setattr(
        documents_route,
        "get_doc_sections",
        lambda document_id: (FAQ_TITLE, [{"text": FAQ_STRUCTURED_TEXT}]),
    )

    captured = {}

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {
            "document_id": kwargs["document_id"],
            "filename": kwargs["filename"],
            "summary": kwargs.get("summary"),
            "chunks": 3,
            "has_embedding": True,
        }

    monkeypatch.setattr(
        documents_route, "upsert_documents_structured", fake_upsert
    )

    response = client.post(
        "/documents/google-doc/structured",
        json={
            "document_url": FAQ_DOCUMENT_URL,
            "summary_tag": "h1",
            "question_tag": "h2",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == FAQ_DOCUMENT_ID
    assert body["filename"] == FAQ_TITLE
    assert body["chunks"] == 3
    assert captured["document_id"] == FAQ_DOCUMENT_ID
    assert captured["content"] == FAQ_STRUCTURED_TEXT
    assert captured["summary_tag"] == "h1"
    assert captured["question_tag"] == "h2"
    assert captured["summary"] is None
    assert captured["sections"] == [{"text": FAQ_STRUCTURED_TEXT}]


def test_ingest_structured_google_doc_keeps_caller_summary(client, monkeypatch):
    monkeypatch.setattr(
        documents_route,
        "get_doc_sections",
        lambda document_id: (FAQ_TITLE, [{"text": FAQ_STRUCTURED_TEXT}]),
    )

    captured = {}

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {
            "document_id": kwargs["document_id"],
            "filename": kwargs["filename"],
            "summary": kwargs["summary"],
            "chunks": 3,
            "has_embedding": True,
        }

    monkeypatch.setattr(
        documents_route, "upsert_documents_structured", fake_upsert
    )

    response = client.post(
        "/documents/google-doc/structured",
        json={
            "document_url": FAQ_DOCUMENT_URL,
            "summary_tag": "h1",
            "question_tag": "h2",
            "summary": "Caller-provided summary",
        },
    )

    assert response.status_code == 200
    assert captured["summary"] == "Caller-provided summary"


def test_ingest_structured_google_doc_rejects_non_docs_url(client):
    response = client.post(
        "/documents/google-doc/structured",
        json={
            "document_url": "https://example.com/not-a-doc",
            "summary_tag": "h1",
            "question_tag": "h2",
        },
    )
    assert response.status_code == 400
    assert "Google Doc URL" in response.json()["detail"]


def test_parse_structured_document_uses_h1_body_as_summary():
    summary, chunks = parse_structured_document(
        FAQ_STRUCTURED_TEXT,
        summary_tag="h1",
        question_tag="h2",
    )

    assert summary == "FAQ covering shipping, returns, and customer support."
    assert chunks == [
        (
            "## How do I contact customer service?\n"
            "Chat with us here, email support@jonas-demo.com, or call."
        ),
        "## Do you accept credit cards\nYes",
        "## Are you opened on holidays\nLoremp ipsum",
    ]


def test_upsert_documents_structured_embeds_h2_chunks_and_h1_summary(monkeypatch):
    captured = {}

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {
            "document_id": kwargs["document_id"],
            "filename": kwargs["filename"],
            "summary": kwargs["summary"],
            "chunks": len(kwargs["chunks"]),
            "has_embedding": True,
        }

    monkeypatch.setattr(
        "_core.ingest.documents.upsert_document",
        fake_upsert,
    )

    result = upsert_documents_structured(
        filename=FAQ_TITLE,
        content=FAQ_STRUCTURED_TEXT,
        summary_tag="h1",
        question_tag="h2",
        document_id=FAQ_DOCUMENT_ID,
    )

    assert captured["document_id"] == FAQ_DOCUMENT_ID
    assert captured["filename"] == FAQ_TITLE
    assert captured["content"] == FAQ_STRUCTURED_TEXT
    assert captured["summary"] == (
        "FAQ covering shipping, returns, and customer support."
    )
    assert captured["chunks"] == [
        (
            "## How do I contact customer service?\n"
            "Chat with us here, email support@jonas-demo.com, or call."
        ),
        "## Do you accept credit cards\nYes",
        "## Are you opened on holidays\nLoremp ipsum",
    ]
    assert result["chunks"] == 3
    assert result["summary"] == captured["summary"]


def test_upsert_documents_structured_keeps_caller_summary(monkeypatch):
    captured = {}

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {
            "document_id": kwargs.get("document_id"),
            "filename": kwargs["filename"],
            "summary": kwargs["summary"],
            "chunks": len(kwargs["chunks"]),
            "has_embedding": True,
        }

    monkeypatch.setattr(
        "_core.ingest.documents.upsert_document",
        fake_upsert,
    )

    upsert_documents_structured(
        filename=FAQ_TITLE,
        content=FAQ_STRUCTURED_TEXT,
        summary_tag="h1",
        question_tag="h2",
        summary="Caller-provided summary",
    )

    assert captured["summary"] == "Caller-provided summary"


def test_upsert_documents_structured_requires_question_chunks():
    with pytest.raises(ValueError, match="no question"):
        upsert_documents_structured(
            filename=FAQ_TITLE,
            content="# Frequently Asked Questions\nOnly a summary, no questions.\n",
            summary_tag="h1",
            question_tag="h2",
        )


def test_google_doc_body_emits_markdown_headings():
    from _core.integrations.google_docs import doc_body_to_text

    body = {
        "content": [
            {
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "HEADING_1"},
                    "elements": [
                        {"textRun": {"content": "Frequently Asked Questions\n"}}
                    ],
                }
            },
            {
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                    "elements": [
                        {
                            "textRun": {
                                "content": (
                                    "FAQ covering shipping, returns, "
                                    "and customer support.\n"
                                )
                            }
                        }
                    ],
                }
            },
            {
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "HEADING_2"},
                    "elements": [
                        {"textRun": {"content": "Do you accept credit cards\n"}}
                    ],
                }
            },
            {
                "paragraph": {
                    "paragraphStyle": {"namedStyleType": "NORMAL_TEXT"},
                    "elements": [{"textRun": {"content": "Yes\n"}}],
                }
            },
        ]
    }

    text = doc_body_to_text(body)
    summary, chunks = parse_structured_document(
        text, summary_tag="h1", question_tag="h2"
    )
    assert summary == "FAQ covering shipping, returns, and customer support."
    assert chunks == ["## Do you accept credit cards\nYes"]


def test_credentials_resolve_relative_path_from_project_root(monkeypatch, tmp_path):
    from _core.config import PROJECT_ROOT
    from _core.integrations import google_docs as gdocs

    captured = {}

    def fake_from_file(path, scopes=None):
        captured["path"] = path
        captured["scopes"] = scopes
        return Mock()

    monkeypatch.setattr(
        gdocs.service_account.Credentials,
        "from_service_account_file",
        fake_from_file,
    )
    monkeypatch.chdir(tmp_path)

    gdocs._credentials("secrets/google_service_account.json")

    assert captured["path"] == str(
        PROJECT_ROOT / "secrets" / "google_service_account.json"
    )


def test_credentials_keep_absolute_path(monkeypatch, tmp_path):
    from _core.integrations import google_docs as gdocs

    captured = {}
    absolute = str(tmp_path / "custom.json")

    monkeypatch.setattr(
        gdocs.service_account.Credentials,
        "from_service_account_file",
        lambda path, scopes=None: captured.update(path=path) or Mock(),
    )

    gdocs._credentials(absolute)

    assert captured["path"] == absolute


def _tab_body(paragraphs: list[str]) -> dict:
    return {
        "body": {
            "content": [
                {
                    "paragraph": {
                        "elements": [{"textRun": {"content": text}}],
                    }
                }
                for text in paragraphs
            ]
        }
    }


def test_document_to_text_reads_all_tabs_including_nested():
    from _core.integrations.google_docs import document_to_text

    doc = {
        "title": "Multi-tab contract",
        "body": {"content": []},
        "tabs": [
            {
                "tabProperties": {"title": "Cover"},
                "documentTab": _tab_body(["Cover page\n"]),
                "childTabs": [
                    {
                        "tabProperties": {"title": "Terms"},
                        "documentTab": _tab_body(["## Terms\n", "Pay within 30 days.\n"]),
                        "childTabs": [],
                    }
                ],
            },
            {
                "tabProperties": {"title": "Annex"},
                "documentTab": _tab_body(["Annex A\n"]),
                "childTabs": [],
            },
        ],
    }

    text = document_to_text(doc)
    assert "Cover page" in text
    assert "Pay within 30 days." in text
    assert "Annex A" in text


def test_document_to_text_falls_back_to_body_without_tabs():
    from _core.integrations.google_docs import document_to_text

    doc = {
        "title": "Legacy",
        "body": {
            "content": [
                {
                    "paragraph": {
                        "paragraphStyle": {"namedStyleType": "HEADING_1"},
                        "elements": [{"textRun": {"content": "Only body\n"}}],
                    }
                }
            ]
        },
    }

    assert document_to_text(doc) == "# Only body\n"


def test_get_doc_requests_all_tabs_content(monkeypatch):
    from _core.integrations import google_docs as gdocs

    captured = {}

    class FakeDocuments:
        def get(self, **kwargs):
            captured.update(kwargs)

            class Execute:
                def execute(self_inner):
                    return {
                        "title": "Tabbed",
                        "tabs": [
                            {
                                "documentTab": _tab_body(["First tab\n"]),
                                "childTabs": [
                                    {
                                        "documentTab": _tab_body(["Second tab\n"]),
                                        "childTabs": [],
                                    }
                                ],
                            }
                        ],
                    }

            return Execute()

    class FakeService:
        def documents(self):
            return FakeDocuments()

    monkeypatch.setattr(gdocs, "_credentials", lambda creds_path=None: Mock())
    monkeypatch.setattr(gdocs, "build", lambda *args, **kwargs: FakeService())

    title, text = gdocs.get_doc("doc-123")

    assert captured["documentId"] == "doc-123"
    assert captured["includeTabsContent"] is True
    assert title == "Tabbed"
    assert "First tab" in text
    assert "Second tab" in text


def test_document_sections_include_tab_title_and_id():
    from _core.integrations.google_docs import document_sections

    doc = {
        "tabs": [
            {
                "tabProperties": {"title": "Cover", "tabId": "t.0"},
                "documentTab": _tab_body(["Cover page\n"]),
                "childTabs": [
                    {
                        "tabProperties": {"title": "Terms", "tabId": "t.1"},
                        "documentTab": _tab_body(["## Payment\n", "Net 30.\n"]),
                        "childTabs": [],
                    }
                ],
            }
        ]
    }

    sections = document_sections(doc)
    assert sections == [
        {"text": "Cover page\n", "tab": "Cover", "tab_id": "t.0"},
        {"text": "## Payment\nNet 30.\n", "tab": "Terms", "tab_id": "t.1"},
    ]


def test_google_doc_url_includes_tab_when_present():
    from _core.integrations.google_docs import google_doc_url

    assert (
        google_doc_url("abc123")
        == "https://docs.google.com/document/d/abc123/edit"
    )
    assert (
        google_doc_url("abc123", tab_id="t.0")
        == "https://docs.google.com/document/d/abc123/edit?tab=t.0"
    )


def test_chunk_records_from_sections_attach_tab_and_heading():
    from _core.ingest.documents import chunk_records_from_sections

    records = chunk_records_from_sections(
        [
            {
                "text": "# Summary\nPolicy overview.\n\n## Returns\n30 days.\n",
                "tab": "FAQ",
                "tab_id": "t.0",
            },
            {
                "text": "Plain annex text without a heading.\n",
                "tab": "Annex",
                "tab_id": "t.1",
            },
        ],
        chunk_chars=5000,
    )

    assert len(records) == 2
    assert records[0]["metadata"]["tab"] == "FAQ"
    assert records[0]["metadata"]["tab_id"] == "t.0"
    assert records[0]["metadata"]["heading"] == "Summary"
    assert records[1]["metadata"] == {"tab": "Annex", "tab_id": "t.1"}
    assert "heading" not in records[1]["metadata"]


def test_upsert_document_inserts_chunk_metadata(monkeypatch):
    import json
    from types import SimpleNamespace

    from _core.ingest import documents as docs_mod

    inserted = []

    class FakeResult:
        def scalar_one_or_none(self):
            return None

        def scalar_one(self):
            return "summary"

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, params=None):
            sql = statement.text if hasattr(statement, "text") else str(statement)
            if "INSERT INTO document_embeddings" in sql:
                for row in params:
                    inserted.append(row)
            return FakeResult()

        def commit(self):
            pass

    monkeypatch.setattr(
        docs_mod,
        "get_provider",
        lambda: SimpleNamespace(
            embed=lambda texts: [SimpleNamespace(tolist=lambda: [0.1, 0.2])] * len(texts),
            model_name="test-model",
        ),
    )
    monkeypatch.setattr(docs_mod, "Session", lambda _engine: FakeSession())
    monkeypatch.setattr(docs_mod, "ensure_document_embeddings_metadata", lambda: None)

    docs_mod.upsert_document(
        filename="Contract",
        content="ignored when sections provided",
        document_id="gdoc-1",
        summary="summary",
        sections=[
            {"text": "## Clause 1\nPay now.\n", "tab": "Terms", "tab_id": "t.0"},
        ],
        chunk_chars=5000,
    )

    assert len(inserted) == 1
    meta = inserted[0]["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert meta["tab"] == "Terms"
    assert meta["tab_id"] == "t.0"
    assert meta["heading"] == "Clause 1"


def test_search_documents_returns_metadata_and_source_url(monkeypatch):
    from types import SimpleNamespace

    from _core.retrieval import documents as retrieval

    class FakeResult:
        def all(self):
            return [
                SimpleNamespace(
                    filename="Contract",
                    document_id="gdoc-1",
                    chunk_index=0,
                    content="Pay within 30 days.",
                    metadata={"tab": "Terms", "tab_id": "t.0", "heading": "Payment"},
                )
            ]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, params=None):
            return FakeResult()

    monkeypatch.setattr(
        retrieval,
        "get_provider",
        lambda: SimpleNamespace(
            embed=lambda texts: [SimpleNamespace(tolist=lambda: [0.1])]
        ),
    )
    monkeypatch.setattr(retrieval, "Session", lambda _engine: FakeSession())
    monkeypatch.setattr(
        retrieval, "ensure_document_embeddings_metadata", lambda: None
    )

    results = retrieval.search_documents("payment terms")
    assert results[0]["metadata"]["tab"] == "Terms"
    assert results[0]["metadata"]["heading"] == "Payment"
    assert (
        results[0]["source_url"]
        == "https://docs.google.com/document/d/gdoc-1/edit?tab=t.0"
    )


def test_agent_instructions_require_source_link():
    from pathlib import Path

    text = (
        Path(__file__).resolve().parent.parent
        / "_core"
        / "agent"
        / "instructions.md"
    ).read_text(encoding="utf-8")
    assert "Source:" in text
    assert "source_url" in text


def test_init_db_document_embeddings_include_metadata_jsonb(monkeypatch):
    from types import SimpleNamespace

    from _core.ingest import schema as mod

    executed = []

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement):
            executed.append(
                statement.text if hasattr(statement, "text") else str(statement)
            )
            return FakeResult()

        def commit(self):
            pass

    monkeypatch.setattr(mod, "get_provider", lambda: SimpleNamespace(embedding_dim=768))
    monkeypatch.setattr(mod, "Session", lambda _engine: FakeSession())
    monkeypatch.setattr(mod, "ensure_memory_tables", lambda session: None)
    monkeypatch.setattr(mod, "ensure_monitoring_tables", lambda session: None)

    mod.init_db()

    doc_emb = next(sql for sql in executed if "document_embeddings" in sql and "CREATE TABLE" in sql)
    assert "metadata" in doc_emb.lower()
    assert "jsonb" in doc_emb.lower()