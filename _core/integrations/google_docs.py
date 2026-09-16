"""Read Google Doc title/text and Drive modifiedTime via a service account."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build

from _core.config import PROJECT_ROOT, settings

SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
]
_GDOC_ID_RE = re.compile(r"/document/d/([a-zA-Z0-9_-]+)")
_HEADING_MARKDOWN = {
    "HEADING_1": "#",
    "HEADING_2": "##",
    "HEADING_3": "###",
    "HEADING_4": "####",
    "HEADING_5": "#####",
    "HEADING_6": "######",
}


def google_doc_id_from_url(url: str) -> str | None:
    match = _GDOC_ID_RE.search(url.strip())
    if match:
        return match.group(1)
    return None


def google_doc_url(document_id: str, tab_id: str | None = None) -> str:
    """Build a Google Docs edit URL, optionally deep-linking to a tab."""
    url = f"https://docs.google.com/document/d/{document_id}/edit"
    if tab_id:
        return f"{url}?tab={tab_id}"
    return url


def _credentials(creds_path: str | None = None):
    path = Path(creds_path or settings.google_service_account_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return service_account.Credentials.from_service_account_file(
        str(path), scopes=SCOPES
    )


def _paragraph_text(paragraph: dict) -> str:
    parts = []
    for run in paragraph.get("elements", []):
        text_run = run.get("textRun")
        if text_run and "content" in text_run:
            parts.append(text_run["content"])
    return "".join(parts)


def doc_body_to_text(body: dict) -> str:
    """Flatten a Google Docs `body` to text, with ATX headings for HEADING_N."""
    text_parts = []
    for element in body.get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        text = _paragraph_text(paragraph)
        if not text:
            continue
        style = paragraph.get("paragraphStyle", {}).get("namedStyleType")
        prefix = _HEADING_MARKDOWN.get(style)
        if prefix:
            newline = "\n" if text.endswith("\n") else ""
            text = f"{prefix} {text.rstrip('\n')}{newline}"
        text_parts.append(text)
    return "".join(text_parts)


def _flatten_tabs(tabs: list[dict] | None) -> list[dict]:
    """Depth-first flatten of top-level and nested `childTabs`."""
    flat: list[dict] = []

    def walk(tab: dict) -> None:
        flat.append(tab)
        for child in tab.get("childTabs") or []:
            walk(child)

    for tab in tabs or []:
        walk(tab)
    return flat


def document_sections(doc: dict) -> list[dict]:
    """Return per-tab text sections with ``tab`` / ``tab_id`` metadata.

    With ``includeTabsContent=true``, content lives under ``tabs``. Without
    tabs, return a single section from top-level ``body`` (no tab fields).
    """
    tabs = _flatten_tabs(doc.get("tabs"))
    if not tabs:
        text = doc_body_to_text(doc.get("body", {}))
        return [{"text": text}] if text.strip() else []

    sections: list[dict] = []
    for tab in tabs:
        props = tab.get("tabProperties") or {}
        document_tab = tab.get("documentTab") or {}
        body = document_tab.get("body") or {}
        text = doc_body_to_text(body)
        if not text.strip():
            continue
        section: dict = {"text": text}
        title = (props.get("title") or "").strip()
        tab_id = (props.get("tabId") or "").strip()
        if title:
            section["tab"] = title
        if tab_id:
            section["tab_id"] = tab_id
        sections.append(section)
    return sections


def document_to_text(doc: dict) -> str:
    """Flatten a Documents API resource to text, including every tab when present."""
    parts = [section["text"] for section in document_sections(doc) if section.get("text")]
    return "\n".join(parts)


def sections_to_text(sections: list[dict]) -> str:
    return "\n".join(
        section["text"] for section in sections if (section.get("text") or "").strip()
    )


def get_doc_sections(
    document_id: str, creds_path: str | None = None
) -> tuple[str, list[dict]]:
    """Fetch a Google Doc and return `(title, sections)` for all tabs."""
    credentials = _credentials(creds_path)
    service = build("docs", "v1", credentials=credentials)
    doc = (
        service.documents()
        .get(documentId=document_id, includeTabsContent=True)
        .execute()
    )
    title = (doc.get("title") or "").strip()
    return title, document_sections(doc)


def get_doc(document_id: str, creds_path: str | None = None) -> tuple[str, str]:
    """Fetch a Google Doc and return `(title, text)` with markdown headings.

    Requests all tab contents so multi-tab docs are fully ingested.
    """
    title, sections = get_doc_sections(document_id, creds_path)
    return title, sections_to_text(sections)


def get_doc_text(document_id: str, creds_path: str | None = None) -> str:
    """Fetch a Google Doc and return its text (ATX headings for HEADING_N)."""
    _, text = get_doc(document_id, creds_path)
    return text


def get_doc_text_from_json_string(document_id: str, key_json: str) -> str:
    info = json.loads(key_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=SCOPES
    )
    service = build("docs", "v1", credentials=credentials)
    doc = (
        service.documents()
        .get(documentId=document_id, includeTabsContent=True)
        .execute()
    )
    return document_to_text(doc)


def get_doc_modified_time(
    document_id: str, creds_path: str | None = None
) -> datetime:
    """Return Drive `modifiedTime` as an aware UTC datetime."""
    credentials = _credentials(creds_path)
    drive = build("drive", "v3", credentials=credentials)
    meta = (
        drive.files()
        .get(fileId=document_id, fields="modifiedTime")
        .execute()
    )
    raw = meta["modifiedTime"]
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a Google Doc's text content.")
    parser.add_argument("document_id", help="The Google Doc's document ID (from its URL)")
    parser.add_argument(
        "--creds",
        default=None,
        help="Path to the service account JSON key file",
    )
    args = parser.parse_args()

    try:
        text = get_doc_text(args.document_id, args.creds)
    except FileNotFoundError:
        path = args.creds or settings.google_service_account_file
        print(f"Error: credentials file not found at '{path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading document: {e}", file=sys.stderr)
        sys.exit(1)

    print(text)


if __name__ == "__main__":
    main()
