from datetime import datetime, timezone

from _core.agent import conversations as conversations_mod
from _core.api.routes import sessions as sessions_route


def test_list_sessions_returns_conversation_summaries(client, monkeypatch):
    monkeypatch.setattr(
        sessions_route,
        "list_conversations",
        lambda: [
            {
                "session_id": "user-alice",
                "created_at": datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 9, 9, 12, 30, tzinfo=timezone.utc),
                "turn_count": 2,
                "last_question": "Where is my order?",
            }
        ],
    )

    response = client.get("/sessions")

    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "session_id": "user-alice",
            "created_at": "2026-09-09T12:00:00Z",
            "updated_at": "2026-09-09T12:30:00Z",
            "turn_count": 2,
            "last_question": "Where is my order?",
        }
    ]


def test_get_session_returns_turns_for_chat_ui(client, monkeypatch):
    monkeypatch.setattr(
        sessions_route,
        "get_conversation",
        lambda session_id: {
            "session_id": session_id,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "question": "Do you ship to Brazil?",
                    "answer": "Yes, we ship worldwide.",
                }
            ],
        },
    )

    response = client.get("/sessions/user-alice")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "user-alice",
        "turns": [
            {
                "turn_id": "turn-1",
                "question": "Do you ship to Brazil?",
                "answer": "Yes, we ship worldwide.",
            }
        ],
    }


def test_admin_conversations_html_partial(client, monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return [
            {
                "session_id": "user-alice",
                "created_at": datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 9, 9, 12, 30, tzinfo=timezone.utc),
                "turn_count": 2,
                "last_question": "Where is my order?",
            }
        ]

    monkeypatch.setattr(sessions_route, "list_conversations", fake_list)

    response = client.get("/admin/conversations")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "user-alice" in body
    assert "Where is my order?" in body
    assert "/?session_id=user-alice" in body
    assert "2 turns" in body
    assert "<script" not in body
    assert captured == {"limit": 10, "min_turns": 0}


def test_admin_conversations_accepts_limit_and_min_turns(client, monkeypatch):
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(sessions_route, "list_conversations", fake_list)

    response = client.get("/admin/conversations?limit=25&min_turns=3")

    assert response.status_code == 200
    assert captured == {"limit": 25, "min_turns": 3}


def test_admin_conversations_rejects_invalid_limit(client):
    response = client.get("/admin/conversations?limit=0")
    assert response.status_code == 422


def test_admin_conversations_html_empty(client, monkeypatch):
    monkeypatch.setattr(
        sessions_route, "list_conversations", lambda **kwargs: []
    )

    response = client.get("/admin/conversations")

    assert response.status_code == 200
    assert "No conversations yet." in response.text


def test_get_unknown_session_returns_empty_turns(client, monkeypatch):
    monkeypatch.setattr(
        sessions_route,
        "get_conversation",
        lambda session_id: {"session_id": session_id, "turns": []},
    )

    response = client.get("/sessions/brand-new-user")

    assert response.status_code == 200
    assert response.json() == {"session_id": "brand-new-user", "turns": []}


def test_list_conversations_merges_sessions_and_ask_turns(monkeypatch):
    created = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)
    updated = datetime(2026, 9, 9, 11, 0, tzinfo=timezone.utc)
    later = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)

    class FakeResult:
        def __init__(self, rows=()):
            self._rows = list(rows)

        def all(self):
            return self._rows

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def commit(self):
            return None

        def execute(self, statement, params=None):
            sql = " ".join(statement.text.upper().split())
            if sql.startswith("CREATE TABLE") or sql.startswith("CREATE INDEX"):
                return FakeResult()
            if "FROM AGENT_SESSIONS" in sql:
                return FakeResult(
                    [
                        type(
                            "Row",
                            (),
                            {
                                "session_id": "browser-uuid",
                                "created_at": created,
                                "updated_at": updated,
                            },
                        )()
                    ]
                )
            if "FROM ASK_TURNS" in sql:
                return FakeResult(
                    [
                        type(
                            "Row",
                            (),
                            {
                                "id": "t1",
                                "session_id": "user-alice",
                                "question": "Hi",
                                "answer": "Hello",
                                "created_at": later,
                            },
                        )(),
                        type(
                            "Row",
                            (),
                            {
                                "id": "t2",
                                "session_id": "user-alice",
                                "question": "Where is my order?",
                                "answer": "It shipped.",
                                "created_at": later,
                            },
                        )(),
                    ]
                )
            return FakeResult()

    monkeypatch.setattr(conversations_mod, "ensure_memory_tables", lambda: None)
    monkeypatch.setattr(conversations_mod, "ensure_monitoring_tables", lambda: None)
    monkeypatch.setattr(conversations_mod, "Session", lambda *_args, **_kwargs: FakeSession())

    rows = conversations_mod.list_conversations()
    by_id = {row["session_id"]: row for row in rows}
    assert by_id["user-alice"]["turn_count"] == 2
    assert by_id["user-alice"]["last_question"] == "Where is my order?"
    assert by_id["browser-uuid"]["turn_count"] == 0
    assert rows[0]["session_id"] == "user-alice"


def test_filter_conversations_by_min_turns_and_limit():
    rows = [
        {
            "session_id": "a",
            "updated_at": datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
            "turn_count": 5,
        },
        {
            "session_id": "b",
            "updated_at": datetime(2026, 9, 9, 11, 0, tzinfo=timezone.utc),
            "turn_count": 2,
        },
        {
            "session_id": "c",
            "updated_at": datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc),
            "turn_count": 4,
        },
        {
            "session_id": "d",
            "updated_at": datetime(2026, 9, 9, 9, 0, tzinfo=timezone.utc),
            "turn_count": 1,
        },
    ]

    filtered = conversations_mod.filter_conversations(rows, limit=2, min_turns=2)
    assert [row["session_id"] for row in filtered] == ["a", "c"]


def test_get_conversation_prefers_ask_turns(monkeypatch):
    class FakeResult:
        def __init__(self, rows=()):
            self._rows = list(rows)

        def all(self):
            return self._rows

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, params=None):
            sql = " ".join(statement.text.upper().split())
            if sql.startswith("CREATE TABLE") or sql.startswith("CREATE INDEX"):
                return FakeResult()
            if "FROM ASK_TURNS" in sql:
                return FakeResult(
                    [
                        type(
                            "Row",
                            (),
                            {
                                "id": "turn-9",
                                "question": "Need a refund",
                                "answer": "I can help with that.",
                            },
                        )()
                    ]
                )
            return FakeResult()

    monkeypatch.setattr(conversations_mod, "ensure_memory_tables", lambda: None)
    monkeypatch.setattr(conversations_mod, "ensure_monitoring_tables", lambda: None)
    monkeypatch.setattr(conversations_mod, "Session", lambda *_args, **_kwargs: FakeSession())

    detail = conversations_mod.get_conversation("user-bob")
    assert detail == {
        "session_id": "user-bob",
        "turns": [
            {
                "turn_id": "turn-9",
                "question": "Need a refund",
                "answer": "I can help with that.",
            }
        ],
    }


def test_turns_from_sdk_items_pairs_user_and_assistant():
    turns = conversations_mod.turns_from_sdk_items(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {
                "type": "function_call",
                "name": "search_products",
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Size M?"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Yes."}],
            },
        ]
    )
    assert turns == [
        {"turn_id": None, "question": "Hello", "answer": "Hi there"},
        {"turn_id": None, "question": "Size M?", "answer": "Yes."},
    ]


def test_list_conversations_ensures_tables_once(monkeypatch):
    class FakeResult:
        def all(self):
            return []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, params=None):
            return FakeResult()

    counts = {"memory": 0, "monitoring": 0}
    monkeypatch.setattr(conversations_mod, "_read_tables_ready", False)
    monkeypatch.setattr(
        conversations_mod,
        "ensure_memory_tables",
        lambda: counts.__setitem__("memory", counts["memory"] + 1),
    )
    monkeypatch.setattr(
        conversations_mod,
        "ensure_monitoring_tables",
        lambda: counts.__setitem__("monitoring", counts["monitoring"] + 1),
    )
    monkeypatch.setattr(
        conversations_mod, "Session", lambda *_args, **_kwargs: FakeSession()
    )

    conversations_mod.list_conversations()
    conversations_mod.list_conversations()

    assert counts == {"memory": 1, "monitoring": 1}
