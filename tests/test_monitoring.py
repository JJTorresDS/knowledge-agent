from unittest.mock import AsyncMock, Mock

from ecommerce_agent.api.routes import ask as ask_route


def test_word_count_splits_on_whitespace():
    from ecommerce_agent.monitoring.metrics import word_count

    assert word_count("hello world") == 2
    assert word_count("  one   two three ") == 3
    assert word_count("") == 0


def test_metrics_endpoint_includes_ask_and_feedback_series(client, monkeypatch):
    result = Mock()
    result.final_output = "short answer"
    monkeypatch.setattr(ask_route.Runner, "run", AsyncMock(return_value=result))
    monkeypatch.setattr(
        ask_route,
        "record_ask_turn",
        lambda **kwargs: "turn-metrics",
    )

    ask_response = client.post("/ask", json={"question": "hello there"})
    assert ask_response.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    for name in (
        "ask_latency_seconds",
        "ask_question_words",
        "ask_answer_words",
        "ask_turns_total",
        "feedback_total",
    ):
        assert name in body


def test_record_ask_turn_survives_persist_failure(monkeypatch):
    from ecommerce_agent import monitoring as mon

    monkeypatch.setattr(
        mon,
        "persist_ask_turn",
        Mock(side_effect=OSError("postgres unavailable")),
    )
    observed = {}

    def fake_observe(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(mon, "observe_ask", fake_observe)

    turn_id = mon.record_ask_turn(
        session_id="sess-1",
        question="hello",
        answer="world",
        question_words=1,
        answer_words=1,
        latency_ms=12.5,
    )

    assert turn_id
    assert observed["latency_seconds"] == 0.0125
    assert observed["question_words"] == 1
    assert observed["answer_words"] == 1


def test_record_ask_turn_survives_prometheus_observe_failure(monkeypatch):
    from ecommerce_agent import monitoring as mon

    monkeypatch.setattr(mon, "persist_ask_turn", lambda **kwargs: "turn-ok")
    monkeypatch.setattr(
        mon,
        "observe_ask",
        Mock(side_effect=RuntimeError("prometheus client failed")),
    )

    turn_id = mon.record_ask_turn(
        session_id=None,
        question="hello",
        answer="world",
        question_words=1,
        answer_words=1,
        latency_ms=8.0,
    )

    assert turn_id == "turn-ok"


def test_record_feedback_survives_persist_and_observe_failures(monkeypatch):
    from ecommerce_agent import monitoring as mon

    monkeypatch.setattr(
        mon,
        "persist_feedback",
        Mock(side_effect=OSError("postgres unavailable")),
    )
    monkeypatch.setattr(
        mon,
        "observe_feedback",
        Mock(side_effect=RuntimeError("prometheus client failed")),
    )

    mon.record_feedback(session_id="sess-1", rating=1, turn_id="turn-1")


def test_ask_returns_answer_when_monitoring_fails(client, monkeypatch):
    result = Mock()
    result.final_output = "still answered"
    monkeypatch.setattr(ask_route.Runner, "run", AsyncMock(return_value=result))
    monkeypatch.setattr(
        ask_route,
        "record_ask_turn",
        Mock(side_effect=OSError("monitoring down")),
    )

    response = client.post("/ask", json={"question": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "still answered"
    assert body["turn_id"]


def test_feedback_ok_when_monitoring_fails(client, monkeypatch):
    from ecommerce_agent.api.routes import feedback as feedback_route

    monkeypatch.setattr(
        feedback_route,
        "record_feedback",
        Mock(side_effect=OSError("monitoring down")),
    )

    response = client.post(
        "/feedback",
        json={"session_id": "sess-1", "turn_id": "turn-1", "rating": 1},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
