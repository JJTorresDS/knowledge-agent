import importlib
import logging

from fastapi.testclient import TestClient

app_mod = importlib.import_module("_core.api.app")
HTTP_LOGGER = "uvicorn.error"


def test_http_middleware_logs_method_path_and_status(client, caplog):
    with caplog.at_level(logging.INFO, logger=HTTP_LOGGER):
        response = client.get("/health")

    assert response.status_code == 200
    assert "[http] GET /health 200" in caplog.text


def test_http_middleware_includes_query_string(client, caplog):
    with caplog.at_level(logging.INFO, logger=HTTP_LOGGER):
        response = client.get("/?session_id=user-alice")

    assert response.status_code == 200
    assert "[http] GET /?session_id=user-alice 200" in caplog.text


def test_http_logging_announces_on_startup(caplog):
    with caplog.at_level(logging.INFO, logger=HTTP_LOGGER):
        with TestClient(app_mod.app):
            pass

    assert "[http] request logging enabled" in caplog.text


def test_http_logging_can_be_disabled(client, monkeypatch, caplog):
    monkeypatch.setattr(app_mod, "LOG_HTTP_REQUESTS", False)

    with caplog.at_level(logging.INFO, logger=HTTP_LOGGER):
        response = client.get("/health")

    assert response.status_code == 200
    assert "[http] GET /health 200" not in caplog.text
