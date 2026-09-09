from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_runs_uvicorn_on_all_interfaces():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12" in text
    assert "uvicorn" in text
    assert "0.0.0.0" in text
    assert "8000" in text
    assert "uv run --frozen --no-dev" in text or '"--frozen"' in text


def test_pyproject_limits_python_below_3_14():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.14"' in text
    assert "https://pytorch.org" not in text


def test_compose_defines_only_the_app():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "app:" in text
    assert "8000:8000" in text
    assert "./static:/app/static" in text
    assert "./_core:/app/_core" in text
    assert "host.docker.internal:host-gateway" in text
    assert "POSTGRES_HOST: postgres" not in text


def test_compose_does_not_run_database_or_observability_infrastructure():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for needle in (
        "  postgres:",
        "  pgadmin:",
        "pgvector/pgvector",
        "dpage/pgadmin4",
        "pg_isready",
        "5050:80",
        "mlflow:",
        "grafana:",
        "prometheus:",
        "prom/prometheus",
        "grafana/grafana",
        "5000:5000",
        "3000:3000",
        "9090:9090",
        "MLFLOW_TRACKING_URI: http://mlflow:5000",
    ):
        assert needle not in text
    assert "container_name: postgres-pgvector" not in text
    assert not (ROOT / "grafana").exists()
    assert not (ROOT / "prometheus").exists()


def test_inspect_sql_lists_catalog_tables():
    text = (ROOT / "db" / "inspect.sql").read_text(encoding="utf-8")
    for table in (
        "product_embeddings",
        "documents",
        "document_embeddings",
        "agent_sessions",
        "agent_messages",
        "ask_turns",
        "conversation_feedback",
    ):
        assert table in text


def test_env_example_lists_required_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in (
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "OPENAI_API_KEY",
        "OPEN_ROUTER_API_KEY",
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "MLFLOW_TRACKING_URI",
    ):
        assert key in text
    assert "GRAFANA_ADMIN_USER" not in text
    assert "GRAFANA_ADMIN_PASSWORD" not in text
    assert "PGADMIN_DEFAULT_EMAIL" not in text
    assert "PGADMIN_DEFAULT_PASSWORD" not in text
    assert "POSTGRES_HOST=localhost" in text
    assert "POSTGRES_HOST=host.docker.internal" not in text


def test_gitignore_excludes_local_mlflow_and_os_junk():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "mlruns/",
        "mlartifacts/",
        "mlflow.db",
        "mlflow.db-journal",
        ".env",
        "/secrets/",
        ".DS_Store",
    ):
        assert pattern in text


def test_dockerignore_excludes_local_mlflow_and_secrets():
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (
        "mlruns/",
        "mlartifacts/",
        "mlflow.db",
        "mlflow.db-journal",
        ".env",
        "secrets/",
        ".DS_Store",
    ):
        assert pattern in text


def test_app_image_does_not_start_mlflow_server():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "mlflow server" not in text
    assert "uvicorn" in text


def test_makefile_does_not_start_postgres_or_pgadmin():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "docker-pgadmin" not in text
    assert "docker compose up -d pgadmin" not in text
    assert "docker compose exec" not in text


def test_makefile_local_dev_targets_reload_env_and_seed_on_the_host():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert text.index("run_app:") < text.index("docker-up:")
    run_block = text[text.index("run_app:") : text.index("llm_api_tests:")]
    assert "--reload" in run_block
    assert "--reload-include .env" in run_block
    assert "_core.api.app:app" in run_block
    assert "\nseed:\n" in text
    seed_block = text[text.index("\nseed:") :]
    assert "db/seed_products.py" in seed_block.split("docker-seed:")[0]
    assert "uv run python db/seed_products.py" in text
    assert "/_core:/app/_core" in text


def test_runtime_package_is_core():
    assert (ROOT / "_core" / "config.py").is_file()
    assert (ROOT / "_core" / "agent" / "factory.py").is_file()
    assert not (ROOT / "ecommerce_agent").exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["_core"]' in pyproject
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY _core ./_core" in dockerfile
    assert "_core.api.app:app" in dockerfile
