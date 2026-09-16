# Application architecture

As-built. This repository is a **knowledge agent** API for internal staff and external customers: document ingest/search (`POST /documents/...`, retrieval tools) and agent endpoints (`POST /ask`, `GET /models`, sessions, feedback, `GET /metrics`). A mock ecommerce catalog (`POST /products/upload`, `search_products`, `/ecommerce`) shows the same pattern for an external-facing client. Postgres, Prometheus, Grafana, MLflow, pgAdmin, and Ollama are **external**. `docker-compose.yml` runs only this app.

Runtime Python is the `_core` package. Root shims (`app.py`, `agent.py`, `tools.py`, `vector_store.py`, `google_doc_reader.py`, `embeddings/`, `init/`) are gone.

```bash
make run_app
```

Same as `uv run uvicorn _core.api.app:app --reload --reload-include .env`. Seed on the host: `make seed`. Docker is optional (`make docker-up`) and still starts **only the app**. If Compose is publishing 8000, `http://localhost:8000` can hit the container (IPv6) instead of the host process — use `http://127.0.0.1:8000` or `make docker-down`.

## Layout

```text
ecommerce-agent/
├── _core/
│   ├── config.py                 # env: DB, LLM, embeddings, Google, tracing
│   ├── db.py                     # one SQLAlchemy engine
│   ├── api/
│   │   ├── app.py                # FastAPI factory
│   │   ├── schemas.py
│   │   └── routes/               # ask (+ GET /models), feedback, products, documents, sessions, health, metrics
│   ├── agent/                    # llm, tracing (Langfuse), memory.py, conversations.py, instructions.md, instructions_v1.md, hooks, factory
│   ├── monitoring/               # Prometheus metrics + Postgres ask_turns / feedback
│   ├── tools/                    # catalog.py, knowledge.py
│   ├── retrieval/                # read-only products + documents
│   ├── ingest/                   # chunking, product/document writes, schema
│   ├── embeddings/               # lazy HF | Gemini | OpenAI | Mistral
│   ├── integrations/google_docs.py
│   └── jobs/sync_google_docs.py
├── static/                       # chat, admin inbox, catalog HTML
├── db/                           # schema.md, init_vector_db.sql, seed, inspect.sql, pgadmin/servers.json, download_model
├── notebooks/
├── assets/                       # screenshots for README and evals/evaluation.md
├── evals/                        # datasets, eval scripts, evaluation.md runbook
├── llm-api-tests/                # live chat/embedding API pings (not in uv run pytest)
├── tests/
├── Dockerfile                    # app + evals image (Python 3.12, uv)
├── docker-compose.yml            # app only
├── .env.example                  # secrets template (copy to .env)
├── Makefile                      # make run_app, seed first; docker-up / docker-seed optional
├── AGENTS.md                     # TDD + keep README and architecture.md current
├── todo.md                       # scratch backlog
└── secrets/                      # gitignored service account
```

## Run (host first)

Develop on the host with `make run_app` and `make seed`. `config.py` loads `{project}/.env` at import. Uvicorn `--reload --reload-include .env` restarts the worker on Python or `.env` changes. Shell-exported `POSTGRES_*` still win over `.env` (`load_dotenv` does not override).

## Docker Compose

`docker-compose.yml` runs only **app** (this Dockerfile, port 8000) for deploy or a throwaway image, not day-to-day API edits. Postgres, pgAdmin, MLflow, Prometheus, Grafana, and Ollama are not Compose services. `POSTGRES_*` and `MLFLOW_TRACKING_URI` come from `.env` (`env_file`). The app container adds `host.docker.internal:host-gateway` so it can reach Postgres (or Ollama) published on the host. Google credentials are mounted from `./secrets`. The chat/catalog HTML is bind-mounted from `./static` and the Python package from `./_core`. The image Uvicorn **does not reload**; recreate after Python or `.env` changes (`docker compose up -d --force-recreate app`); `/docs` and `/openapi.json` send `Cache-Control: no-store`. The image is Python 3.12; `pyproject.toml` sets `requires-python = ">=3.12,<3.14"` so uv does not try to resolve Torch for 3.14/Windows. The app starts with `uv run --frozen --no-dev` so container start uses `uv.lock` and does not re-resolve. Schema is created by `make seed` or `make docker-seed` (`db/seed_products.py` → `init_db()`), not `db/init_vector_db.sql`. `init_db()` also ensures conversation tables `agent_sessions` and `agent_messages` and production tables `ask_turns` and `conversation_feedback` with `CREATE IF NOT EXISTS` when the catalog already exists. If `conversation_feedback.rating` is still text (`'up'` / `'down'`), monitoring drops that table and recreates it with integer `1` / `-1` (no `ALTER`). `db/inspect.sql` lists tables and row counts for your own Postgres client. `db/pgadmin/servers.json` is a sample pgAdmin server pointing at `localhost`.

This stack does not run Postgres, pgAdmin, MLflow, Prometheus, or Grafana. Evals log to whatever `MLFLOW_TRACKING_URI` points at (default `http://127.0.0.1:5000`). Production latency, word counts, and thumbs feedback are recorded in Postgres and exported as Prometheus series on `GET /metrics` for an external scraper. **Langfuse** remains the cloud trace UI for tool calls and generations.

```mermaid
flowchart LR
    subgraph Compose["docker compose"]
        App["app :8000"]
    end
    subgraph External["external"]
        PG["Postgres pgvector"]
        MF["MLflow"]
        Prom["Prometheus"]
        GF["Grafana"]
    end
    App --> PG
    App -->|"eval logs"| MF
    Prom -->|"GET /metrics"| App
    GF --> Prom
    GF --> PG
```

## System overview

Three paths share Postgres. The agent is a knowledge worker over ingested documents; mock product search is an extra path for an external-facing demo. Tool names, ingest endpoints, and sequence detail are in **Ask flow** and **Ingest and sync** below. `config.settings` is omitted here (see **Layer rules**).

```mermaid
flowchart TB
    subgraph paths [Runtime]
        direction LR

        subgraph askPath [Ask]
            direction TB
            Chat["Chat UI / Admin"]
            Ask["POST /ask"]
            Models["GET /models"]
            Sessions["GET /sessions"]
            AdminList["GET /admin/conversations"]
            Agent["agent: factory, LLM,<br/>memory, conversations, Langfuse"]
            Tools["tools → retrieval"]
            Chat --> Ask --> Agent --> Tools
            Chat --> Models
            Chat --> Sessions --> Agent
            Chat --> AdminList --> Agent
        end

        subgraph ingestPath [Ingest]
            direction TB
            Src["Catalog UI / OpenAPI"]
            IngEP["POST /products<br/>POST /documents"]
            Ing["ingest + chunking"]
            Job["sync_google_docs"]
            GDocs["Google Docs / Drive"]
            Src --> IngEP --> Ing
            Job --> GDocs --> Ing
        end

        subgraph opsPath [Metrics]
            direction TB
            FB["POST /feedback"]
            Met["GET /metrics"]
            Mon["monitoring"]
            Obs["external Prometheus / Grafana"]
            FB --> Mon --> Met
            Met --> Obs
        end
    end

    Emb["embeddings: HF | Gemini | OpenAI | Mistral"]
    PG[(PostgreSQL + pgvector)]

    Ask --> Mon
    Tools --> Emb
    Ing --> Emb
    Agent --> PG
    Tools --> PG
    Ing --> PG
    Mon --> PG
    Obs --> PG
```

## Layer rules

- **api** calls the agent factory, conversations helpers, or ingest. It does not run SQL or embedding math. HTTP middleware logs `[http] METHOD path status` on the Uvicorn `INFO` logger when `LOG_HTTP_REQUESTS` is true (plus a startup line).
- **tools** call retrieval only. Tools never ingest.
- **retrieval** is SELECT + cosine search.
- **ingest** is the only writer of embeddings. Catalog `init_db()` still refuses to recreate product/document tables that already exist, but it always ensures conversation memory tables and production monitoring tables.
- **agent.memory** is the writer of chat turns (`agent_sessions`, `agent_messages`). Tables are `CREATE IF NOT EXISTS` so existing catalogs keep working.
- **agent.conversations** lists sessions and hydrates chat-UI turns from `ask_turns` (falling back to `agent_messages`). Table `CREATE IF NOT EXISTS` runs once per process on the first read, not on every admin poll.
- **monitoring** records production `ask_turns` (latency, word counts) and `conversation_feedback` (thumbs: `rating` 1 or -1), and exposes Prometheus series on `GET /metrics`. Persist and Prometheus observe are best-effort: failures are logged and do not fail `POST /ask` or `POST /feedback`. A leftover text-rating `conversation_feedback` table is dropped and recreated as integer.
- **jobs** reuse ingest + integrations. Not a second write path.
- **config.py** is the only module that reads environment variables.

## Ask flow

For FAQ / support, the agent lists document summaries first, then searches with that `document_id`. It must not invent contact details or policies. `search_faq_knowledgebase` requires `document_id` unless exactly one document exists.

The chat UI sends a `session_id`: **Auto** uses a browser `sessionStorage` UUID, **Custom** uses a caller-chosen string (support user / ticket id). It also loads `GET /models` (the `_DEFAULT_CHAT_MODELS` names in `_core/config.py`) into a **Model** dropdown and sends the chosen `model` on `POST /ask`. Unknown names return 422; omit `model` to use `settings.model`. Picking a listed name builds that provider's client for the request (`build_agent(model=...)`) without changing the process default. `GET /admin` is a static HTML inbox; it `fetch`es `GET /admin/conversations?limit=&min_turns=` on load and every 3s while the tab is visible (same-origin JS, no CDN). Defaults: `limit=10` (newest first) and `min_turns=0` (any length); the UI can raise the limit and filter to conversations with `turn_count > N`. Opening a row uses `/?session_id=` on the same chat UI, which polls `GET /sessions/{session_id}` every 3s while the tab is visible. That is enough for a live support inbox; there is no WebSocket or channel layer. `POST /ask` passes `Runner.run(..., session=PostgresSession(session_id))` so prior turns are loaded from Postgres and new items are stored. Omit `session_id` for a single-turn call (evals do this); blank strings are treated as omitted. Each reply includes a `turn_id`; thumbs on the bubble `POST /feedback` with `rating` 1 (up) or -1 (down). Latency and word counts are recorded for `GET /metrics`. `GET /sessions/{session_id}` returns question/answer turns for that thread (from `ask_turns`, or SDK items if monitoring rows are missing).

```mermaid
sequenceDiagram
    actor User
    participant UI as Chat UI
    participant Admin as Admin UI
    participant Sessions as GET /sessions
    participant Models as GET /models
    participant Ask as POST /ask
    participant Memory as agent.memory
    participant Agent as agent.factory
    participant LLM as Ollama, OpenRouter, OpenAI, or Mistral
    participant Tools as tools
    participant Retrieval as retrieval
    participant DB as PostgreSQL

    Admin->>Sessions: GET /admin/conversations
    Sessions->>DB: agent_sessions + ask_turns
    Sessions-->>Admin: HTML list
    Note over Admin: poll every 3s
    Admin->>UI: /?session_id=
    UI->>Sessions: GET /sessions/{id}
    Sessions-->>UI: turns
    UI->>Models: GET /models
    Models-->>UI: default + model names
    User->>UI: question
    UI->>Ask: JSON + session_id + model
    Ask->>Memory: PostgresSession
    Memory->>DB: agent_messages for session_id
    Ask->>Agent: Runner.run session=

    loop until final answer
        Agent->>LLM: history + messages + tool schemas
        alt knowledge base
            LLM->>Tools: list_knowledgebase_documents
            Tools->>Retrieval: list_documents
            Retrieval->>DB: id, filename, summary
            LLM->>Tools: search_faq_knowledgebase query + document_id
            Tools->>Retrieval: search_documents
            Retrieval->>DB: embedding <=> query
        else catalog
            LLM->>Tools: search_products / get_item_details
            Tools->>Retrieval: search / get_by_sku
            Retrieval->>DB: product_embeddings
        else done
            LLM-->>Agent: final_output
        end
    end

    Agent-->>Memory: add_items
    Memory->>DB: agent_messages
    Agent-->>Ask: answer
    Ask->>DB: ask_turns latency + words
    Ask-->>UI: JSON + turn_id
    UI-->>User: bubble + thumbs
    User->>UI: thumbs up or down
    UI->>Ask: POST /feedback
```

## Ingest and sync

Google Doc **id** is `documents.id`. The Doc **title** is `filename`. Optional `summary` is what the LLM reads before searching.

`POST /documents/google-doc` and the sync job call `upsert_document`, which splits on `chunk_chars` (omit for OpenAI File Search default: 3200 chars, 50% overlap; FAQ pages 1200–1400; contracts ~3200). OpenAPI examples for this route use the long-form sample doc in `_CHUNK_DOCUMENT_URL_EXAMPLE` (`_core/api/schemas.py`). `get_doc_sections` fetches with `includeTabsContent=true` and returns every tab (including nested `childTabs`); ingest chunks per tab and stores `tab` / `tab_id` / `heading` on `document_embeddings.metadata`.

`POST /documents/google-doc/structured` calls `upsert_documents_structured` with `summary_tag` / `question_tag` (`h1`, `h2`, …). Text under the first summary heading is stored on `documents.summary` unless the caller passes `summary`. Each question heading plus the text beneath it is one embedded chunk. `get_doc` turns Google Docs `HEADING_N` styles into ATX markdown (`#`, `##`) so those tags match. OpenAPI examples keep the FAQ sample in `_STRUCTURED_DOCUMENT_URL_EXAMPLE`.

`search_faq_knowledgebase` returns `metadata` and `source_url` (edit link, with `?tab=` when `tab_id` is present). Agent instructions require ending grounded answers with `Source: <source_url>`.

```mermaid
flowchart LR
    subgraph Catalog["Product catalog"]
        CSV["CSV sku, description"]
        Seed["db/seed_products.py"]
        UP["POST /products/upload"]
        Batch["ingest.products"]
        PE["product_embeddings"]
    end

    subgraph KB["Knowledge base"]
        URL["Google Doc URL"]
        GD["POST /documents/google-doc"]
        GDS["POST /documents/google-doc/structured"]
        GDocs["Docs API + Drive"]
        UD["ingest.documents"]
        DT["documents"]
        DE["document_embeddings"]
        Cron["python -m _core.jobs.sync_google_docs"]
    end

    CSV --> UP --> Batch
    Seed --> Batch
    Batch --> PE

    URL --> GD --> GDocs --> UD
    URL --> GDS --> GDocs
    Cron -->|"re-embed if Drive newer than updated_at / embedded_at"| GDocs
    UD --> DT
    UD --> DE
```

The sync job skips ids that start with `file_`. It does not `ALTER` tables.

## Evals

Runbook: `evals/evaluation.md` (includes screenshots under `assets/`). Scripts are not on the ask/ingest path.

`evals/datasets/faq_ground_truth.json` holds gold FAQ chunks. `evals/generate_eval_data.py` writes two synthetic shopper questions per FAQ to `evals/datasets/retrieval_eval_dataset.json` and `evals/datasets/llm_eval_dataset.json`.

`evals/evaluate_knowledge_search.py` scores `search_faq_knowledgebase` with hit@1, hit@k, MRR, and mean latency. `--search-type` is required (e.g. `genai_001_embedding`). `evals/evaluate_knowledge_search_mlflow.py` logs the same metrics plus params `search_type` and `embedding_model`; `--experiment` is required and reuses that MLflow experiment if it exists (otherwise creates it). Run names are `search-eval-{search_type}-{embedding_model}`. `make evaluate_retrieval SEARCH_TYPE=...` runs the terminal script. Local MLflow files (`mlruns/`, `mlartifacts/`, `mlflow.db`) are gitignored.

`evals/evaluate_llm_response.py` runs `build_agent()` (same tools and instructions as `POST /ask`) on those questions, one row at a time, then pauses 1 second after each prediction. `--provider` and `--experiment` are required; optional `--n` evaluates only the first n rows. The chat model is the provider default in `config.py`. After the agent is built, `provider` and `model` are read from it (`AgentLlmIdentity`) and logged as MLflow params, with mean `latency_ms` and token totals as metrics. Run names are `llm-eval-{provider}-{model}`. MLflow Correctness scores answers on the named experiment (created if missing). Default tracking URI is `http://127.0.0.1:5000`; override with `MLFLOW_TRACKING_URI`. This repo does not start an MLflow server. Needs ingested FAQ chunks in Postgres. `make evaluate_llms PROVIDER=... EXPERIMENT=...` runs this script (`N=...` passes `--n`). Experiment names: `ecommerce-agent-{kind}` (`llm_eval`, `search_eval`).

## LLM API smoke tests

`llm-api-tests/` pings each chat and embedding API with a one-token prompt. It is not on the ask/ingest path and is not collected by `uv run pytest`. `make llm_api_tests` runs the folder; a missing key (or unreachable Ollama) skips that test. Shared helpers live in `llm-api-tests/providers.py`.

## Data model and indexes

Column-level types and purpose: `db/schema.md`. The ER diagram below matches `init_db()` / memory / monitoring helpers.

New databases (`db/init_vector_db.sql` and `ingest.schema.init_db`) use **HNSW**. Existing databases that still have IVFFlat `lists = 100` keep working because retrieval sets `ivfflat.probes = 100` per query. No live `ALTER`. Conversation memory tables are additive (`CREATE TABLE IF NOT EXISTS`). `conversation_feedback` is dropped and recreated when `rating` is still text (`'up'` / `'down'`) so new rows store integer `1` / `-1`.

`embedding VECTOR(...)` width is fixed at `CREATE`. Python `init_db()` enables `CREATE EXTENSION IF NOT EXISTS vector`, then uses `provider.embedding_dim` (`hf` / bge-m3: 1024; `mistral` / mistral-embed-2312: 1024; `gemini`: 768; `openai` / text-embedding-3-small: 1536). `db/init_vector_db.sql` is hardcoded `VECTOR(1024)` for HF/Mistral-width. Gemini's API returns 3072-d vectors; `GeminiEmbeddingProvider` requests `dimensions=768` and, if the API still returns 3072, truncates and L2-normalizes (Matryoshka). Mistral embeddings use the OpenAI SDK against `MISTRAL_BASE_URL` and do not pass a `dimensions` override (native 1024-d). Switching providers after tables exist requires dropping `product_embeddings`, `document_embeddings`, and `documents`.

```mermaid
erDiagram
    product_embeddings {
        serial id PK
        text sku UK
        text name
        text price
        text description
        text content
        vector embedding
        text embedding_model
    }

    documents {
        text id PK
        text filename UK
        text content
        text summary
        boolean has_embedding
        timestamptz updated_at
        timestamptz embedded_at
    }

    document_embeddings {
        serial id PK
        text document_id FK
        int chunk_index
        text content
        vector embedding
        text embedding_model
    }

    documents ||--o{ document_embeddings : chunks

    agent_sessions {
        text session_id PK
        timestamptz created_at
        timestamptz updated_at
    }

    agent_messages {
        serial id PK
        text session_id FK
        jsonb message_data
        timestamptz created_at
    }

    agent_sessions ||--o{ agent_messages : turns

    ask_turns {
        text id PK
        text session_id
        text question
        text answer
        int question_words
        int answer_words
        float latency_ms
        timestamptz created_at
    }

    conversation_feedback {
        serial id PK
        text session_id
        text turn_id
        int rating
        timestamptz created_at
    }
```

## Config

`_core.config.settings` reads **secrets** from `.env`. Chat/embedding backends are module constants in `config.py` and are not overridden by the environment. `build_model()` in `_core/agent/llm.py` uses `OpenAIChatCompletionsModel` for Mistral (Chat Completions at `MISTRAL_BASE_URL`) and `OpenAIResponsesModel` for OpenAI, OpenRouter, and Ollama.

| Variable / constant | Role |
|---|---|
| `POSTGRES_*` | Database URL (`.env`). This repo does not start Postgres. Local `.env.example` default is `POSTGRES_HOST=localhost`. From the app container use `host.docker.internal` |
| `LLM_PROVIDER` | `config.py` constant: `ollama`, `openrouter`, `openai`, or `mistral` (currently `mistral`) |
| `LOCAL_MODEL` | `config.py` constant derived from `LLM_PROVIDER == "ollama"` |
| `EMBEDDING_PROVIDER` | `config.py` constant: `hf` (1024-d), `mistral` (1024-d), `gemini` (768-d), or `openai` (1536-d) (currently `mistral`) |
| `MODEL` | Optional `.env` chat-model override (`settings.model`). Defaults: Ollama `qwen2.5:7b`, OpenRouter `nvidia/nemotron-3.5-lightning:free`, OpenAI `gpt-4o-mini`, Mistral `mistral-small-latest`. Those names are `GET /models` and the chat UI picker. Fallbacks: `OLLAMA_MODEL` / `OPENROUTER_MODEL` / `OPENAI_MODEL` / `MISTRAL_MODEL` |
| `OPEN_ROUTER_API_KEY` / `OPENAI_API_KEY` / `MISTRAL_API_KEY` / `GEMINI_API_KEY` | Secrets in `.env`. Resolved into `settings.api_key` / `settings.embedding_api_key` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Secrets in `.env`. Tracing is on when `LANGFUSE_TRACING` is true in `config.py` and both keys are set (`settings.langfuse_enabled`). Agents SDK tracing stays enabled so OpenInference can export tool and generation spans under the `ask` observation |
| `LANGFUSE_BASE_URL` | Optional `.env` host (EU `https://cloud.langfuse.com`, US `https://us.cloud.langfuse.com`). Fallback constant `LANGFUSE_BASE_URL` in `config.py` |
| `LANGFUSE_ENVIRONMENT` | `config.py` constant (`development`) sent as `LANGFUSE_TRACING_ENVIRONMENT` |
| `EMBEDDING_MODEL` | Optional `.env` override (`settings.embedding_model`). Defaults in `DEFAULT_EMBEDDING_MODELS`: `BAAI/bge-m3`, `mistral-embed-2312`, `gemini-embedding-001`, `text-embedding-3-small`. Fallback: `OPENAI_EMBEDDING_MODEL`. Using another provider's default model, or constructing a backend that does not match `EMBEDDING_PROVIDER`, raises `ValueError` (`Provider model mismatch, please check your config.py file`) |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | Defaults to `secrets/google_service_account.json` at the project root. Relative `creds_path` values passed to `get_doc` / `get_doc_text` are also resolved from the project root |
| `AGENT_TRACING` | `true` enables OpenAI Agents SDK platform traces (separate from Langfuse) |
| `LOG_HTTP_REQUESTS` | `config.py` constant (`True`). HTTP middleware logs `[http] METHOD path?query status` on Uvicorn's `INFO` stream; startup also logs `[http] request logging enabled`. Set `False` to silence |
| `MLFLOW_TRACKING_URI` | Optional. Used by evals that log to MLflow. Defaults to `http://127.0.0.1:5000`. Not set by Compose; point it at your MLflow host |

Provider base URLs are module constants in `_core/config.py` (`OLLAMA_BASE_URL`, `OPENROUTER_BASE_URL`, `OPENAI_BASE_URL`, `MISTRAL_BASE_URL`, `GEMINI_OPENAI_BASE_URL`), each overridable by the same-named env var. They are not Settings fields.

The Hugging Face model loads on first `get_provider()` call, not at process import. `/health` does not embed.
