"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from _core.api.routes import ask, documents, feedback, health, metrics, products, sessions
from _core.config import PROJECT_ROOT

STATIC_DIR = PROJECT_ROOT / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    from _core.agent.tracing import flush_tracing

    flush_tracing()


def create_app() -> FastAPI:
    app = FastAPI(title="Ecommerce Agent API", lifespan=lifespan)
    app.include_router(ask.router)
    app.include_router(feedback.router)
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(products.router)
    app.include_router(documents.router)
    app.include_router(sessions.router)

    @app.middleware("http")
    async def disable_docs_cache(request, call_next):
        response = await call_next(request)
        if request.url.path in {"/docs", "/redoc", "/openapi.json"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def ui():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/ecommerce")
    def ecommerce_catalog():
        return FileResponse(STATIC_DIR / "ecommerce.html")

    @app.get("/admin")
    def admin_ui():
        return FileResponse(STATIC_DIR / "admin.html")

    return app


app = create_app()
