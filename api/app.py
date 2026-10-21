"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import admin, chat, session
from config.settings import Settings
from core.bootstrap import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. The container is wired in the lifespan handler."""
    settings = settings or Settings.load()
    configure_logging(settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from api.container import AppContainer
        container = AppContainer(settings)
        app.state.container = container
        yield
        container.shutdown()

    app = FastAPI(title="Portfolio Assistant API", version="1.0.0", lifespan=lifespan)
    app.include_router(chat.router)
    app.include_router(session.router)
    app.include_router(admin.router)

    @app.get("/health", tags=["health"])
    def health_check():
        return {"status": "ok"}

    return app