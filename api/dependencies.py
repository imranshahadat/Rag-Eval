"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Request

from api.container import AppContainer


def get_container(request: Request) -> AppContainer:
    """Access the wired application container stored on app.state."""
    return request.app.state.container
