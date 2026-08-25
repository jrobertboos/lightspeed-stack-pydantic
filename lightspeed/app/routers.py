"""REST API routers."""

from __future__ import annotations

from fastapi import FastAPI

from lightspeed.app.endpoints import health, query, root


def include_routers(app: FastAPI) -> None:
    """Include FastAPI routers for the endpoints implemented so far.

    Follows the original Lightspeed Stack layout: ``root`` and ``health`` are
    mounted without a version prefix, and versioned endpoints go under
    ``/v1``. Endpoints not yet implemented in this rewrite (info, tools,
    conversations, MCP, etc.) will be added here as they land.

    Parameters:
        app: The FastAPI application to attach routers to.
    """
    app.include_router(root.router)
    app.include_router(health.router)

    app.include_router(query.router, prefix="/v1")
