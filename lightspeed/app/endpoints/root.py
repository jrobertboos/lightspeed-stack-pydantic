"""Handler for the "/" endpoint.

Minimal rewrite of the original Lightspeed Stack root endpoint. It returns a
small JSON banner instead of the original's static HTML index page; auth and
tracing are not yet implemented.
"""

from __future__ import annotations

from fastapi import APIRouter

from lightspeed.app.models.responses.success.root import RootResponse
from lightspeed.core.config import configuration

router = APIRouter(tags=["root"])


@router.get("/", summary="Root Endpoint Handler")
async def root_endpoint_handler() -> RootResponse:
    """Return a small banner with the service name and a link to the API docs."""
    return RootResponse(name=configuration.configuration.name)
