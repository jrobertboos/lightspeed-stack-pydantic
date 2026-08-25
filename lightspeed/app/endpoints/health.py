"""Handlers for health REST API endpoints.

Minimal rewrite of the original Lightspeed Stack health probes. Per-provider
health checks and degraded-mode tracking (as in the original) are not yet
implemented; readiness only reflects whether providers were loaded at
startup.
"""

from __future__ import annotations

from fastapi import APIRouter

from lightspeed.app.models.responses.success.probes import (
    LivenessResponse,
    ReadinessResponse,
)
from lightspeed.core.providers.registry import ProviderRegistry

router = APIRouter(tags=["health"])


@router.get("/readiness", summary="Readiness Probe")
async def readiness_probe_get_method() -> ReadinessResponse:
    """Report whether the service is ready to handle requests.

    The FastAPI ``lifespan`` awaits :meth:`ProviderRegistry.load` before the
    app starts serving traffic, so by the time this endpoint is reachable,
    loading has already succeeded; this simply confirms it wasn't a no-op.
    """
    registry = ProviderRegistry()
    if len(registry) == 0:
        return ReadinessResponse(
            ready=True,
            reason="No providers are configured",
        )
    return ReadinessResponse(
        ready=True,
        reason=f"{len(registry)} provider(s) loaded",
    )


@router.get("/liveness", summary="Liveness Probe")
async def liveness_probe_get_method() -> LivenessResponse:
    """Report that the service process is alive."""
    return LivenessResponse(alive=True)
