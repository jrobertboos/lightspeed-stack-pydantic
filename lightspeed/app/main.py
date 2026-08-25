"""Definition of the FastAPI application.

Minimal rewrite of the original Lightspeed Stack ``app/main.py``. Sentry,
the database, A2A storage, Azure token management, and REST API metrics are
not yet implemented; only configuration loading, provider/model discovery,
and a generic exception-to-500-JSON middleware are wired up here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from lightspeed.app import routers
from lightspeed.app.models.responses.error import InternalServerErrorResponse
from lightspeed.core.providers.registry import ProviderRegistry
from lightspeed.core.config import configuration, configuration_path_from_env

logger = logging.getLogger(__name__)

# Relies on `configuration` already being loaded by the entrypoint
# (lightspeed/main.py) before it calls `start_uvicorn`, the same way the
# original app/main.py's `service_name = configuration.configuration.name`
# relies on `lightspeed_stack.py` having loaded it first. That only holds
# because Uvicorn runs this module in the entrypoint's own process when
# workers=1 (our default); see the re-load in `lifespan` below for the
# multi-worker case.
service_name = configuration.configuration.name


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load configuration and providers before serving traffic.

    Configuration is re-loaded here (from CONFIG_PATH_ENV_VAR) because each
    Uvicorn worker is a separate process that may not inherit the
    entrypoint's already-loaded singleton, depending on the platform's
    multiprocessing start method. Provider/model discovery needs an event
    loop too, so it happens here as well.
    """
    configuration.load_configuration(configuration_path_from_env())

    providers = configuration.configuration.providers
    logger.info("Loading %d configured provider(s)", len(providers))
    await ProviderRegistry().load(providers)
    logger.info("App startup complete")

    yield

    logger.info("App shutdown complete")


app = FastAPI(
    title=f"{service_name} service - OpenAPI",
    summary=f"{service_name} service API specification.",
    description=f"{service_name} service API specification.",
    version="0.1.0",
    lifespan=lifespan,
)


class GlobalExceptionMiddleware:  # pylint: disable=too-few-public-methods
    """Pure ASGI middleware that maps uncaught exceptions to a 500 JSON response.

    Implemented as pure ASGI (instead of Starlette's ``BaseHTTPMiddleware``)
    to avoid the ``RuntimeError: No response returned`` failure mode that
    ``call_next`` runs into with long-running handlers such as LLM inference.
    """

    def __init__(self, asgi_app: ASGIApp) -> None:
        """Wrap the inner ASGI application."""
        self.app = asgi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Forward the request, converting uncaught exceptions to a 500 response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except HTTPException:
            raise
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Uncaught exception in endpoint")
            if response_started:
                raise
            error_response = InternalServerErrorResponse.generic()
            response = JSONResponse(
                status_code=error_response.status_code,
                content={"detail": error_response.detail.model_dump()},
            )
            await response(scope, receive, send)


app.add_middleware(GlobalExceptionMiddleware)

routers.include_routers(app)
