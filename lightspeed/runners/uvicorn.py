"""Uvicorn runner.

Minimal rewrite of the original Lightspeed Stack Uvicorn runner. TLS
(``ssl_keyfile`` / ``ssl_certfile``) and custom logging configuration
(colorized output, the original's ``log.py``) are not yet implemented.
"""

from __future__ import annotations

import logging

import uvicorn

from lightspeed.app.models.config import ServiceConfiguration

logger = logging.getLogger(__name__)


def start_uvicorn(configuration: ServiceConfiguration) -> None:
    """Start the Uvicorn server using the provided service configuration.

    Parameters:
        configuration: Service configuration providing ``host``, ``port``,
            ``workers``, and ``access_log``. TLS and custom logging
            configuration are not yet implemented.
    """
    logger.info(
        "Starting Uvicorn on %s:%d with %d worker(s)",
        configuration.host,
        configuration.port,
        configuration.workers,
    )

    uvicorn.run(
        "lightspeed.app.main:app",
        host=configuration.host,
        port=configuration.port,
        workers=configuration.workers,
        access_log=configuration.access_log,
    )
