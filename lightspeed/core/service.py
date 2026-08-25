from lightspeed.app.models.config import Configuration
from lightspeed.core.config import configuration as app_config


def get_configuration() -> Configuration:
    """Return the loaded :class:`Configuration` from the process-wide singleton.

    The FastAPI ``lifespan`` (see :mod:`lightspeed.app.main`) loads it before
    the app starts serving traffic, so this should only raise if called
    outside of a running app (e.g. a unit test that imports this module
    without loading configuration first).

    This is transport-agnostic: it's up to callers (e.g. endpoint handlers)
    to decide how a not-yet-loaded configuration should surface, such as
    translating it into an HTTP error response.

    Raises:
        RuntimeError: If the configuration hasn't been loaded yet.
    """
    return app_config.configuration