"""Configuration loading and access.

Singleton mirror of the original Lightspeed Stack's ``configuration.py``
``AppConfig``, built on the :class:`~lightspeed.core.utils.types.Singleton`
metaclass already used by
:class:`~lightspeed.core.providers.registry.ProviderRegistry` (instead of the
original's ``__new__`` override).

There is one :class:`AppConfig` instance per process: the entrypoint
(:mod:`lightspeed.main`) loads it once to validate the file up front, and
each Uvicorn worker process (a separate process, so a separate singleton)
loads it again from :data:`CONFIG_PATH_ENV_VAR` in :mod:`lightspeed.app.main`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from lightspeed.app.models.config import Configuration, ServiceConfiguration
from lightspeed.core.types import Singleton

#: Environment variable used to pass the resolved configuration file path
#: from the entrypoint to each Uvicorn worker process.
CONFIG_PATH_ENV_VAR = "LIGHTSPEED_STACK_CONFIG_PATH"

#: Default configuration file name, relative to the current working directory.
DEFAULT_CONFIGURATION_FILE = "lightspeed-stack.yaml"


class AppConfig(metaclass=Singleton):
    """Singleton that loads and holds the Lightspeed Stack :class:`Configuration`."""

    def __init__(self) -> None:
        """Initialize the class instance with no configuration loaded yet."""
        self._configuration: Optional[Configuration] = None

    def load_configuration(self, path: str | Path) -> None:
        """Load and validate the configuration from a YAML file.

        Parameters:
            path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            pydantic.ValidationError: If the YAML contents don't match the schema.
        """
        with open(path, "r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file) or {}
        self._configuration = Configuration.model_validate(raw_config)

    @property
    def configuration(self) -> Configuration:
        """Return the loaded configuration.

        Returns:
            The loaded :class:`Configuration`.

        Raises:
            RuntimeError: If the configuration has not been loaded yet.
        """
        if self._configuration is None:
            raise RuntimeError("logic error: configuration is not loaded")
        return self._configuration

    @property
    def service_configuration(self) -> ServiceConfiguration:
        """Return the service (host/port/workers/...) configuration.

        Returns:
            The :class:`ServiceConfiguration` from the loaded configuration.

        Raises:
            RuntimeError: If the configuration has not been loaded yet.
        """
        return self.configuration.service


def configuration_path_from_env(
    default: str | Path = DEFAULT_CONFIGURATION_FILE,
) -> str:
    """Resolve the configuration file path set by the entrypoint.

    Parameters:
        default: Fallback path used when :data:`CONFIG_PATH_ENV_VAR` is unset,
            e.g. when :mod:`lightspeed.app.main` is imported directly (tests,
            ``uvicorn --reload``) rather than via :func:`lightspeed.main.main`.

    Returns:
        The configuration file path to load.
    """
    return os.environ.get(CONFIG_PATH_ENV_VAR, str(default))


# Process-wide singleton, imported and used the same way as the original
# Lightspeed Stack's `from configuration import configuration` (e.g.
# `configuration.load_configuration(path)`, `configuration.configuration.name`,
# `configuration.service_configuration`).
configuration: AppConfig = AppConfig()
