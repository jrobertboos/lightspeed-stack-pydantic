"""Registry of client-configured pydantic-ai providers."""

from __future__ import annotations

import inspect
from typing import Any, Iterable, Iterator, Mapping, Optional

from pydantic_ai.providers import Provider, infer_provider_class

from lightspeed.app.models.config import ProviderConfiguration, ProviderType

# Constructor kwargs used by pydantic-ai providers for the configured base URL.
_URL_PARAM_NAMES = ("base_url", "api_base", "azure_endpoint")

# Lightspeed provider type -> pydantic-ai name for infer_provider_class.
PROVIDER_BACKEND_MAP: dict[ProviderType, str] = {
    "openai": "openai",
    "azure": "azure",
    "bedrock": "bedrock",
    "vertexai": "google-cloud",
    "watsonx": "litellm",
    "vllm": "openai",  # OpenAI-compatible endpoint
}


class ProviderRegistry:
    """Lookup table of pydantic-ai :class:`~pydantic_ai.providers.Provider` instances.

    Providers are created from :class:`~lightspeed.app.models.config.ProviderConfiguration`
    entries. ``type`` is mapped through :data:`PROVIDER_BACKEND_MAP` onto a
    pydantic-ai provider class; ``url`` / ``api_key`` are forwarded using the
    parameter names that class accepts.
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider[Any]] = {}

    @classmethod
    def from_configs(
        cls, configs: Iterable[ProviderConfiguration]
    ) -> ProviderRegistry:
        """Build a registry from a sequence of provider configurations."""
        registry = cls()
        for config in configs:
            registry.register(config)
        return registry

    def register(self, config: ProviderConfiguration) -> Provider[Any]:
        """Create a pydantic-ai provider from ``config`` and store it by ``name``.

        Raises:
            ValueError: If ``name`` is already registered, ``type`` has no
                backend mapping, or ``url`` / ``api_key`` are incompatible with
                the backend provider.
        """
        if config.name in self._providers:
            raise ValueError(f"Provider already registered: {config.name!r}")

        backend = PROVIDER_BACKEND_MAP.get(config.type)
        if backend is None:
            raise ValueError(f"Unsupported provider type: {config.type!r}")

        provider_cls = infer_provider_class(backend)
        provider = provider_cls(**_provider_kwargs(config, provider_cls))
        self._providers[config.name] = provider
        return provider

    def get(self, name: str) -> Provider[Any]:
        """Return the provider registered under ``name``.

        Raises:
            KeyError: If no provider with that name exists.
        """
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name!r}") from exc

    def get_optional(self, name: str) -> Optional[Provider[Any]]:
        """Return the provider registered under ``name``, or ``None``."""
        return self._providers.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._providers

    def __len__(self) -> int:
        return len(self._providers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._providers)

    def items(self) -> Iterator[tuple[str, Provider[Any]]]:
        """Iterate ``(name, provider)`` pairs."""
        return iter(self._providers.items())

    @property
    def providers(self) -> Mapping[str, Provider[Any]]:
        """Read-only view of registered providers."""
        return self._providers


def _provider_kwargs(
    config: ProviderConfiguration,
    provider_cls: type[Provider[Any]],
) -> dict[str, Any]:
    """Map Lightspeed config fields onto a pydantic-ai provider constructor."""
    params = inspect.signature(provider_cls.__init__).parameters
    kwargs: dict[str, Any] = {}

    if config.api_key is not None:
        if "api_key" not in params:
            raise ValueError(
                f"Provider type {config.type!r} does not accept an api_key"
            )
        kwargs["api_key"] = config.api_key.get_secret_value()

    if config.url is not None:
        url = str(config.url)
        url_param = next((name for name in _URL_PARAM_NAMES if name in params), None)
        if url_param is None:
            raise ValueError(
                f"Provider type {config.type!r} does not accept a url"
            )
        kwargs[url_param] = url

    return kwargs
