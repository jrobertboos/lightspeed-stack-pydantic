"""Registry of client-configured pydantic-ai providers and models."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional

from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider_class
from pydantic_ai.settings import ModelSettings

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

# Lightspeed provider type -> model-kind prefix for infer_model.
# Distinct from PROVIDER_BACKEND_MAP so OpenAI-compatible endpoints (vllm) can
# use chat completions while native openai uses the Responses API.
PROVIDER_MODEL_KIND_MAP: dict[ProviderType, str] = {
    "openai": "openai",
    "azure": "azure",
    "bedrock": "bedrock",
    "vertexai": "google-cloud",
    "watsonx": "litellm",
    "vllm": "openai-chat",
}


@dataclass(frozen=True, slots=True)
class RegisteredProvider:
    """A configured provider entry held by the registry."""

    config: ProviderConfiguration
    provider: Provider[Any]


class ProviderRegistry:
    """Lookup table of pydantic-ai providers and models.

    Providers are created from :class:`~lightspeed.app.models.config.ProviderConfiguration`
    entries. ``type`` is mapped through :data:`PROVIDER_BACKEND_MAP` onto a
    pydantic-ai provider class; ``url`` / ``api_key`` are forwarded using the
    parameter names that class accepts.

    :meth:`get_model` binds a model name to a registered provider via
    pydantic-ai's :func:`~pydantic_ai.models.infer_model`, so a future agent
    loader can do ``Agent(registry.get_model(...))`` similarly to the original
    ``build_agent`` helper.
    """

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredProvider] = {}

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
        if config.name in self._entries:
            raise ValueError(f"Provider already registered: {config.name!r}")

        backend = PROVIDER_BACKEND_MAP.get(config.type)
        if backend is None:
            raise ValueError(f"Unsupported provider type: {config.type!r}")

        provider_cls = infer_provider_class(backend)
        provider = provider_cls(**_provider_kwargs(config, provider_cls))
        self._entries[config.name] = RegisteredProvider(
            config=config, provider=provider
        )
        return provider

    def get(self, name: str) -> Provider[Any]:
        """Return the provider registered under ``name``.

        Raises:
            KeyError: If no provider with that name exists.
        """
        return self._get_entry(name).provider

    def get_model(
        self,
        name: str,
        model_name: str,
        *,
        settings: Optional[ModelSettings] = None,
    ) -> Model:
        """Return a pydantic-ai :class:`~pydantic_ai.models.Model` for ``name``.

        The model is constructed with the registered provider instance, using
        pydantic-ai's model-kind rules for this Lightspeed provider type.

        Parameters:
            name: Registry key of the configured provider.
            model_name: Upstream model identifier (e.g. ``gpt-4o``).
            settings: Optional pydantic-ai :class:`~pydantic_ai.settings.ModelSettings`
                applied as defaults on the returned model.

        Returns:
            A pydantic-ai ``Model`` ready to pass to ``Agent(...)``.

        Raises:
            KeyError: If no provider with that name exists.
            ValueError: If ``model_name`` is empty.
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        entry = self._get_entry(name)
        model_kind = PROVIDER_MODEL_KIND_MAP[entry.config.type]
        provider = entry.provider
        model = infer_model(
            f"{model_kind}:{model_name.strip()}",
            provider_factory=lambda _provider_name: provider,
        )
        return type(model)(model.model_name, provider=provider, settings=settings)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def items(self) -> Iterator[tuple[str, Provider[Any]]]:
        """Iterate ``(name, provider)`` pairs."""
        return ((name, entry.provider) for name, entry in self._entries.items())

    @property
    def providers(self) -> Mapping[str, Provider[Any]]:
        """Read-only view of registered providers."""
        return {name: entry.provider for name, entry in self._entries.items()}

    def _get_entry(self, name: str) -> RegisteredProvider:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name!r}") from exc


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
