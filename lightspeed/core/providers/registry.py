"""Singleton registry of client-configured pydantic-ai providers."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Iterable, Iterator, Mapping

import boto3
from botocore.client import BaseClient as BedrockClient
from google.genai import Client as GoogleClient
from openai import AsyncOpenAI as OpenAIClient

from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider_class

from lightspeed.app.models.config import ProviderConfiguration, ProviderType
from lightspeed.core.utils.types import Singleton

# Constructor kwargs used by pydantic-ai providers for the configured base URL.
_URL_PARAM_NAMES = ("base_url", "api_base", "azure_endpoint")

# Lightspeed provider type -> pydantic-ai name for infer_provider_class.
_PROVIDER_BACKEND_MAP: dict[ProviderType, str] = {
    "openai": "openai",
    "azure": "azure",
    "bedrock": "bedrock",
    "vertexai": "google-cloud",
    "watsonx": "litellm",
    "vllm": "openai",  # OpenAI-compatible endpoint
}

# Lightspeed provider type -> model-kind prefix for infer_model. Kept distinct
# from _PROVIDER_BACKEND_MAP so OpenAI-compatible endpoints (vllm) use chat
# completions while native openai uses the Responses API.
_PROVIDER_MODEL_KIND_MAP: dict[ProviderType, str] = {
    "openai": "openai",
    "azure": "azure",
    "bedrock": "bedrock",
    "vertexai": "google-cloud",
    "watsonx": "litellm",
    "vllm": "openai-chat",
}


class ProviderRegistry(metaclass=Singleton):
    """Process-wide singleton holding the client-configured LLM providers and their models.

    :meth:`load` builds a pydantic-ai :class:`~pydantic_ai.providers.Provider`
    for each :class:`~lightspeed.app.models.config.ProviderConfiguration` entry
    and eagerly queries each one for its available models (e.g. at startup),
    so callers such as the agent factory can look providers and models up by
    name without any further client calls.
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider[Any]] = {}
        self._models: dict[str, list[Model]] = {}

    async def load(self, configs: Iterable[ProviderConfiguration]) -> None:
        """Replace the registry contents with providers (and their models) from ``configs``.

        Raises:
            ValueError: If two configs share a name, a ``type`` has no
                pydantic-ai backend mapping, or ``url`` / ``api_key`` are
                incompatible with the backend provider.
            TypeError: If a provider's client type has no known way to list
                models.
        """
        providers: dict[str, Provider[Any]] = {}
        model_kinds: dict[str, str] = {}
        for config in configs:
            if config.name in providers:
                raise ValueError(f"Provider already registered: {config.name!r}")
            providers[config.name] = _build_provider(config)
            model_kinds[config.name] = _PROVIDER_MODEL_KIND_MAP[config.type]

        names = list(providers)
        listings = await asyncio.gather(
            *(
                _list_models(name, providers[name], model_kinds[name])
                for name in names
            )
        )

        self._providers = providers
        self._models = dict(zip(names, listings))

    def get(self, name: str) -> Provider[Any]:
        """Return the pydantic-ai provider registered under ``name``.

        Raises:
            KeyError: If no provider with that name exists.
        """
        try:
            return self._providers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name!r}") from exc

    def get_models(self, name: str) -> list[Model]:
        """Return the models available to the provider registered under ``name``.

        Raises:
            KeyError: If no provider with that name exists.
        """
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Unknown provider: {name!r}") from exc

    def get_model(self, provider: str, model: str) -> Model:
        """Return the ``model`` registered for the provider named ``provider``.

        Raises:
            KeyError: If no provider named ``provider`` exists, or no model
                named ``model`` is available for that provider.
        """
        for candidate in self.get_models(provider):
            if candidate.model_name == model:
                return candidate

        raise KeyError(f"Unknown model {model!r} for provider {provider!r}")

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._providers

    def __len__(self) -> int:
        return len(self._providers)

    def __iter__(self) -> Iterator[str]:
        return iter(self._providers)

    @property
    def providers(self) -> Mapping[str, Provider[Any]]:
        """Read-only view of registered providers, keyed by name."""
        return dict(self._providers)

    @property
    def models(self) -> Mapping[str, list[Model]]:
        """Read-only view of each registered provider's available models."""
        return dict(self._models)


def _build_provider(config: ProviderConfiguration) -> Provider[Any]:
    """Construct a pydantic-ai provider from a Lightspeed provider config."""
    backend = _PROVIDER_BACKEND_MAP.get(config.type)
    if backend is None:
        raise ValueError(f"Unsupported provider type: {config.type!r}")

    provider_cls = infer_provider_class(backend)
    return provider_cls(**_provider_kwargs(config, provider_cls))


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
            raise ValueError(f"Provider type {config.type!r} does not accept a url")
        kwargs[url_param] = url

    return kwargs


async def _list_models(
    name: str, provider: Provider[Any], model_kind: str
) -> list[Model]:
    """List the models available to ``provider``, bound to it via ``model_kind``."""
    model_ids = await _list_model_ids(name, provider)
    return [
        infer_model(f"{model_kind}:{model_id}", provider_factory=lambda _n: provider)
        for model_id in model_ids
    ]


async def _list_model_ids(name: str, provider: Provider[Any]) -> list[str]:
    """Dispatch model-ID listing to the handler for the provider's client type."""
    client = provider.client

    if isinstance(client, OpenAIClient):
        return await _list_openai_compatible_model_ids(client)
    if isinstance(client, GoogleClient):
        return await _list_google_cloud_model_ids(client)
    if isinstance(client, BedrockClient):
        return await _list_bedrock_model_ids(client)

    raise TypeError(
        f"Don't know how to list models for client type {type(client).__name__!r} "
        f"(provider {name!r})"
    )


async def _list_openai_compatible_model_ids(client: OpenAIClient) -> list[str]:
    """List model IDs via an OpenAI-compatible client (openai, azure, litellm, vllm)."""
    return sorted([model.id async for model in client.models.list()])


async def _list_google_cloud_model_ids(client: GoogleClient) -> list[str]:
    """List model IDs via the Google GenAI (Vertex AI) client."""
    return sorted([model.name async for model in await client.aio.models.list()])


async def _list_bedrock_model_ids(client: BedrockClient) -> list[str]:
    """List Bedrock foundation model IDs available in the client's region.

    ``client`` is the ``bedrock-runtime`` client the provider invokes models
    through; foundation-model listing only exists on the separate ``bedrock``
    control-plane API, so a sibling client is built for the same region.
    Credentials are resolved via boto3's default provider chain, which does
    not cover the bearer-token API keys ``BedrockProvider`` accepts via
    ``api_key`` -- a bearer-token-only provider will fail to list models here.
    """
    control_client = boto3.client("bedrock", region_name=client.meta.region_name)
    response = await asyncio.to_thread(control_client.list_foundation_models)
    return sorted(
        summary["modelId"]
        for summary in response.get("modelSummaries", [])
        if "modelId" in summary
    )
