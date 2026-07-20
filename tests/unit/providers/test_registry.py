"""Tests for the client-side provider registry."""

from __future__ import annotations

import typing

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.providers.openai import OpenAIProvider

from lightspeed.app.models.config import (
    Configuration,
    ProviderConfiguration,
    ProviderType,
)
from lightspeed.src.providers.registry import PROVIDER_BACKEND_MAP, ProviderRegistry

SUPPORTED_TYPES: tuple[ProviderType, ...] = typing.get_args(ProviderType)


def test_provider_configuration_parses_yaml_shape() -> None:
    config = ProviderConfiguration.model_validate(
        {
            "name": "openai",
            "type": "openai",
            "url": "https://api.openai.com/v1",
            "api_key": "sk-test",
        }
    )
    assert config.name == "openai"
    assert config.type == "openai"
    assert str(config.url).rstrip("/") == "https://api.openai.com/v1"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "sk-test"


def test_configuration_rejects_unknown_provider_fields() -> None:
    with pytest.raises(ValidationError):
        ProviderConfiguration.model_validate(
            {
                "name": "openai",
                "type": "openai",
                "unexpected": True,
            }
        )


def test_configuration_rejects_unsupported_provider_type() -> None:
    with pytest.raises(ValidationError):
        ProviderConfiguration.model_validate(
            {"name": "x", "type": "ollama", "api_key": "sk"}
        )


def test_supported_types_match_backend_map() -> None:
    assert set(SUPPORTED_TYPES) == {
        "openai",
        "azure",
        "bedrock",
        "vertexai",
        "watsonx",
        "vllm",
    }
    assert set(PROVIDER_BACKEND_MAP) == set(SUPPORTED_TYPES)
    for provider_type in SUPPORTED_TYPES:
        TypeAdapter(ProviderType).validate_python(provider_type)


def test_root_configuration_accepts_providers_list() -> None:
    config = Configuration.model_validate(
        {
            "name": "Lightspeed Stack",
            "providers": [
                {"name": "openai", "type": "openai", "api_key": "sk-test"},
                {
                    "name": "my-vllm",
                    "type": "vllm",
                    "url": "http://localhost:8000/v1",
                    "api_key": "not-needed",
                },
            ],
        }
    )
    assert len(config.providers) == 2
    assert config.providers[1].type == "vllm"


def test_registry_creates_pydantic_ai_providers() -> None:
    registry = ProviderRegistry.from_configs(
        [
            ProviderConfiguration(name="openai", type="openai", api_key="sk-test"),
            ProviderConfiguration(
                name="watsonx",
                type="watsonx",
                api_key="wx-test",
            ),
        ]
    )

    openai = registry.get("openai")
    watsonx = registry.get("watsonx")

    assert isinstance(openai, OpenAIProvider)
    assert isinstance(watsonx, LiteLLMProvider)
    assert openai.name == "openai"
    assert watsonx.name == "litellm"
    assert list(registry) == ["openai", "watsonx"]


def test_registry_maps_vllm_to_openai_compatible_provider() -> None:
    registry = ProviderRegistry.from_configs(
        [
            ProviderConfiguration(
                name="my-vllm",
                type="vllm",
                url="http://localhost:8000/v1",
                api_key="not-needed",
            )
        ]
    )
    provider = registry.get("my-vllm")
    assert isinstance(provider, OpenAIProvider)
    assert str(provider.base_url).rstrip("/") == "http://localhost:8000/v1"


def test_registry_rejects_duplicate_names() -> None:
    registry = ProviderRegistry()
    registry.register(ProviderConfiguration(name="openai", type="openai", api_key="sk"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            ProviderConfiguration(name="openai", type="openai", api_key="sk-other")
        )


def test_registry_get_unknown_name() -> None:
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="Unknown provider"):
        registry.get("missing")
