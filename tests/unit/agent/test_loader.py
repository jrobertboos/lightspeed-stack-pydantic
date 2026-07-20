"""Tests for loading pydantic-ai agents from the provider registry."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.settings import ModelSettings

from lightspeed.app.models.config import ProviderConfiguration
from lightspeed.src.agent.loader import load_agent
from lightspeed.src.providers.registry import ProviderRegistry


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry.from_configs(
        [
            ProviderConfiguration(name="openai", type="openai", api_key="sk-test"),
            ProviderConfiguration(
                name="my-vllm",
                type="vllm",
                url="http://localhost:8000/v1",
                api_key="not-needed",
            ),
        ]
    )


def test_load_agent_binds_registry_model(registry: ProviderRegistry) -> None:
    agent = load_agent(registry, "openai", "gpt-4o", instructions="Be concise.")

    assert isinstance(agent, Agent)
    assert isinstance(agent.model, OpenAIResponsesModel)
    assert agent.model.model_name == "gpt-4o"
    assert agent.model.provider is registry.get("openai")


def test_load_agent_applies_model_settings(registry: ProviderRegistry) -> None:
    settings = ModelSettings(temperature=0.1, max_tokens=64)
    agent = load_agent(registry, "openai", "gpt-4o", settings=settings)

    assert agent.model is not None
    assert agent.model.settings == settings


def test_load_agent_uses_chat_model_for_vllm(registry: ProviderRegistry) -> None:
    agent = load_agent(registry, "my-vllm", "granite-3.3-8b-instruct")

    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "granite-3.3-8b-instruct"


def test_load_agent_unknown_provider(registry: ProviderRegistry) -> None:
    with pytest.raises(KeyError, match="Unknown provider"):
        load_agent(registry, "missing", "gpt-4o")


def test_load_agent_rejects_empty_model_name(registry: ProviderRegistry) -> None:
    with pytest.raises(ValueError, match="model_name"):
        load_agent(registry, "openai", "")
