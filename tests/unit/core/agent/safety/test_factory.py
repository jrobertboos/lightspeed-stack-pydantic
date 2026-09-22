"""Unit tests for :mod:`lightspeed.core.agent.safety.factory`."""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

import lightspeed.core.agent.safety.factory as factory_module
from lightspeed.app.models.config import (
    GraniteGuardianConfig,
    GraniteGuardianSafetyConfiguration,
    QuestionValiditySafetyConfiguration,
    RedactionSafetyConfiguration,
)
from lightspeed.core.agent.safety.factory import SafetyCapabilityFactory
from lightspeed.core.agent.safety.granite_guardian.capability import GraniteGuardian
from lightspeed.core.agent.safety.question_validity.capability import (
    QuestionValidityCapability,
)
from lightspeed.core.agent.safety.redaction.capability import RedactionCapability


class FakeProviderRegistry:
    """Stand-in for the `ProviderRegistry` singleton, returning a fixed model."""

    def __init__(self, model=None):
        self._model = model or TestModel()
        self.requested: list[str] = []

    def get_model(self, model: str):
        self.requested.append(model)
        return self._model


@pytest.fixture(autouse=True)
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> FakeProviderRegistry:
    registry = FakeProviderRegistry()
    monkeypatch.setattr(factory_module, "ProviderRegistry", lambda: registry)
    return registry


class TestSafetyCapabilityFactory:
    """Tests for :meth:`SafetyCapabilityFactory.build_capabilities`."""

    def test_empty_config_returns_empty_list(self) -> None:
        assert SafetyCapabilityFactory.build_capabilities([]) == []

    def test_question_validity_config_builds_capability(self, fake_registry: FakeProviderRegistry) -> None:
        config = QuestionValiditySafetyConfiguration(name="qv", type="question_validity")
        config.config.model = "openai:gpt-4o"

        capabilities = SafetyCapabilityFactory.build_capabilities([config])

        assert len(capabilities) == 1
        assert isinstance(capabilities[0], QuestionValidityCapability)
        assert capabilities[0].id == "qv"
        assert capabilities[0].model is fake_registry._model
        assert fake_registry.requested == ["openai:gpt-4o"]

    def test_redaction_config_builds_capability_without_provider_registry(
        self, fake_registry: FakeProviderRegistry
    ) -> None:
        config = RedactionSafetyConfiguration(name="redact", type="redaction")
        config.config.replacement = "***"

        capabilities = SafetyCapabilityFactory.build_capabilities([config])

        assert len(capabilities) == 1
        assert isinstance(capabilities[0], RedactionCapability)
        assert capabilities[0].id == "redact"
        assert capabilities[0].replacement == "***"
        assert fake_registry.requested == []

    def test_granite_guardian_config_builds_combined_capability(
        self, fake_registry: FakeProviderRegistry
    ) -> None:
        config = GraniteGuardianSafetyConfiguration(
            name="guardian",
            type="granite_guardian",
            config=GraniteGuardianConfig(
                model="watsonx:granite-guardian-3-8b",
                risks=[{"name": "harm", "criteria": "flags harm"}],
            ),
        )

        capabilities = SafetyCapabilityFactory.build_capabilities([config])

        assert len(capabilities) == 1
        guardian = capabilities[0]
        assert isinstance(guardian, GraniteGuardian)
        assert guardian.id == "guardian"
        assert guardian.model is fake_registry._model
        assert fake_registry.requested == ["watsonx:granite-guardian-3-8b"]
        assert len(guardian.capabilities) == 1

    def test_multiple_configs_build_multiple_capabilities(self) -> None:
        redaction_config = RedactionSafetyConfiguration(name="redact", type="redaction")
        qv_config = QuestionValiditySafetyConfiguration(name="qv", type="question_validity")

        capabilities = SafetyCapabilityFactory.build_capabilities([redaction_config, qv_config])

        assert len(capabilities) == 2
        assert isinstance(capabilities[0], RedactionCapability)
        assert isinstance(capabilities[1], QuestionValidityCapability)

    def test_duplicate_name_raises_value_error(self) -> None:
        first = RedactionSafetyConfiguration(name="dup", type="redaction")
        second = QuestionValiditySafetyConfiguration(name="dup", type="question_validity")

        with pytest.raises(ValueError, match="dup"):
            SafetyCapabilityFactory.build_capabilities([first, second])
