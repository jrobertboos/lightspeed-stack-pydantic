"""Unit tests for :mod:`lightspeed.core.agent.safety.granite_guardian.capability`."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

import lightspeed.core.agent.safety.granite_guardian.capability as capability_module
from lightspeed.core.agent.safety.granite_guardian.capability import (
    DEFAULT_VIOLATION_MESSAGE,
    GraniteGuardian,
    GraniteGuardianRisk,
    Risk,
)


def run(coro):
    return asyncio.run(coro)


def _response_with_logprobs(logprobs: dict) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="")], provider_details={"logprobs": logprobs})


def patch_model_request(monkeypatch: pytest.MonkeyPatch, response: ModelResponse | Exception):
    """Stub out `model_request` so `GraniteGuardianRisk.evaluate` never hits a real model."""
    captured: dict = {}

    async def fake_model_request(*, model, messages, model_settings=None):
        captured["model"] = model
        captured["messages"] = messages
        captured["model_settings"] = model_settings
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(capability_module, "model_request", fake_model_request)
    return captured


class TestGraniteGuardianRiskEvaluate:
    """Tests for :meth:`GraniteGuardianRisk.evaluate`."""

    def test_safe_verdict_allows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_model_request(monkeypatch, _response_with_logprobs({"safe": True}))
        monkeypatch.setattr(capability_module, "is_safe", lambda threshold, logprobs: True)
        risk = Risk(name="harm", criteria="flags harmful content")
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())

        result = run(capability.evaluate("hello"))
        assert result.action == "allow"

    def test_risky_verdict_blocks_with_violation_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_model_request(monkeypatch, _response_with_logprobs({"safe": False}))
        monkeypatch.setattr(capability_module, "is_safe", lambda threshold, logprobs: False)
        risk = Risk(name="harm", criteria="flags harmful content", violation_message="Blocked: harm")
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())

        result = run(capability.evaluate("hurt someone"))
        assert result.action == "block"
        assert result.message == "Blocked: harm"

    def test_default_violation_message_used_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_model_request(monkeypatch, _response_with_logprobs({"safe": False}))
        monkeypatch.setattr(capability_module, "is_safe", lambda threshold, logprobs: False)
        risk = Risk(name="harm", criteria="flags harmful content")
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())

        result = run(capability.evaluate("hurt someone"))
        assert result.message == DEFAULT_VIOLATION_MESSAGE

    def test_missing_provider_details_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_model_request(monkeypatch, ModelResponse(parts=[TextPart(content="")]))
        risk = Risk(name="harm", criteria="flags harmful content")
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())

        with pytest.raises(UnexpectedModelBehavior):
            run(capability.evaluate("hello"))

    def test_missing_logprobs_field_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_model_request(
            monkeypatch,
            ModelResponse(parts=[TextPart(content="")], provider_details={"other": "value"}),
        )
        risk = Risk(name="harm", criteria="flags harmful content")
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())

        with pytest.raises(UnexpectedModelBehavior):
            run(capability.evaluate("hello"))

    def test_type_derived_from_risk_type_in_post_init(self) -> None:
        risk = Risk(name="harm", criteria="c", type=["input", "output"])
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())
        assert list(capability.type) == ["input", "output"]

    def test_evaluate_sends_risk_criteria_as_instructions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = patch_model_request(monkeypatch, _response_with_logprobs({"safe": True}))
        monkeypatch.setattr(capability_module, "is_safe", lambda threshold, logprobs: True)
        risk = Risk(name="harm", criteria="flags harmful content")
        capability = GraniteGuardianRisk(id="shield.harm", risk=risk, model=TestModel())

        run(capability.evaluate("hello there"))

        sent_request: ModelRequest = captured["messages"][0]
        assert "flags harmful content" in sent_request.instructions
        assert captured["model_settings"].get("openai_logprobs") is True


class TestGraniteGuardian:
    """Tests for :class:`GraniteGuardian`, which combines risks into one shield."""

    def test_builds_one_sub_capability_per_risk(self) -> None:
        model = TestModel()
        risks = [
            Risk(name="harm", criteria="flags harm"),
            Risk(name="jailbreak", criteria="flags jailbreak"),
        ]
        guardian = GraniteGuardian(id="my_shield", risks=risks, model=model)

        assert len(guardian.capabilities) == 2
        assert all(isinstance(sub, GraniteGuardianRisk) for sub in guardian.capabilities)
        assert {sub.risk.name for sub in guardian.capabilities} == {"harm", "jailbreak"}

    def test_sub_capability_ids_are_prefixed_with_shield_id(self) -> None:
        guardian = GraniteGuardian(
            id="my_shield", risks=[Risk(name="harm", criteria="flags harm")], model=TestModel()
        )
        assert guardian.capabilities[0].id == "my_shield.harm"

    def test_sub_capabilities_share_the_model(self) -> None:
        model = TestModel()
        guardian = GraniteGuardian(
            id="my_shield", risks=[Risk(name="harm", criteria="flags harm")], model=model
        )
        assert guardian.capabilities[0].model is model

    def test_output_check_interval_tokens_propagated(self) -> None:
        guardian = GraniteGuardian(
            id="my_shield",
            risks=[Risk(name="harm", criteria="flags harm")],
            model=TestModel(),
            output_check_interval_tokens=10,
        )
        assert guardian.capabilities[0].output_check_interval_tokens == 10

    def test_no_id_produces_unprefixed_sub_ids(self) -> None:
        guardian = GraniteGuardian(risks=[Risk(name="harm", criteria="flags harm")], model=TestModel())
        assert guardian.capabilities[0].id == "harm"
