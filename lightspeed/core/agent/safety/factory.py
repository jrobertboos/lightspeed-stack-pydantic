"""Build safety capabilities from configured safety shields.

Each configured ``safety`` entry (see
:data:`~lightspeed.app.models.config.SafetyCapabilityConfiguration`) becomes one
:class:`~lightspeed.core.agent.safety.base.AbstractSafetyCapability`, selected by ``type`` and
constructed from ``config``:

- ``question_validity`` ->
  :class:`~lightspeed.core.agent.safety.question_validity.capability.QuestionValidityCapability`
- ``redaction`` -> :class:`~lightspeed.core.agent.safety.redaction.capability.RedactionCapability`
- ``granite_guardian`` -> :class:`~lightspeed.core.agent.safety.granite_guardian.capability.GraniteGuardian`,
  a combined capability of one
  :class:`~lightspeed.core.agent.safety.granite_guardian.capability.GraniteGuardianRisk` per
  configured risk

``config`` fields left unset fall back to the target capability's own defaults, rather than
this factory (or the configuration schema) duplicating them. For ``question_validity``,
``config.model`` (``<provider>:<model>``) is resolved to a pydantic-ai ``Model`` via
:class:`~lightspeed.core.providers.registry.ProviderRegistry` before construction;
``ProviderRegistry`` must already be loaded when set. Omitting it falls back to the run's own
model. ``granite_guardian``'s ``config.model`` is resolved the same way and falls back the
same way when omitted.
"""

from __future__ import annotations

from typing import Any, Iterable

from pydantic_ai.capabilities import AgentCapability

from lightspeed.app.models.config import (
    GraniteGuardianSafetyConfiguration,
    QuestionValiditySafetyConfiguration,
    RedactionSafetyConfiguration,
    SafetyCapabilityConfiguration,
)
from lightspeed.core.agent.capability_factory import CapabilityFactory
from lightspeed.core.agent.safety.granite_guardian.capability import GraniteGuardian
from lightspeed.core.agent.safety.question_validity.capability import (
    QuestionValidityCapability,
)
from lightspeed.core.agent.safety.redaction.capability import RedactionCapability


class SafetyCapabilityFactory(CapabilityFactory[Iterable[SafetyCapabilityConfiguration]]):
    """Builds safety capabilities (``question_validity``, ``redaction``, ...) from config."""

    @staticmethod
    def build_capabilities(
        configs: Iterable[SafetyCapabilityConfiguration],
    ) -> list[AgentCapability[Any]]:
        """Build one safety capability per configured shield.

        Parameters:
            configs: The configured safety shields (e.g. ``Configuration.safety``).

        Returns:
            One safety capability per config, suitable for ``Agent(capabilities=...)``.

        Raises:
            ValueError: If two configs share a name.
        """
        seen: set[str] = set()
        capabilities: list[AgentCapability[Any]] = []
        for config in configs:
            if config.name in seen:
                raise ValueError(f"Safety shield already registered: {config.name!r}")
            seen.add(config.name)
            capabilities.append(_build_capability(config))
        return capabilities


def _build_capability(config: SafetyCapabilityConfiguration) -> AgentCapability[Any]:
    """Build a single safety capability, dispatching on ``config.type``."""
    if isinstance(config, GraniteGuardianSafetyConfiguration):
        return GraniteGuardian(
            id=config.name,
            model=config.config.model,
            risks=config.config.risks,
            output_check_interval_tokens=config.config.output_check_interval_tokens,
        )

    if isinstance(config, QuestionValiditySafetyConfiguration):
        return QuestionValidityCapability(
            id=config.name,
            model=config.config.model,
            invalid_question_response=config.config.invalid_question_response,
            classifier_instructions=config.config.classifier_instructions,
        )
    if isinstance(config, RedactionSafetyConfiguration):
        return RedactionCapability(
            id=config.name,
            patterns=config.config.patterns,
            replacement=config.config.replacement,
        )
    raise ValueError(f"Unsupported safety shield type: {config.type!r}")
