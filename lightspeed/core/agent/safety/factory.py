"""Build safety capabilities from configured safety shields.

Each configured ``safety`` entry (see
:data:`~lightspeed.app.models.config.SafetyCapabilityConfiguration`) becomes one
:class:`~lightspeed.core.agent.safety.base.AbstractSafetyCapability`, selected by ``type`` and
constructed from ``config``:

- ``question_validity`` ->
  :class:`~lightspeed.core.agent.safety.question_validity.capability.QuestionValidityCapability`
- ``redaction`` -> :class:`~lightspeed.core.agent.safety.redaction.capability.RedactionCapability`

``config`` fields left unset fall back to the target capability's own defaults, rather than
this factory (or the configuration schema) duplicating them. For ``question_validity``,
``config.model`` (``<provider>:<model>``) is resolved to a pydantic-ai ``Model`` via
:class:`~lightspeed.core.providers.registry.ProviderRegistry` before construction;
``ProviderRegistry`` must already be loaded when set. Omitting it falls back to the run's own
model.
"""

from __future__ import annotations

from typing import Any, Iterable

from pydantic_ai.capabilities import AgentCapability

from lightspeed.app.models.config import (
    QuestionValiditySafetyConfiguration,
    RedactionSafetyConfiguration,
    SafetyCapabilityConfiguration,
)
from lightspeed.core.agent.capability_factory import CapabilityFactory
from lightspeed.core.agent.safety.question_validity.capability import (
    QuestionValidityCapability,
)
from lightspeed.core.agent.safety.redaction.capability import RedactionCapability
from lightspeed.core.providers.registry import ProviderRegistry


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
    kwargs = config.config.model_dump(exclude_none=True)
    if isinstance(config, QuestionValiditySafetyConfiguration):
        model = kwargs.pop("model", None)
        if model is not None:
            kwargs["model"] = ProviderRegistry().get_model(model)
        return QuestionValidityCapability(id=config.name, **kwargs)
    if isinstance(config, RedactionSafetyConfiguration):
        return RedactionCapability(id=config.name, **kwargs)
    # Unreachable while `SafetyCapabilityConfiguration` only has the two variants above;
    # guards against silently ignoring a new variant added without updating this factory.
    raise ValueError(f"Unsupported safety shield type: {config.type!r}")
