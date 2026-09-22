"""``GraniteGuardianRisk`` and ``GraniteGuardian``: Granite-Guardian risk checks and their shield.

A ``granite_guardian`` shield can list several independent risks (e.g. jailbreak, harm,
groundedness). Rather than one capability that loops over all of them internally, each risk is
its own :class:`GraniteGuardianRisk`, and :class:`GraniteGuardian` combines one per risk via
:class:`~pydantic_ai.capabilities.CombinedCapability`, so each risk is its own guard in the
agent's capability chain -- screened, ordered, and reasoned about independently -- while still
sharing the one Granite Guardian `model`.

Neither class depends on :mod:`lightspeed.app.models.config`: `GraniteGuardian` takes plain
`Risk` values rather than a config object, so this module stays usable (and testable)
independent of the YAML config schema. `GraniteGuardianConfig.risks` is `Risk` directly (the
dependency runs the other way -- config on capability, not capability on config), so
:mod:`~lightspeed.core.agent.safety.factory` passes it through unchanged, resolving only
`GraniteGuardianConfig.model` to a pydantic-ai `Model` via `ProviderRegistry`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.tools import AgentDepsT

from lightspeed.core.agent.safety.base import AbstractSafetyCapability, GuardrailResult
from lightspeed.core.agent.safety.granite_guardian.utils import build_guardian_block, is_safe

DEFAULT_VIOLATION_MESSAGE = "I can't help with that."
"""Fallback `Risk.violation_message` / `GraniteGuardianRisk.violation_message` when a risk
doesn't set its own. Deliberately generic -- it says nothing about which risk was flagged --
since callers who want to tell the flagged risk apart in the response should set their own
per-risk message instead."""


@dataclass(frozen=True)
class Risk:
    """Plain description of one risk for :class:`GraniteGuardian` to build a
    :class:`GraniteGuardianRisk` from.

    Not a pydantic model: keeping this a plain dataclass means constructing a `GraniteGuardian`
    doesn't require going through YAML config at all. `GraniteGuardianConfig.risks` uses this
    type directly (pydantic validates dicts onto stdlib dataclasses like any other field type),
    so there's no separate config/schema type duplicating these fields.
    """

    name: str
    """Unique name for this risk within its shield; combined with the shield's `id` for this
    risk's own `GraniteGuardianRisk.id`."""

    criteria: str
    """Natural-language description of what this risk flags, sent to Granite Guardian as the
    judging criteria."""

    violation_message: str = DEFAULT_VIOLATION_MESSAGE
    """Message returned to the caller in place of a prompt or response that violates this
    risk."""

    threshold: float = 0.5
    """Risk is flagged when Granite Guardian's normalized probability of a risky verdict
    meets or exceeds this value."""

    enable_thinking: bool = False
    """Whether to prompt Granite Guardian to reason before scoring (`<think>` block) rather
    than score immediately. Slower but can improve accuracy on subtler risks."""

    type: tuple[Literal['input', 'output', 'tool'], ...] = ('input', 'output')
    """Which guardrail points this risk applies to. See
    `AbstractSafetyCapability.type`."""

@dataclass
class GraniteGuardianRisk(AbstractSafetyCapability[AgentDepsT]):
    """Screens a prompt or response for one risk, via a direct request to Granite Guardian.

    Attach the same way any other safety capability is attached -- see
    :class:`~lightspeed.core.agent.safety.base.AbstractSafetyCapability` -- or call
    :meth:`evaluate` directly for a standalone check. Typically constructed via
    :class:`GraniteGuardian` rather than directly, so sibling risks in the same shield share
    one `model`.
    """

    risk: Risk = field(kw_only=True)
    """The risk to screen for -- its judging criteria, violation message, threshold, etc.
    `type` (see `AbstractSafetyCapability.type`) is copied from `risk.type` in
    `__post_init__`, so this is the only risk-specific input the constructor needs."""

    model: Optional[Model] = field(kw_only=True, default=None)
    """The Granite Guardian model to send the risk check to. Defaults to the run's own model
    -- see `for_run` -- so only needs setting explicitly for standalone use (calling
    `evaluate` without an agent) or to screen against a different model than the run
    answers with."""

    type: Sequence[Literal['input', 'output', 'tool']] = field(default_factory=list, init=False)
    """Which guardrail points this risk applies to; derived from `risk.type` in `__post_init__`.
    See `AbstractSafetyCapability.type`."""

    def __post_init__(self) -> None:
        """Adopt `risk.type` as this capability's own `type`, so `AbstractSafetyCapability`'s
        `wrap_model_request`/`wrap_run_event_stream` hooks screen the guardrail points `risk`
        asks for without `type` needing to be passed (and kept in sync) separately."""
        self.type = self.risk.type

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> GraniteGuardianRisk[AgentDepsT]:
        """Bind `model` to the run's own model when none was configured explicitly.

        Called once per run (see `AbstractCapability.for_run`), so this never mutates the
        shared, possibly-multi-run instance attached to the agent -- concurrent runs on
        different models each get their own bound copy.
        """
        if self.model is not None:
            return self
        if not isinstance(ctx.model, Model):
            raise UserError(
                f"GraniteGuardianRisk can't screen against {type(ctx.model).__name__}; pass "
                "`model=` explicitly to use it with a non-standard run model."
            )
        return dataclasses.replace(self, model=ctx.model)

    async def evaluate(self, prompt: str) -> GuardrailResult:
        """Score `prompt` against `risk.criteria` and return the verdict.

        Calls `model.request` directly (via `pydantic_ai.direct.model_request`) instead of
        running an `Agent`: the check needs no tools, output schema, or agent-graph
        machinery, just the one request/response pair scored by its logprobs.

        Raises:
            UserError: If `model` is `None`, i.e. `evaluate` is called directly without
                going through `for_run` first.
            UnexpectedModelBehavior: If the response is missing the `provider_details` or
                `logprobs` Granite Guardian's verdict is scored from.
        """
        if self.model is None:
            raise UserError(
                'GraniteGuardianRisk has no `model` to screen with. Pass `model=` when '
                "calling `evaluate` directly; attached to an agent, `for_run` fills this in "
                "from the run's own model automatically."
            )
        request = ModelRequest.user_text_prompt(
            prompt,
            instructions=build_guardian_block(self.risk.criteria, think=self.risk.enable_thinking),
        )
        response = await model_request(
            model=self.model,
            messages=[request],
            model_settings=OpenAIChatModelSettings(openai_logprobs=True, openai_top_logprobs=20),
        )
        if not response.provider_details:
            raise UnexpectedModelBehavior("No provider_details in Granite Guardian's response")

        logprobs = response.provider_details.get('logprobs')
        if not logprobs:
            raise UnexpectedModelBehavior("No logprobs field in Granite Guardian's provider_details")

        if is_safe(self.risk.threshold, logprobs):
            return GuardrailResult.allow()
        return GuardrailResult.block(message=self.risk.violation_message)

@dataclass
class GraniteGuardian(CombinedCapability[AgentDepsT]):
    """A ``granite_guardian`` shield: one :class:`GraniteGuardianRisk` per risk in `risks`.

    `capabilities` is derived from `risks` and `model` in `__post_init__`, rather than passed
    directly -- construct with `risks=`/`model=` instead of `capabilities=`.
    """

    risks: Sequence[Risk] = field(default_factory=list)
    """Risks to screen for. Each becomes its own `GraniteGuardianRisk` in `capabilities`."""

    model: Optional[Model] = field(kw_only=True, default=None)
    """The Granite Guardian model shared by every risk in `capabilities`. Defaults to the
    run's own model when left unset: each `GraniteGuardianRisk` built here gets `model=None`
    and independently binds to the run's model via its own `for_run` -- see
    `GraniteGuardianRisk.model` -- since `CombinedCapability.for_run` already calls `for_run`
    on every sub-capability."""

    output_check_interval_tokens: int = field(kw_only=True, default=50)
    """Passed through to each risk's `GraniteGuardianRisk.output_check_interval_tokens`."""

    capabilities: Sequence[AbstractCapability[AgentDepsT]] = field(default_factory=list, init=False)
    """Populated in `__post_init__` from `risks`; not meant to be passed directly."""

    def __post_init__(self) -> None:
        """Build one `GraniteGuardianRisk` per entry in `risks`, then let `CombinedCapability`
        normalize them."""
        prefix = f'{self.id}:' if self.id is not None else ''
        self.capabilities = [
            GraniteGuardianRisk(
                id=f'{prefix}{risk.name}',
                risk=risk,
                model=self.model,
                output_check_interval_tokens=self.output_check_interval_tokens,
            )
            for risk in self.risks
        ]
        super().__post_init__()
