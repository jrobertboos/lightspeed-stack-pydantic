"""``RedactionCapability``: pattern-based text redaction for prompts and model output.

A minimal, standalone :class:`~lightspeed.core.agent.safety.base.AbstractSafetyCapability`:
unlike :class:`~lightspeed.core.agent.safety.question_validity.capability.QuestionValidityCapability`,
it never blocks a request -- it only ever rewrites the spans its patterns match, via
`GuardrailResult.replace`, so it composes safely alongside guards that do block without
competing with them over the same verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from pydantic_ai.tools import AgentDepsT

from lightspeed.core.agent.safety.base import AbstractSafetyCapability, GuardrailResult

DEFAULT_REPLACEMENT = "[REDACTED]"

DEFAULT_PATTERNS: Mapping[str, str] = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}
"""A small set of common PII patterns, used when `patterns` isn't overridden.

Not exhaustive -- swap in a stricter or looser set via `patterns` for anything
these miss or over-match (e.g. locale-specific phone numbers).
"""


@dataclass
class RedactionCapability(AbstractSafetyCapability[AgentDepsT]):
    """Rewrites text matching configured regex patterns to a placeholder.

    Attach the same way any other safety capability is attached -- see
    :class:`~lightspeed.core.agent.safety.base.AbstractSafetyCapability` -- or call
    :meth:`evaluate` directly for a standalone check. Guards both `'input'` and `'output'`
    by default, so PII typed by the user and PII a model happens to generate are both caught.
    """

    type: Sequence[Literal["input", "output"]] = field(default_factory=list)
    """Defaults to guarding both the user's prompt and the model's streamed output."""

    patterns: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_PATTERNS))
    """Named regex patterns to redact, keyed by a label used only for readability.

    Defaults to :data:`DEFAULT_PATTERNS`. Compiled once in :meth:`__post_init__`, so
    replacing this after construction (rather than passing it to `__init__`) won't take
    effect -- build a new instance instead.
    """

    replacement: str = DEFAULT_REPLACEMENT
    """Text substituted in place of each match."""

    def __post_init__(self) -> None:
        self._compiled: tuple[re.Pattern[str], ...] = tuple(
            re.compile(pattern) for pattern in self.patterns.values()
        )

    async def evaluate(self, prompt: str) -> GuardrailResult:
        """Redact every configured pattern's matches in `prompt`.

        Returns `GuardrailResult.allow()` unchanged when nothing matches, so a redaction
        guard with no hits never triggers `replace_latest_message`'s multimodal-prompt
        check on a request it had no reason to touch.
        """
        redacted = prompt
        for pattern in self._compiled:
            redacted = pattern.sub(self.replacement, redacted)
        if redacted == prompt:
            return GuardrailResult.allow()
        return GuardrailResult.replace(redacted)
