"""Unit tests for :mod:`lightspeed.core.agent.safety.redaction.capability`."""

from __future__ import annotations

import asyncio

from lightspeed.core.agent.safety.redaction.capability import (
    DEFAULT_PATTERNS,
    DEFAULT_REPLACEMENT,
    RedactionCapability,
)


def run(coro):
    return asyncio.run(coro)


class TestRedactionCapabilityDefaults:
    """Tests for default construction of :class:`RedactionCapability`."""

    def test_defaults_to_input_type(self) -> None:
        capability = RedactionCapability()
        assert list(capability.type) == ["input"]

    def test_defaults_fill_patterns_and_replacement(self) -> None:
        capability = RedactionCapability()
        assert capability.patterns == dict(DEFAULT_PATTERNS)
        assert capability.replacement == DEFAULT_REPLACEMENT


class TestRedactionCapabilityEvaluate:
    """Tests for :meth:`RedactionCapability.evaluate`."""

    def test_no_match_allows(self) -> None:
        capability = RedactionCapability()
        result = run(capability.evaluate("nothing sensitive here"))
        assert result.action == "allow"

    def test_email_is_redacted(self) -> None:
        capability = RedactionCapability()
        result = run(capability.evaluate("contact me at jane.doe@example.com please"))
        assert result.action == "replace"
        assert "jane.doe@example.com" not in str(result.replacement)
        assert "[REDACTED]" in str(result.replacement)

    def test_ssn_is_redacted(self) -> None:
        capability = RedactionCapability()
        result = run(capability.evaluate("my ssn is 123-45-6789"))
        assert result.action == "replace"
        assert "123-45-6789" not in str(result.replacement)

    def test_custom_patterns_and_replacement(self) -> None:
        capability = RedactionCapability(patterns={"secret": r"secret-\d+"}, replacement="<hidden>")
        result = run(capability.evaluate("the code is secret-42"))
        assert result.action == "replace"
        assert result.replacement == "the code is <hidden>"

    def test_custom_patterns_do_not_use_defaults(self) -> None:
        capability = RedactionCapability(patterns={"secret": r"secret-\d+"})
        result = run(capability.evaluate("email me at a@b.com"))
        assert result.action == "allow"

    def test_multiple_matches_all_redacted(self) -> None:
        capability = RedactionCapability()
        result = run(capability.evaluate("emails: a@b.com and c@d.com"))
        assert str(result.replacement).count("[REDACTED]") == 2

    def test_output_type_supported(self) -> None:
        capability = RedactionCapability(type=["output"])
        result = run(capability.evaluate("leaking a@b.com"))
        assert result.action == "replace"
