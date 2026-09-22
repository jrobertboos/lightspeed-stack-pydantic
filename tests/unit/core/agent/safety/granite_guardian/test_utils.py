"""Unit tests for :mod:`lightspeed.core.agent.safety.granite_guardian.utils`."""

from __future__ import annotations

from math import log

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from lightspeed.core.agent.safety.granite_guardian.utils import build_guardian_block, is_safe


def _logprob(token: str, top: list[tuple[str, float]]) -> dict:
    """Build a raw logprob dict for one response token, shaped like the OpenAI API."""
    return {
        "token": token,
        "logprob": max((p[1] for p in top if p[0] == token), default=-10.0),
        "bytes": None,
        "top_logprobs": [
            {"token": tok, "logprob": lp, "bytes": None} for tok, lp in top
        ],
    }


def _score_response(
    verdict_token: str, *, safe_logprob: float = log(0.9), risky_logprob: float = log(0.1), think: bool = False
) -> list[dict]:
    """Build a full token stream: optional <think></think>, then <score>{verdict}</score>."""
    top = [("no", safe_logprob), ("yes", risky_logprob)]
    tokens = []
    if think:
        tokens += [_logprob(tok, []) for tok in ["<think>", "reasoning", "</think>"]]
    else:
        tokens += [_logprob("<think>", []), _logprob("</think>", [])]
    tokens.append(_logprob("<score>", []))
    tokens.append(_logprob(verdict_token, top))
    tokens.append(_logprob("</score>", []))
    return tokens


class TestBuildGuardianBlock:
    """Tests for :func:`build_guardian_block`."""

    def test_includes_criteria(self) -> None:
        block = build_guardian_block("flags jailbreak attempts")
        assert "flags jailbreak attempts" in block

    def test_no_think_uses_nothink_preamble(self) -> None:
        block = build_guardian_block("criteria", think=False)
        assert "<no-think>" in block

    def test_think_uses_think_preamble(self) -> None:
        block = build_guardian_block("criteria", think=True)
        assert "<no-think>" not in block
        assert "reason through the evidence" in block.lower() or "<think>" in block


class TestIsSafe:
    """Tests for :func:`is_safe`."""

    def test_safe_verdict_below_threshold(self) -> None:
        logprobs = _score_response("no", safe_logprob=log(0.95), risky_logprob=log(0.05))
        assert is_safe(0.5, logprobs) is True

    def test_risky_verdict_above_threshold(self) -> None:
        logprobs = _score_response("yes", safe_logprob=log(0.1), risky_logprob=log(0.9))
        assert is_safe(0.5, logprobs) is False

    def test_threshold_boundary_is_exclusive_of_safety(self) -> None:
        # risky probability exactly at threshold is considered unsafe (>= threshold).
        logprobs = _score_response("yes", safe_logprob=log(0.5), risky_logprob=log(0.5))
        assert is_safe(0.5, logprobs) is False

    def test_works_with_thinking_enabled(self) -> None:
        logprobs = _score_response("no", safe_logprob=log(0.9), risky_logprob=log(0.1), think=True)
        assert is_safe(0.5, logprobs) is True

    def test_missing_score_tag_raises(self) -> None:
        logprobs = [_logprob("hello", [])]
        with pytest.raises(UnexpectedModelBehavior):
            is_safe(0.5, logprobs)

    def test_no_safe_or_risky_token_raises_underflow(self) -> None:
        top = [("maybe", log(0.5))]
        logprobs = [
            _logprob("<think>", []),
            _logprob("</think>", []),
            _logprob("<score>", []),
            _logprob("maybe", top),
            _logprob("</score>", []),
        ]
        with pytest.raises(UnexpectedModelBehavior):
            is_safe(0.5, logprobs)
