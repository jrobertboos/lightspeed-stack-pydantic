"""Unit tests for :mod:`lightspeed.core.agent.safety.base`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal, Sequence

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import SkipModelRequest, UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage

from lightspeed.core.agent.safety.base import (
    AbstractSafetyCapability,
    GuardrailResult,
    OutputBlocked,
    OutputState,
)


def run(coro):
    """Run a coroutine to completion without needing a pytest-asyncio plugin."""
    return asyncio.run(coro)


@dataclass
class FixedVerdictCapability(AbstractSafetyCapability):
    """A minimal concrete safety capability returning a pre-set verdict."""

    type: Sequence[Literal["input", "output", "tool"]] = field(default_factory=list)
    verdict: GuardrailResult = field(default_factory=GuardrailResult.allow)

    async def evaluate(self, prompt: str) -> GuardrailResult:
        return self.verdict


def make_run_context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def make_request_context(messages: list) -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


class TestOutputState:
    """Tests for :class:`OutputState`."""

    def test_append_and_text_accumulates_start_and_delta_events(self) -> None:
        state = OutputState()
        state.append(PartStartEvent(index=0, part=TextPart(content="Hello, ")))
        state.append(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world!")))
        assert state.text() == "Hello, world!"

    def test_text_ignores_non_text_events(self) -> None:
        state = OutputState()
        state.append(PartEndEvent(index=0, part=TextPart(content="ignored")))
        assert state.text() == ""

    def test_ready_when_interval_reached(self) -> None:
        state = OutputState()
        for _ in range(3):
            state.append(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="x")))
        assert state.ready(3) is True
        assert state.ready(4) is False

    def test_ready_when_part_ends_even_below_interval(self) -> None:
        state = OutputState()
        state.append(PartEndEvent(index=0, part=TextPart(content="x")))
        assert state.ready(50) is True

    def test_ready_false_when_empty(self) -> None:
        assert OutputState().ready(1) is False

    def test_flush_returns_and_clears_buffer(self) -> None:
        state = OutputState()
        event = PartStartEvent(index=0, part=TextPart(content="hi"))
        state.append(event)
        text, events = state.flush()
        assert text == "hi"
        assert events == [event]
        assert state.events == []
        assert state.text() == ""


class TestGuardrailResult:
    """Tests for :class:`GuardrailResult` classmethods."""

    def test_allow(self) -> None:
        result = GuardrailResult.allow()
        assert result.action == "allow"
        assert result.message is None

    def test_block_with_message(self) -> None:
        result = GuardrailResult.block("nope")
        assert result.action == "block"
        assert result.message == "nope"

    def test_block_without_message(self) -> None:
        result = GuardrailResult.block()
        assert result.action == "block"
        assert result.message is None

    def test_replace(self) -> None:
        result = GuardrailResult.replace("redacted text")
        assert result.action == "replace"
        assert result.replacement == "redacted text"


class TestWrapModelRequest:
    """Tests for :meth:`AbstractSafetyCapability.wrap_model_request`."""

    def test_ignores_prompt_when_input_not_in_type(self) -> None:
        capability = FixedVerdictCapability(type=[], verdict=GuardrailResult.block("blocked"))
        request_context = make_request_context(
            [ModelRequest(parts=[UserPromptPart(content="hello")])]
        )

        async def handler(ctx):
            return ModelResponse(parts=[TextPart(content="ok")])

        response = run(
            capability.wrap_model_request(
                make_run_context(), request_context=request_context, handler=handler
            )
        )
        assert response.parts[0].content == "ok"

    def test_allow_calls_handler(self) -> None:
        capability = FixedVerdictCapability(type=["input"], verdict=GuardrailResult.allow())
        request_context = make_request_context(
            [ModelRequest(parts=[UserPromptPart(content="hello")])]
        )

        async def handler(ctx):
            return ModelResponse(parts=[TextPart(content="handled")])

        response = run(
            capability.wrap_model_request(
                make_run_context(), request_context=request_context, handler=handler
            )
        )
        assert response.parts[0].content == "handled"

    def test_block_raises_skip_model_request_with_message(self) -> None:
        capability = FixedVerdictCapability(type=["input"], verdict=GuardrailResult.block("blocked!"))
        request_context = make_request_context(
            [ModelRequest(parts=[UserPromptPart(content="hello")])]
        )

        async def handler(ctx):  # pragma: no cover - must not be called
            raise AssertionError("handler should not be called when blocked")

        with pytest.raises(SkipModelRequest) as exc_info:
            run(
                capability.wrap_model_request(
                    make_run_context(), request_context=request_context, handler=handler
                )
            )
        assert exc_info.value.response.parts[0].content == "blocked!"

    def test_replace_rewrites_latest_user_prompt(self) -> None:
        capability = FixedVerdictCapability(type=["input"], verdict=GuardrailResult.replace("cleaned"))
        request_context = make_request_context(
            [ModelRequest(parts=[UserPromptPart(content="secret")])]
        )

        async def handler(ctx):
            return ctx.messages[-1].parts[-1].content

        result = run(
            capability.wrap_model_request(
                make_run_context(), request_context=request_context, handler=handler
            )
        )
        assert result == "cleaned"

    def test_unsupported_action_raises_user_error(self) -> None:
        capability = FixedVerdictCapability(
            type=["input"], verdict=GuardrailResult(action="bogus")  # type: ignore[arg-type]
        )
        request_context = make_request_context(
            [ModelRequest(parts=[UserPromptPart(content="hello")])]
        )

        async def handler(ctx):  # pragma: no cover
            raise AssertionError("handler should not be called")

        with pytest.raises(UserError):
            run(
                capability.wrap_model_request(
                    make_run_context(), request_context=request_context, handler=handler
                )
            )

    def test_no_prompt_still_allows_through(self) -> None:
        capability = FixedVerdictCapability(type=["input"], verdict=GuardrailResult.block("nope"))
        request_context = make_request_context([ModelRequest(parts=[])])

        async def handler(ctx):
            return "handled"

        result = run(
            capability.wrap_model_request(
                make_run_context(), request_context=request_context, handler=handler
            )
        )
        assert result == "handled"


class TestWrapRun:
    """Tests for :meth:`AbstractSafetyCapability.wrap_run`."""

    def test_ignores_output_when_output_not_in_type(self) -> None:
        capability = FixedVerdictCapability(type=[], verdict=GuardrailResult.block("blocked"))

        async def handler():
            return AgentRunResult(output="original")

        result = run(capability.wrap_run(make_run_context(), handler=handler))
        assert result.output == "original"

    def test_allow_keeps_original_output(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.allow())

        async def handler():
            return AgentRunResult(output="original")

        result = run(capability.wrap_run(make_run_context(), handler=handler))
        assert result.output == "original"

    def test_block_replaces_output_with_message(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.block("blocked!"))

        async def handler():
            return AgentRunResult(output="original")

        result = run(capability.wrap_run(make_run_context(), handler=handler))
        assert result.output == "blocked!"

    def test_block_without_message_uses_default(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.block())

        async def handler():
            return AgentRunResult(output="original")

        result = run(capability.wrap_run(make_run_context(), handler=handler))
        assert result.output == "Response blocked by a safety guard."

    def test_replace_rewrites_output(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.replace("redacted"))

        async def handler():
            return AgentRunResult(output="original")

        result = run(capability.wrap_run(make_run_context(), handler=handler))
        assert result.output == "redacted"

    def test_unsupported_action_raises_user_error(self) -> None:
        capability = FixedVerdictCapability(
            type=["output"], verdict=GuardrailResult(action="bogus")  # type: ignore[arg-type]
        )

        async def handler():
            return AgentRunResult(output="original")

        with pytest.raises(UserError):
            run(capability.wrap_run(make_run_context(), handler=handler))


class TestWrapRunEventStream:
    """Tests for :meth:`AbstractSafetyCapability.wrap_run_event_stream`."""

    async def _collect(self, capability, events):
        async def stream():
            for event in events:
                yield event

        collected = []
        async for event in capability.wrap_run_event_stream(make_run_context(), stream=stream()):
            collected.append(event)
        return collected

    def test_passes_through_when_output_not_in_type(self) -> None:
        capability = FixedVerdictCapability(type=[], verdict=GuardrailResult.block("blocked"))
        events = [PartStartEvent(index=0, part=TextPart(content="hi"))]
        collected = run(self._collect(capability, events))
        assert collected == events

    def test_allow_releases_buffered_events_once_ready(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.allow())
        event = PartEndEvent(index=0, part=TextPart(content="hi"))
        collected = run(self._collect(capability, [event]))
        assert collected == [event]

    def test_replace_yields_replacement_text_part(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.replace("redacted"))
        event = PartEndEvent(index=0, part=TextPart(content="secret"))
        collected = run(self._collect(capability, [event]))
        assert len(collected) == 1
        assert isinstance(collected[0], PartStartEvent)
        assert collected[0].part.content == "redacted"

    def test_block_raises_output_blocked_and_drains_stream(self) -> None:
        capability = FixedVerdictCapability(type=["output"], verdict=GuardrailResult.block("blocked!"))
        event = PartEndEvent(index=0, part=TextPart(content="secret"))
        with pytest.raises(OutputBlocked) as exc_info:
            run(self._collect(capability, [event]))
        assert exc_info.value.reason == "blocked!"

    def test_not_ready_events_stay_buffered(self) -> None:
        """A single delta event (no part-end) below the interval releases nothing."""
        capability = FixedVerdictCapability(
            type=["output"], verdict=GuardrailResult.block("should not fire")
        )
        capability.output_check_interval_tokens = 50
        event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="x"))
        collected = run(self._collect(capability, [event]))
        assert collected == []


class TestOnRunError:
    """Tests for :meth:`AbstractSafetyCapability.on_run_error`."""

    def test_recovers_output_blocked_into_run_result(self) -> None:
        capability = FixedVerdictCapability(type=["output"])
        ctx = make_run_context()
        result = run(capability.on_run_error(ctx, error=OutputBlocked("blocked reason")))
        assert isinstance(result, AgentRunResult)
        assert result.output == "blocked reason"

    def test_recovers_output_blocked_without_reason_uses_default(self) -> None:
        capability = FixedVerdictCapability(type=["output"])
        ctx = make_run_context()
        result = run(capability.on_run_error(ctx, error=OutputBlocked()))
        assert result.output == "Response blocked by a safety guard."

    def test_reraises_other_errors(self) -> None:
        capability = FixedVerdictCapability(type=["output"])
        ctx = make_run_context()
        with pytest.raises(ValueError):
            run(capability.on_run_error(ctx, error=ValueError("boom")))
