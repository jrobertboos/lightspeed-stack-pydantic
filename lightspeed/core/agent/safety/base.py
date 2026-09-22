"""Abstract base for safety capabilities with a standalone run interface."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

from pydantic_ai.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
)
from pydantic_ai.exceptions import SkipModelRequest, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai import RunContext
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.run import AgentRunResult

from lightspeed.core.agent.utils import extract_latest_message_text, replace_latest_message


class OutputBlocked(BaseException):
    """The output was blocked by a safety guard."""

    def __init__(self, reason: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason

@dataclass
class OutputState:
    """Per-run buffer of model-response stream events awaiting an `evaluate` check.

    `AbstractSafetyCapability` gives each run its own instance (via `for_run`), so concurrent
    runs sharing a capability instance don't share a buffer.
    """

    events: list[AgentStreamEvent] = field(default_factory=list)
    """Stream events accumulated since the last check."""

    def append(self, event: AgentStreamEvent) -> None:
        """Buffer a stream event."""
        self.events.append(event)

    def ready(self, interval: int) -> bool:
        """Whether enough events have accumulated to run the next check.

        Also ready as soon as a part finishes, so a completed part is checked promptly instead of
        waiting on the next part's events to cross `interval`.
        """
        if self.events and isinstance(self.events[-1], PartEndEvent):
            return True
        return len(self.events) >= interval

    def text(self) -> str:
        """Return the buffered text."""
        chunks: list[str] = []
        for event in self.events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                chunks.append(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                chunks.append(event.delta.content_delta)
        return ''.join(chunks)

    def flush(self) -> tuple[str, list[AgentStreamEvent]]:
        """Return the buffered text and events, then clear the buffer."""
        text, events = self.text(), self.events[:]
        self.events.clear()
        return text, events


@dataclass
class GuardrailResult:
    """The verdict a safety guard returns for a prompt or a response."""

    action: Literal['allow', 'block', 'replace']
    message: Optional[str] = None
    replacement: Optional[object] = None

    @classmethod
    def allow(cls) -> GuardrailResult:
        return cls(action='allow')

    @classmethod
    def block(cls, message: str | None = None) -> GuardrailResult:
        return cls(action='block', message=message)

    @classmethod
    def replace(cls, value: object) -> GuardrailResult:
        return cls(action='replace', replacement=value)


@dataclass
class AbstractSafetyCapability(AbstractCapability[AgentDepsT]):
    """Interface for safety/moderation that can be called directly."""

    type: Iterable[Literal['input', 'output', 'tool']]

    output_check_interval_tokens: int = 50
    """How often (in approximate output tokens) to re-run `evaluate` on streamed output.

    Only used when `'output'` is in `type`. Text generated since the last check is buffered and
    withheld from the caller until it clears `evaluate`, so a violation stops the response before
    the flagged content -- or anything after it -- is ever released.
    """

    @abstractmethod
    async def evaluate(self, prompt: str) -> GuardrailResult:
        """Evaluate safety on prompt"""

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Recover a run blocked by an output safety guard.

        `wrap_run_event_stream` raises `OutputBlocked` when `evaluate` blocks streamed output;
        that error propagates up through the run and lands here. Convert it into a successful
        `AgentRunResult` carrying the guard's message as the output, instead of letting the
        exception surface to the caller. Any other error is re-raised unchanged.
        """
        if isinstance(error, OutputBlocked):
            return AgentRunResult(
                output=error.reason or 'Response blocked by a safety guard.',
                _state=GraphAgentState(
                    message_history=ctx.messages,
                    usage=ctx.usage,
                    run_id=ctx.run_id,
                    conversation_id=ctx.conversation_id,
                    metadata=ctx.metadata,
                ),
            )
        raise error

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Evaluate the buffered output once `state` is ready, and act on the verdict."""

        async def release(state: OutputState) -> list[AgentStreamEvent]:
            text, events = state.flush()
            result = await self.evaluate(text)
            match result.action:
                case 'allow':
                    return events
                case 'block':
                    raise OutputBlocked(result.message)
                case 'replace':
                    return [PartStartEvent(index=0, part=TextPart(content=str(result.replacement)))]

        async def drain() -> None:
            """Exhaust `stream` without processing it further.

            Called after a block verdict instead of cancelling the stream, so the model still
            finishes its request and the provider's trailing usage chunk lands in `ctx.usage` --
            cancelling here would abandon the request before that arrives, leaving usage at zero.
            """
            async for _ in stream:
                pass

        if 'output' in self.type:
            state: OutputState = OutputState()
            try:
                async for event in stream:
                    state.append(event)
                    if state.ready(self.output_check_interval_tokens):
                        for released in await release(state):
                            yield released
            except OutputBlocked:
                await drain()
                raise
            finally:
                aclose = getattr(stream, 'aclose', None)
                if aclose is not None:
                    await aclose()
        else:
            async for event in stream:
                yield event

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Wrap the model request with safety, guarding the user's prompt."""
        if 'input' in self.type:
            prompt = extract_latest_message_text(request_context.messages)
            result = await self.evaluate(prompt) if prompt is not None else GuardrailResult.allow()
            match result.action:
                case 'allow':
                    pass
                case 'block':
                    raise SkipModelRequest(ModelResponse(parts=[TextPart(content=result.message or '')]))
                case 'replace':
                    if not replace_latest_message(request_context.messages, str(result.replacement)):
                        raise UserError('A Safety guard could not find a user prompt to redact in the request.')
                case _:
                    raise UserError(
                        f'A Safety guard cannot return {result.action} on input; use allow, block, or replace.'
                    )

        return await handler(request_context)