"""Agent streaming helpers for the ``/v1/query`` SSE flow.

Minimal rewrite of ``ORIGINIAL_LIGHTSPEED/src/utils/agents/streaming.py``: an
``AgentTurnAccumulator``-style mutable state object plus a ``singledispatch``
event dispatcher that maps pydantic-ai stream events onto SSE payloads. Only
text token streaming is handled today -- tool call/result dispatch, RAG,
quota, and persistence hooks aren't wired into the query endpoint yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Optional, Union
import uuid

from pydantic_ai import (
    AgentRunResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.messages import AgentStreamEvent

from lightspeed.app.models.responses.success.stream import (
    EndStreamPayload,
    StreamEventPayload,
    TextStreamPayload,
)

logger = logging.getLogger(__name__)

AgentDispatchEvent = Union[AgentStreamEvent, AgentRunResultEvent]


@dataclass(slots=True)
class StreamState:
    """Mutable per-turn state for streaming response processing.

    Internal to the streaming dispatch machinery -- it only tracks the SSE
    chunk sequence and buffered text needed to build payloads. The final
    :class:`~lightspeed.core.agent.schemas.AgentQuery` result handed back to
    callers is tracked separately, so it stays independent of this type.

    Attributes:
        chunk_id: Monotonic SSE chunk index.
        text_parts: Buffered text deltas before ``turn_complete``.
        conversation_id: Conversation id for the stream; generated when omitted.
    """

    chunk_id: int = 0
    text_parts: list[str] = field(default_factory=list)
    conversation_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.conversation_id is None:
            self.conversation_id = str(uuid.uuid4())


def process_text(state: StreamState, text: str) -> TextStreamPayload:
    """Append text to state and build a token stream payload.

    Args:
        state: Mutable dispatch state.
        text: Token text to append and emit.

    Returns:
        Token stream payload containing the emitted token chunk.
    """
    state.text_parts.append(text)
    payload = TextStreamPayload.create(id=state.chunk_id, text=text)
    state.chunk_id += 1
    return payload


@singledispatch
def dispatch_stream_event(
    event: AgentDispatchEvent,
    _state: StreamState,
) -> Optional[StreamEventPayload]:
    """Map a pydantic-ai stream event to an SSE payload.

    Args:
        event: Agent stream event emitted by the runtime.
        _state: Mutable state for stream processing.

    Returns:
        None when the event does not map to an SSE payload.
    """
    logger.debug("Ignoring stream event of type %s", type(event).__name__)
    return None


@dispatch_stream_event.register
def _(event: AgentRunResultEvent, state: StreamState) -> Optional[StreamEventPayload]:
    """Handle the final run result event and emit the completion payload."""
    return EndStreamPayload.create(
        input_tokens=event.result.usage.input_tokens,
        output_tokens=event.result.usage.output_tokens,
        output=event.result.output,
    )

@dispatch_stream_event.register
def _(event: PartStartEvent, state: StreamState) -> Optional[StreamEventPayload]:
    """Handle the start of a model response part."""
    if isinstance(event.part, TextPart):
        return process_text(state, event.part.content)
    logger.debug("Ignoring part start kind=%s", event.part.part_kind)
    return None


@dispatch_stream_event.register
def _(event: PartDeltaEvent, state: StreamState) -> Optional[StreamEventPayload]:
    """Handle an incremental update to a model response part."""
    if isinstance(event.delta, TextPartDelta):
        return process_text(state, event.delta.content_delta)
    logger.debug("Ignoring part delta kind=%s", event.delta.part_delta_kind)
    return None