"""Typed SSE payload bodies for the ``/v1/query`` streaming response.

Minimal rewrite of ``ORIGINIAL_LIGHTSPEED/src/models/common/agents/stream_payloads.py``.
Only the events this rewrite actually emits are modeled: ``start``, ``token``,
``turn_complete``, ``end``, and ``error``. Tool call/result and interrupt events
are not included since tools and stream interruption aren't wired into the
query endpoint yet.
"""

from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict

from lightspeed.app.models.responses.error.bases import AbstractErrorResponse


class StreamPayloadBase(BaseModel):
    """Base for streaming SSE JSON payloads."""

    model_config = ConfigDict(extra="forbid")

    def serialize_json(self) -> str:
        """Format this payload as an SSE ``data:`` line."""
        return f"data: {json.dumps(self.model_dump(mode='json'))}\n\n"


class StartEventData(BaseModel):
    """Payload for ``event: "start"``."""

    conversation_id: str


class StartStreamPayload(StreamPayloadBase):
    """SSE stream start body."""

    event: Literal["start"] = "start"
    data: StartEventData

    @classmethod
    def create(cls, *, conversation_id: str) -> Self:
        """Create a stream-start payload.

        Args:
            conversation_id: Conversation identifier for the stream.

        Returns:
            Start stream payload instance.
        """
        return cls(
            data=StartEventData(conversation_id=conversation_id)
        )


class TextEventData(BaseModel):
    """Structured data for text stream lines."""

    id: int
    text: str


class TextStreamPayload(StreamPayloadBase):
    """SSE token delta (``event: "token"``)."""

    event: Literal["text"] = "text"
    data: TextEventData

    @classmethod
    def create(cls, *, id: int, text: str) -> Self:
        """Create a token stream payload.

        Args:
            id: Monotonic chunk identifier for the token delta.
            token: Token text for the delta.

        Returns:
            Token stream payload instance.
        """
        return cls(data=TextEventData(id=id, text=text))

class EndEventData(BaseModel):
    """Nested data for ``event: "end"``."""

    output: str
    input_tokens: int
    output_tokens: int


class EndStreamPayload(StreamPayloadBase):
    """SSE end-of-stream body."""

    event: Literal["end"] = "end"
    data: EndEventData

    @classmethod
    def create(cls, *, output: str, input_tokens: int, output_tokens: int) -> Self:
        """Create an end-of-stream payload.

        Args:
            input_tokens: Input token count for the turn.
            output_tokens: Output token count for the turn.

        Returns:
            End stream payload instance.
        """
        return cls(
            data=EndEventData(output=output, input_tokens=input_tokens, output_tokens=output_tokens)
        )


class ErrorEventData(BaseModel):
    """Payload for ``event: "error"``."""

    status_code: int
    response: str
    cause: str


class ErrorStreamPayload(StreamPayloadBase):
    """SSE error event body (event + typed data)."""

    event: Literal["error"] = "error"
    data: ErrorEventData

    @classmethod
    def create(cls, *, status_code: int, response: str, cause: str) -> Self:
        """Create an error stream payload from HTTP error fields.

        Args:
            status_code: HTTP status code for the error.
            response: Short summary of the error.
            cause: Detailed explanation of the error cause.

        Returns:
            Error stream payload instance.
        """
        return cls(
            data=ErrorEventData(status_code=status_code, response=response, cause=cause)
        )

    @classmethod
    def from_error_response(cls, error_response: AbstractErrorResponse) -> Self:
        """Create an error stream payload from a structured API error response.

        Args:
            error_response: Structured error response model.

        Returns:
            Error stream payload instance.
        """
        return cls.create(
            status_code=error_response.status_code,
            response=error_response.detail.response,
            cause=error_response.detail.cause,
        )


StreamEventPayload = (
    StartStreamPayload
    | TextStreamPayload
    | EndStreamPayload
    | ErrorStreamPayload
)
