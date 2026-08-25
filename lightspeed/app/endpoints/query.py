"""Handler for REST API call to provide an answer to a query.

Minimal rewrite of the original Lightspeed ``/query`` endpoint. Inference goes
through :meth:`~lightspeed.core.agent.factory.AgentFactory.create_agent` and
pydantic-ai ``Agent.run``. Flows that are not yet implemented are left as
comments.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic_ai.run import AgentRunResult

from lightspeed.app.models.config import Configuration
from lightspeed.app.models.requests.query import QueryRequest
from lightspeed.app.models.responses.error import (
    InternalServerErrorResponse,
    NotFoundResponse,
    UnprocessableEntityResponse,
)
from lightspeed.app.models.responses.success.query import QueryResponse
from lightspeed.core.agent.factory import AgentFactory
from lightspeed.core.providers.registry import ProviderRegistry
from lightspeed.core.config import LogicError
from lightspeed.core.config import configuration as app_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

query_response: dict[int | str, dict[str, Any]] = {
    404: NotFoundResponse.openapi_response(
        examples=["conversation", "model", "provider"]
    ),
    422: UnprocessableEntityResponse.openapi_response(),
    500: InternalServerErrorResponse.openapi_response(
        examples=["configuration", "query"]
    ),
    # Not yet raised by this minimal rewrite:
    # 401: UnauthorizedResponse.openapi_response(...)
    # 403: ForbiddenResponse.openapi_response(...)
    # 413: PromptTooLongResponse.openapi_response(...)
    # 429: QuotaExceededResponse.openapi_response()
    # 503: ServiceUnavailableResponse.openapi_response(...)
}


@router.post(
    "/query",
    responses=query_response,
    summary="Query Endpoint Handler",
)
# @authorize(Action.QUERY)  # authorization not yet implemented
async def query_endpoint_handler(
    query_request: QueryRequest,
    # auth: Annotated[AuthTuple, Depends(get_auth_dependency())],  # auth not yet implemented
    # mcp_headers: McpHeaders = Depends(mcp_headers_dependency),  # MCP headers not yet implemented
) -> QueryResponse:
    """Handle ``POST /query`` using a pydantic-ai agent.

    Implemented today:
        - Resolve ``Configuration`` from the process-wide singleton
        - ``AgentFactory.create_agent`` + ``agent.run`` for the user query
        - Return a minimal :class:`QueryResponse`
        - Structured error responses (``detail.response`` / ``detail.cause``)

    Not yet implemented (see commented steps below):
        - Auth / authorization / MCP OAuth
        - Quota checks and token accounting
        - Conversation load/store, compaction, topic summary
        - Shield moderation, RAG, tools/MCP
    """

    _get_configuration()  # fail fast with 500 if the app wasn't configured
    provider_name, model_name = _resolve_provider_and_model(query_request)

    try:
        agent = AgentFactory.create_agent(
            provider=provider_name,
            model=model_name,
            instructions=query_request.system_prompt,
        )
        run_result = await agent.run(query_request.query)
    except KeyError as exc:
        if provider_name not in ProviderRegistry():
            error_response = NotFoundResponse(
                resource="provider",
                resource_id=provider_name,
            )
        else:
            error_response = NotFoundResponse(
                resource="model",
                resource_id=model_name,
            )
        raise HTTPException(**error_response.model_dump()) from exc
    except ValueError as exc:
        error_response = UnprocessableEntityResponse(
            response="Invalid attribute value",
            cause=str(exc),
        )
        raise HTTPException(**error_response.model_dump()) from exc
    except Exception as exc:
        # map_agent_inference_error(...) / handle_known_apistatus_errors(...) not yet implemented
        logger.exception("Query agent run failed")
        error_response = InternalServerErrorResponse.query_failed(
            cause="Failed to call backend API",
        )
        raise HTTPException(**error_response.model_dump()) from exc

    # topic_summary = await maybe_get_topic_summary(...)
    # consume_query_tokens(...)
    # available_quotas = get_available_quotas(...)
    # store_query_results(...)

    return _build_query_response(
        run_result,
        conversation_id=query_request.conversation_id,
    )


def _build_query_response(
    run_result: AgentRunResult[str],
    *,
    conversation_id: Optional[str] = None,
    # available_quotas: dict[str, int] | None = None,  # quota limiters not yet implemented
) -> QueryResponse:
    """Map a completed agent run to a :class:`QueryResponse`.

    Minimal today (output text + token usage). Extend here as tools, RAG,
    and finish-reason handling land.
    """
    # finish_reason = get_agent_finish_reason(run_result.response)
    # if finish_reason != AgentFinishReason.SUCCESS: ...
    # tool_calls / tool_results / rag_chunks / referenced_documents from run_result.new_messages()

    usage = run_result.usage
    return QueryResponse(
        conversation_id=conversation_id,
        response=run_result.output,
        truncated=False,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        available_quotas={},
    )


def _get_configuration() -> Configuration:
    """Return the loaded :class:`Configuration` from the process-wide singleton.

    The FastAPI ``lifespan`` (see :mod:`lightspeed.app.main`) loads it before
    the app starts serving traffic, so this should only raise if called
    outside of a running app (e.g. a unit test that imports this module
    without loading configuration first).

    Raises:
        HTTPException: 500 if the configuration hasn't been loaded yet.
    """
    try:
        return app_config.configuration
    except LogicError as exc:
        error_response = InternalServerErrorResponse.configuration_not_loaded()
        raise HTTPException(**error_response.model_dump()) from exc


def _resolve_provider_and_model(query_request: QueryRequest) -> tuple[str, str]:
    """Resolve provider/model for this request.

    Today the request must supply both. Original selection order (conversation
    last-used → configured defaults → first available model) is not implemented.
    """
    # request_model = f"{query_request.provider}/{query_request.model}" if ...
    # model = await select_model_for_responses(request_model, client, user_conversation)

    if not query_request.provider or not query_request.model:
        error_response = UnprocessableEntityResponse(
            response="Missing required attributes",
            cause=(
                "Both provider and model are required until default "
                "model selection is implemented"
            ),
        )
        raise HTTPException(**error_response.model_dump())
    return query_request.provider, query_request.model
