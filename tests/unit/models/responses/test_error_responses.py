"""Unit tests for structured API error response models."""

from fastapi import status

from lightspeed.app.models.responses.error import (
    AbstractErrorResponse,
    DetailModel,
    InternalServerErrorResponse,
    NotFoundResponse,
    UnprocessableEntityResponse,
)


class TestNotFoundResponse:
    """Tests for :class:`NotFoundResponse`."""

    def test_with_resource_id(self) -> None:
        response = NotFoundResponse(resource="provider", resource_id="openai")
        assert isinstance(response, AbstractErrorResponse)
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert isinstance(response.detail, DetailModel)
        assert response.detail.response == "Provider not found"
        assert response.detail.cause == "Provider with ID openai does not exist"

    def test_model_dump_matches_http_exception_kwargs(self) -> None:
        response = NotFoundResponse(resource="model", resource_id="gpt-4o")
        dumped = response.model_dump()
        assert dumped == {
            "status_code": status.HTTP_404_NOT_FOUND,
            "detail": {
                "response": "Model not found",
                "cause": "Model with ID gpt-4o does not exist",
            },
        }


class TestUnprocessableEntityResponse:
    """Tests for :class:`UnprocessableEntityResponse`."""

    def test_constructor(self) -> None:
        response = UnprocessableEntityResponse(
            response="Missing required attributes",
            cause="Both provider and model are required",
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert response.detail.response == "Missing required attributes"
        assert response.detail.cause == "Both provider and model are required"


class TestInternalServerErrorResponse:
    """Tests for :class:`InternalServerErrorResponse`."""

    def test_generic(self) -> None:
        response = InternalServerErrorResponse.generic()
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.detail.response == "Internal server error"

    def test_query_failed(self) -> None:
        response = InternalServerErrorResponse.query_failed(
            cause="Failed to call backend API"
        )
        assert response.detail.response == "Error while processing query"
        assert response.detail.cause == "Failed to call backend API"

    def test_openapi_response_filters_examples(self) -> None:
        result = InternalServerErrorResponse.openapi_response(examples=["query"])
        assert result["model"] == InternalServerErrorResponse
        examples = result["content"]["application/json"]["examples"]
        assert list(examples) == ["query"]
        assert examples["query"]["value"]["detail"]["response"] == (
            "Error while processing query"
        )
