"""Structured HTTP error response models for OpenAPI documentation."""

from lightspeed.app.models.responses.error.bad_request import BadRequestResponse
from lightspeed.app.models.responses.error.bases import AbstractErrorResponse, DetailModel
from lightspeed.app.models.responses.error.conflict import ConflictResponse
from lightspeed.app.models.responses.error.content_too_large import (
    FileTooLargeResponse,
    PromptTooLongResponse,
)
from lightspeed.app.models.responses.error.forbidden import ForbiddenResponse
from lightspeed.app.models.responses.error.internal import InternalServerErrorResponse
from lightspeed.app.models.responses.error.not_found import NotFoundResponse
from lightspeed.app.models.responses.error.service_unavailable import (
    ServiceUnavailableResponse,
)
from lightspeed.app.models.responses.error.too_many_requests import QuotaExceededResponse
from lightspeed.app.models.responses.error.unauthorized import UnauthorizedResponse
from lightspeed.app.models.responses.error.unprocessable_entity import (
    UnprocessableEntityResponse,
)

__all__ = [
    "AbstractErrorResponse",
    "BadRequestResponse",
    "ConflictResponse",
    "DetailModel",
    "ForbiddenResponse",
    "InternalServerErrorResponse",
    "NotFoundResponse",
    "PromptTooLongResponse",
    "FileTooLargeResponse",
    "QuotaExceededResponse",
    "ServiceUnavailableResponse",
    "UnauthorizedResponse",
    "UnprocessableEntityResponse",
]
