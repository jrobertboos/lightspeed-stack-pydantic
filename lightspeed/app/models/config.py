"""Configuration models for Lightspeed Stack."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SecretStr,
    field_validator,
)

from lightspeed.core.agent.safety.granite_guardian.capability import Risk

# Supported inference provider types.
ProviderType = Literal[
    "openai",
    "azure",
    "bedrock",
    "vertexai",
    "watsonx",
    "vllm",
]


class ConfigurationBase(BaseModel):
    """Base class for configuration models that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class ProviderConfiguration(ConfigurationBase):
    """Client-side LLM provider entry.

    ``type`` selects a supported Lightspeed provider. The registry maps each
    type onto a pydantic-ai :class:`~pydantic_ai.providers.Provider`.

    YAML shape::

        providers:
          - name: my-openai
            type: openai
            url: https://api.openai.com/v1
            api_key: sk-...
    """

    name: str = Field(
        ...,
        title="Provider name",
        description="Unique identifier used to look up this provider in the registry.",
        min_length=1,
    )

    type: ProviderType = Field(
        ...,
        title="Provider type",
        description=(
            "Supported provider type: openai, azure, bedrock, vertexai, watsonx, or vllm."
        ),
    )

    url: Optional[AnyHttpUrl] = Field(
        None,
        title="Base URL",
        description=(
            "Optional base URL for the provider API. Mapped to the underlying "
            "pydantic-ai provider as base_url, api_base, or azure_endpoint."
        ),
    )

    api_key: Optional[SecretStr] = Field(
        None,
        title="API key",
        description=(
            "Optional API key. When omitted, the underlying pydantic-ai provider "
            "falls back to its standard environment variables."
        ),
    )


class MCPServerConfiguration(ConfigurationBase):
    """MCP (Model Context Protocol) server entry.

    Each entry becomes an ``MCP`` capability (see
    :class:`~lightspeed.core.agent.tools.mcp.factory.MCPCapabilityFactory`)
    that gives the agent tools beyond what a provider's model exposes
    natively. Only servers listed here are available to agents.

    YAML shape::

        mcp_servers:
          - name: docs-search
            url: http://docs-mcp:8000
            timeout: 30
            forward_headers:
              - x-rh-identity
    """

    name: str = Field(
        ...,
        title="Server name",
        description="Unique identifier for this MCP server.",
        min_length=1,
    )

    url: AnyHttpUrl = Field(
        ...,
        title="Server URL",
        description="URL of the MCP server.",
    )

    timeout: Optional[PositiveInt] = Field(
        None,
        title="Timeout",
        description="Request timeout in seconds.",
    )

    forward_headers: list[str] = Field(
        default_factory=list,
        title="Forwarded headers",
        description=(
            "Header names forwarded verbatim from the incoming client "
            "request to this MCP server (not yet implemented -- needs "
            "per-request context that isn't wired up yet)."
        ),
    )


class SkillsConfiguration(ConfigurationBase):
    """Agent skills configuration.

    Each path is a skill library: a directory whose immediate children
    contain ``SKILL.md``. All paths become a single
    :class:`~pydantic_ai_harness.skills.Skills` capability (see
    :class:`~lightspeed.core.agent.tools.skills.factory.SkillsCapabilityFactory`).

    YAML shape::

        skills:
          paths:
            - /var/skills
            - /opt/custom-skills
    """

    paths: list[Path] = Field(
        default_factory=list,
        title="Skill paths",
        description="Paths to skill libraries (directories of skill packages).",
    )


class QuestionValidityConfig(ConfigurationBase):
    """``config`` for a ``question_validity`` safety shield.

    Mirrors the constructor arguments of
    :class:`~lightspeed.core.agent.safety.question_validity.capability.QuestionValidityCapability`.
    Fields left unset (``None``) fall back to that capability's own defaults, so nothing here
    needs to duplicate them.
    """

    model: Optional[str] = Field(
        None,
        title="Model",
        description=(
            "Model used to classify prompts, as `<provider>:<model>` (provider registry name "
            "and model id, resolved at build time). Omit to classify against the run's own "
            "model."
        ),
        examples=["openai:gpt-4o", "my-vllm:llama-3.1-8b"],
    )

    invalid_question_response: Optional[str] = Field(
        None,
        title="Invalid question response",
        description="Message returned to the caller in place of a rejected prompt.",
    )

    classifier_instructions: Optional[str] = Field(
        None,
        title="Classifier instructions",
        description="System prompt sent with the classification request used to judge a prompt.",
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: Optional[str]) -> Optional[str]:
        """Ensure `model` (when set) has the `<provider>:<model>` shape."""
        if value is None:
            return value
        provider, _, model = value.partition(":")
        if not provider or not model:
            raise ValueError(f"model must be `<provider>:<model>`, got {value!r}")
        return value


class RedactionConfig(ConfigurationBase):
    """``config`` for a ``redaction`` safety shield.

    Mirrors the constructor arguments of
    :class:`~lightspeed.core.agent.safety.redaction.capability.RedactionCapability`.
    Fields left unset (``None``) fall back to that capability's own defaults, so nothing here
    needs to duplicate them.
    """

    patterns: Optional[dict[str, str]] = Field(
        None,
        title="Patterns",
        description=(
            "Named regex patterns to redact, keyed by a label used only for readability. "
            "Defaults to a small set of common PII patterns (email, SSN, credit card, phone)."
        ),
    )

    replacement: Optional[str] = Field(
        None,
        title="Replacement",
        description="Text substituted in place of each match.",
    )


class GraniteGuardianConfig(ConfigurationBase):
    """``config`` for a ``granite_guardian`` safety shield.

    Describes the Granite Guardian model to screen with, plus the risks to screen for.
    ``risks`` entries are :class:`~lightspeed.core.agent.safety.granite_guardian.capability.Risk`
    directly (pydantic validates YAML/JSON mappings onto it like any other field), rather than
    a separate config/schema type -- there's nothing this schema would otherwise add. Mapped
    onto a :class:`~lightspeed.core.agent.safety.granite_guardian.capability.GraniteGuardian`
    by :class:`~lightspeed.core.agent.safety.factory.SafetyCapabilityFactory`, which resolves
    `model` via `ProviderRegistry`.
    """

    model: str = Field(
        title="Model",
        description=(
            "Granite Guardian model to screen with, as `<provider>:<model>` (provider "
            "registry name and model id, resolved at build time). Required -- unlike "
            "`question_validity.model`, there's no run-model fallback."
        ),
        examples=["watsonx:granite-guardian-3-8b", "my-vllm:granite-guardian"],
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Ensure `model` has the `<provider>:<model>` shape."""
        provider, _, model = value.partition(":")
        if not provider or not model:
            raise ValueError(f"model must be `<provider>:<model>`, got {value!r}")
        return value

    output_check_interval_tokens: int = Field(
        50,
        title="Output check interval",
        description=(
            "How often (in approximate output events) to re-run each risk's check on "
            "streamed output. See `AbstractSafetyCapability.output_check_interval_tokens`."
        ),
        gt=0,
    )

    risks: list[Risk] = Field(
        ...,
        title="Risks",
        description="Risks to screen for. Each becomes its own capability.",
        min_length=1,
    )


class QuestionValiditySafetyConfiguration(ConfigurationBase):
    """A ``question_validity`` entry in ``safety``: an LLM-classified on/off-topic guard."""

    name: str = Field(
        ...,
        title="Shield name",
        description="Unique identifier for this safety shield.",
        min_length=1,
    )

    type: Literal["question_validity"] = Field(
        ...,
        title="Shield type",
        description="Fixed to `question_validity` for this shield type.",
    )

    config: QuestionValidityConfig = Field(
        default_factory=QuestionValidityConfig,
        title="Shield config",
        description="Configuration for the `question_validity` shield.",
    )


class RedactionSafetyConfiguration(ConfigurationBase):
    """A ``redaction`` entry in ``safety``: pattern-based text redaction."""

    name: str = Field(
        ...,
        title="Shield name",
        description="Unique identifier for this safety shield.",
        min_length=1,
    )

    type: Literal["redaction"] = Field(
        ...,
        title="Shield type",
        description="Fixed to `redaction` for this shield type.",
    )

    config: RedactionConfig = Field(
        default_factory=RedactionConfig,
        title="Shield config",
        description="Configuration for the `redaction` shield.",
    )


class GraniteGuardianSafetyConfiguration(ConfigurationBase):
    """A ``granite_guardian`` entry in ``safety``: risk-based moderation via Granite Guardian.

    Becomes a single combined capability holding one sub-capability per risk in
    ``config.risks`` -- see
    :class:`~lightspeed.core.agent.safety.granite_guardian.capability.GraniteGuardian`.
    """

    name: str = Field(
        ...,
        title="Shield name",
        description="Unique identifier for this safety shield.",
        min_length=1,
    )

    type: Literal["granite_guardian"] = Field(
        ...,
        title="Shield type",
        description="Fixed to `granite_guardian` for this shield type.",
    )

    config: GraniteGuardianConfig = Field(
        ...,
        title="Shield config",
        description="Connection details and risk definitions for the `granite_guardian` shield.",
    )


SafetyCapabilityConfiguration = Annotated[
    Union[
        QuestionValiditySafetyConfiguration,
        RedactionSafetyConfiguration,
        GraniteGuardianSafetyConfiguration,
    ],
    Field(discriminator="type"),
]
"""A single ``safety`` list entry, discriminated on ``type``.

YAML shape::

    safety:
      - name: <name of shield>
        type: <question_validity, redaction, or granite_guardian>
        config: <config required for type>
"""


class ServiceConfiguration(ConfigurationBase):
    """Service configuration.

    Controls how the REST API service binds and how many Uvicorn worker
    processes handle requests concurrently. TLS and CORS (present in the
    original Lightspeed Stack) are not yet implemented.
    """

    host: str = Field(
        "localhost",
        title="Host",
        description="Service hostname.",
    )

    port: PositiveInt = Field(
        8080,
        title="Port",
        description="Service port.",
    )

    workers: PositiveInt = Field(
        1,
        title="Workers",
        description="Number of Uvicorn worker processes.",
    )

    auth_enabled: bool = Field(
        False,
        title="Auth enabled",
        description="Enable authentication. Not yet implemented.",
    )

    color_log: bool = Field(
        True,
        title="Color log",
        description="Enable colorized console logging. Not yet implemented.",
    )

    access_log: bool = Field(
        True,
        title="Access log",
        description="Enable Uvicorn access logging.",
    )


class Configuration(ConfigurationBase):
    """Root Lightspeed Stack configuration."""

    name: str = Field(
        "Lightspeed Stack",
        title="Service name",
        description="Name of the service.",
    )

    service: ServiceConfiguration = Field(
        default_factory=ServiceConfiguration,
        title="Service",
        description="REST API service configuration (host, port, workers, ...).",
    )

    providers: list[ProviderConfiguration] = Field(
        default_factory=list,
        title="Providers",
        description="Configured LLM providers available to the runtime.",
    )

    mcp_servers: list[MCPServerConfiguration] = Field(
        default_factory=list,
        title="MCP servers",
        description="Configured MCP servers available to agents as tools.",
    )

    skills: Optional[SkillsConfiguration] = Field(
        None,
        title="Agent skills",
        description=(
            "Agent skills configuration. Specifies paths to skill libraries."
        ),
    )

    safety: list[SafetyCapabilityConfiguration] = Field(
        default_factory=list,
        title="Safety",
        description=(
            "Configured safety shields (question_validity, redaction, ...) available to agents."
        ),
    )

    # Sections present in lightspeed-stack.yaml that are not yet implemented in
    # this rewrite. Declared here (instead of relying on `extra="forbid"`
    # rejecting them) so a full configuration file loads without error; each
    # becomes a typed model as its feature lands.
    authentication: Optional[Any] = Field(None, title="Authentication")
    authorization: Optional[Any] = Field(None, title="Authorization")
    knowledge: Optional[Any] = Field(None, title="Knowledge")
