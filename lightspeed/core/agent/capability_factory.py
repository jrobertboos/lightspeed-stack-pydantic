"""Abstract factory for building pydantic-ai agent capabilities from configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic_ai.capabilities import AgentCapability

ConfigT = TypeVar("ConfigT")


class CapabilityFactory(ABC, Generic[ConfigT]):
    """Builds :class:`~pydantic_ai.capabilities.AgentCapability` instances from config.

    Each capability type (MCP, Skills, ...) subclasses this factory and implements
    :meth:`build_capabilities` for its configuration section.
    """

    @staticmethod
    @abstractmethod
    def build_capabilities(config: ConfigT) -> list[AgentCapability[Any]]:
        """Build zero or more capabilities from the given configuration section.

        Parameters:
            config: The configuration section this factory understands (e.g.
                ``Configuration.mcp_servers`` or ``Configuration.skills``).

        Returns:
            Capabilities suitable for ``Agent(capabilities=...)``.
        """
