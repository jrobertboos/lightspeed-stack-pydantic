"""Build pydantic-ai-harness ``Skills`` capabilities from configured skill paths.

Configured library directories are wrapped in a single
:class:`~pydantic_ai_harness.skills.Skills` capability -- harness's Agent
Skills entry point -- rather than attaching skill instructions directly via
``Agent(instructions=...)``. Each discovered ``SKILL.md`` becomes a deferred
pydantic-ai capability: the model first sees the skill's name and
description, then calls ``load_capability`` to receive its Markdown body.

Each path must be a skill **library** (a directory whose immediate children
contain ``SKILL.md``). Harness ``Skills`` rejects a path that points at a
skill package itself. Bundled files (``references/``, ``scripts/``, ...) are
not loaded or executed -- that is harness ``Skills`` behavior, not a
Lightspeed-specific restriction.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.skills import Skills

from lightspeed.app.models.config import SkillsConfiguration
from lightspeed.core.agent.capability_factory import CapabilityFactory


class SkillsCapabilityFactory(CapabilityFactory[Optional[SkillsConfiguration]]):
    """Builds :class:`~pydantic_ai_harness.skills.Skills` capabilities from configured paths."""

    @staticmethod
    def build_capabilities(
        config: Optional[SkillsConfiguration],
    ) -> list[AgentCapability[Any]]:
        """Build one ``Skills`` capability covering every configured library.

        Parameters:
            config: The configured skills section (e.g.
                ``Configuration.skills``), or ``None`` when skills are
                omitted.

        Returns:
            A one-element list with a ``Skills`` capability over
            ``config.paths``, suitable for ``Agent(capabilities=...)``.
            Empty when ``config`` is ``None`` or ``paths`` is empty.

        Raises:
            ValueError: If a path is missing, is not a directory, points at
                a skill package rather than a library, has invalid
                ``SKILL.md`` frontmatter, or would produce duplicate skill
                names -- all raised by harness ``Skills`` during
                construction.
        """
        if config is None or not config.paths:
            return []
        return [Skills(config.paths)]
