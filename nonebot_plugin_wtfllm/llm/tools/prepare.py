from typing import Any, List, Optional
from pydantic_ai import Agent, RunContext, ToolDefinition

from ..deps import AgentDeps
from .tool_group.base import ToolGroupMeta
from ...utils import logger


async def _prepare(
    ctx: RunContext[AgentDeps], tool: ToolDefinition
) -> Optional[ToolDefinition]:
    if "prepared_tools" not in ctx.deps.caches:
        ctx.deps.caches["prepared_tools"] = set()
    added_tools: set[str] = ctx.deps.caches["prepared_tools"]

    if tool.metadata is None:
        logger.warning("Tool metadata is required for skill_prepare but is missing")
        return None
    parent_pack = tool.metadata.get("group_name")

    if parent_pack not in ctx.deps.active_tool_groups:
        return None

    # 预算耗尽时隐藏非免费工具
    if ctx.deps.tool_budget_exhausted:
        tool_cost = tool.metadata.get("cost", 0)
        if tool_cost > 0:
            return None

    if tool.name not in added_tools:
        added_tools.add(tool.name)
        logger.debug(f"Adding tool '{tool.name}' from active group '{parent_pack}'")
    return tool


def register_tools_to_agent(
    agent: Agent[AgentDeps, Any],
    library: List[ToolGroupMeta],
):
    for pack in library:
        for tool_func in pack.tools:
            cost = pack.resolve_tool_cost(tool_func.__name__)
            agent.tool(
                prepare=_prepare,
                metadata={"group_name": pack.name, "cost": cost},
            )(tool_func)
