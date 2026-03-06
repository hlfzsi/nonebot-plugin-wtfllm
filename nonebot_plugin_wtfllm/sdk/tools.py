__all__ = [
    "ToolGroupMeta",
    "register_tool_groups",
    "list_registered_groups",
    "get_tool_group",
]

from ..llm.tools.tool_group.base import ToolGroupMeta
from ..llm.agents import (
    register_tool_groups,
    get_registered_group_names,
)


def list_registered_groups() -> list[str]:
    """返回当前已注册到 CHAT_AGENT 的工具组名称列表（已排序）。"""
    return sorted(get_registered_group_names())


def get_tool_group(name: str) -> ToolGroupMeta | None:
    """按名称查找工具组（不要求已注册到 CHAT_AGENT）。"""
    return ToolGroupMeta.mapping.get(name)
