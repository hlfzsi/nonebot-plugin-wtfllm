from typing import Any

# import orjson
from pydantic_ai import ToolReturn

from ....deps import AgentDeps


def build_budget_suffix(deps: AgentDeps, cost: int) -> str | None:
    """构建预算状态后缀"""
    if not deps.tool_budget_enabled:
        return None
    remaining = deps.tool_points_remaining
    ratio = deps.tool_budget_ratio

    if ratio >= 0.7:
        tier_hint = ""
    elif ratio >= 0.3:
        tier_hint = " 预算紧张，请谨慎使用工具。"
    else:
        tier_hint = " 预算耗尽，立即给出最终回复。"

    return f"\n[工具预算: -{cost}pt, 剩余 {remaining}/{deps.tool_point_budget}pt{tier_hint}]"


def append_budget_suffix(result: Any, suffix: str | None) -> Any:
    """将预算后缀追加到工具返回值"""
    if suffix is None:
        return result
    if isinstance(result, str):
        return result + suffix
    if isinstance(result, ToolReturn) and isinstance(result.return_value, str):
        return ToolReturn(
            return_value=result.return_value + suffix,
            content=result.content,
            metadata=result.metadata,
        )

    # return append_budget_suffix(orjson.dumps(result).decode("utf-8"), suffix)
    return result
