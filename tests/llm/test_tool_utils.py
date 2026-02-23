"""llm/tools/tool_group/utils 单元测试

覆盖: budget.py, reschedule.py
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import ToolReturn

from nonebot_plugin_wtfllm.llm.tools.tool_group.utils.budget import (
    build_budget_suffix,
    append_budget_suffix,
)
from nonebot_plugin_wtfllm.llm.tools.tool_group.utils.reschedule import (
    reschedule_deadline,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_deps(budget: int = 0, used: int = 0) -> AgentDeps:
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx = MagicMock(spec=MemoryContextBuilder)
    return AgentDeps(
        ids=IDs(user_id="u1", agent_id="a1"),
        context=mock_ctx,
        tool_point_budget=budget,
        tool_points_used=used,
    )


# ===================== build_budget_suffix 测试 =====================


class TestBuildBudgetSuffix:
    def test_budget_not_enabled(self):
        deps = _make_deps(budget=0)
        assert build_budget_suffix(deps, cost=1) is None

    def test_high_ratio(self):
        deps = _make_deps(budget=100, used=10)
        result = build_budget_suffix(deps, cost=1)
        assert result is not None
        assert "90/100" in result
        assert "紧张" not in result
        assert "耗尽" not in result

    def test_medium_ratio(self):
        deps = _make_deps(budget=100, used=55)
        result = build_budget_suffix(deps, cost=1)
        assert result is not None
        assert "紧张" in result

    def test_low_ratio(self):
        deps = _make_deps(budget=100, used=80)
        result = build_budget_suffix(deps, cost=1)
        assert result is not None
        assert "耗尽" in result


# ===================== append_budget_suffix 测试 =====================


class TestAppendBudgetSuffix:
    def test_none_suffix(self):
        assert append_budget_suffix("hello", None) == "hello"

    def test_string_result(self):
        result = append_budget_suffix("ok", " [budget]")
        assert result == "ok [budget]"

    def test_tool_return_with_string(self):
        tr = ToolReturn(return_value="result", content=None, metadata=None)
        result = append_budget_suffix(tr, " [budget]")
        assert isinstance(result, ToolReturn)
        assert result.return_value == "result [budget]"

    def test_non_string_result_unchanged(self):
        result = append_budget_suffix(42, " [budget]")
        assert result == 42

    def test_dict_result_unchanged(self):
        data = {"key": "value"}
        result = append_budget_suffix(data, " [budget]")
        assert result is data


# ===================== reschedule_deadline 测试 =====================


class TestRescheduleDeadline:
    def test_no_event_loop(self):
        ctx = MagicMock()
        ctx.deps.cm = MagicMock()
        # 在没有事件循环的情况下不应抛出异常
        reschedule_deadline(ctx, 10.0)

    @pytest.mark.asyncio
    async def test_cm_is_none(self):
        ctx = MagicMock()
        ctx.deps.cm = None
        reschedule_deadline(ctx, 10.0)

    @pytest.mark.asyncio
    async def test_reschedules_correctly(self):
        mock_cm = MagicMock()
        mock_cm.when.return_value = 100.0

        ctx = MagicMock()
        ctx.deps.cm = mock_cm

        reschedule_deadline(ctx, 30.0)
        mock_cm.reschedule.assert_called_once_with(130.0)
