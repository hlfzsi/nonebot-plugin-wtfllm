"""llm/deps.py 单元测试

覆盖 IDs、NonebotRuntime、AgentDeps 模型验证。
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

# 直接导入子模块避免 llm.__init__ 的循环导入
import nonebot_plugin_wtfllm.llm.deps as _deps

IDs = _deps.IDs
AgentDeps = _deps.AgentDeps


class TestIDs:
    """IDs 模型测试"""

    def test_create_with_group(self):
        ids = IDs(user_id="u1", group_id="g1", agent_id="a1")
        assert ids.user_id == "u1"
        assert ids.group_id == "g1"
        assert ids.agent_id == "a1"

    def test_create_private(self):
        ids = IDs(user_id="u1", group_id=None, agent_id="a1")
        assert ids.group_id is None

    def test_user_id_can_be_none(self):
        ids = IDs(user_id=None, agent_id="a1")
        assert ids.user_id is None


class TestAgentDeps:
    """AgentDeps 模型测试"""

    @pytest.fixture
    def mock_context(self):
        """创建 mock MemoryContextBuilder"""
        from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

        ctx = MagicMock(spec=MemoryContextBuilder)
        ctx.ctx = MagicMock()
        ctx.ctx.alias_provider = MagicMock()
        return ctx

    def test_create_basic(self, mock_context):
        deps = AgentDeps(
            ids=IDs(user_id="u1", agent_id="a1"),
            context=mock_context,
        )
        assert deps.ids.user_id == "u1"
        assert deps.active_tool_groups == set()
        assert deps.nb_runtime is None
        assert deps.caches == {}

    def test_active_tool_groups(self, mock_context):
        deps = AgentDeps(
            ids=IDs(user_id="u1", agent_id="a1"),
            context=mock_context,
            active_tool_groups={"Core", "Chat"},
        )
        assert "Core" in deps.active_tool_groups
        assert "Chat" in deps.active_tool_groups

    def test_invalid_context_raises(self):
        with pytest.raises(ValueError, match="MemoryContextBuilder"):
            AgentDeps(
                ids=IDs(user_id="u1", agent_id="a1"),
                context="not_a_context",
            )

    def test_reply_segments_default(self, mock_context):
        deps = AgentDeps(
            ids=IDs(user_id="u1", agent_id="a1"),
            context=mock_context,
        )
        # reply_segments defaults to UniMessage()
        assert deps.reply_segments is not None

    def test_caches_usage(self, mock_context):
        deps = AgentDeps(
            ids=IDs(user_id="u1", agent_id="a1"),
            context=mock_context,
            caches={"key": "value"},
        )
        assert deps.caches["key"] == "value"
