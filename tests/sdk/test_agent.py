"""SDK Agent 调用测试。

验证：
- run_chat_agent 正确委托 CHAT_AGENT.run 的参数
- 返回值透传
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 直接导入子模块
import nonebot_plugin_wtfllm.llm.deps as _deps
import nonebot_plugin_wtfllm.sdk.agent as sdk_agent
from nonebot_plugin_wtfllm.memory.director import MemoryContextBuilder

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


# ===================== Fixtures =====================


@pytest.fixture
def mock_deps():
    """创建最小化的 AgentDeps 用于测试"""
    builder = MemoryContextBuilder(agent_id="a1")
    return AgentDeps(
        ids=IDs(user_id="u1", agent_id="a1"),
        context=builder,
        active_tool_groups={"Core"},
    )


# ===================== run_chat_agent 测试 =====================


class TestRunChatAgent:

    @pytest.mark.asyncio
    async def test_delegates_to_chat_agent(self, mock_deps):
        """run_chat_agent 正确委托到 CHAT_AGENT.run"""
        sentinel_result = MagicMock()

        with patch.object(
            sdk_agent, "CHAT_AGENT", autospec=True
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=sentinel_result)
            result = await sdk_agent.run_chat_agent(
                prompt="你好", deps=mock_deps
            )

        mock_agent.run.assert_awaited_once_with(
            user_prompt="你好", deps=mock_deps
        )
        assert result is sentinel_result

    @pytest.mark.asyncio
    async def test_passes_kwargs(self, mock_deps):
        """额外 kwargs 透传给 CHAT_AGENT.run"""
        sentinel_result = MagicMock()
        fake_history = [MagicMock()]

        with patch.object(
            sdk_agent, "CHAT_AGENT", autospec=True
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=sentinel_result)
            result = await sdk_agent.run_chat_agent(
                prompt="test",
                deps=mock_deps,
                message_history=fake_history,
            )

        mock_agent.run.assert_awaited_once_with(
            user_prompt="test",
            deps=mock_deps,
            message_history=fake_history,
        )
        assert result is sentinel_result

    @pytest.mark.asyncio
    async def test_returns_result_unchanged(self, mock_deps):
        """返回值不被修改地透传"""
        expected = MagicMock()
        expected.output = MagicMock()
        expected.usage = MagicMock(return_value={"tokens": 100})

        with patch.object(
            sdk_agent, "CHAT_AGENT", autospec=True
        ) as mock_agent:
            mock_agent.run = AsyncMock(return_value=expected)
            result = await sdk_agent.run_chat_agent(
                prompt="hello", deps=mock_deps
            )

        assert result is expected
        assert result.output is expected.output
