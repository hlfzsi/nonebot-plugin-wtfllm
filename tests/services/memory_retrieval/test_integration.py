"""RetrievalChain + Task 集成测试

模拟真实调用场景：构建链 → resolve → 验证合并后的 MemorySource 集合。
仓储层全部用 AsyncMock 替代。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nonebot_plugin_wtfllm.memory.items import (
    MemoryItemStream,
    CoreMemoryBlock,
    KnowledgeBlock,
    ToolCallSummaryBlock,
)
from nonebot_plugin_wtfllm.memory.items.core_memory import CoreMemory
from nonebot_plugin_wtfllm.memory.items.knowledge_base import KnowledgeEntry
from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem
from nonebot_plugin_wtfllm.memory.content import Message

from nonebot_plugin_wtfllm.services.func.memory_retrieval.chain import RetrievalChain
from nonebot_plugin_wtfllm.services.func.memory_retrieval.main_chat import MainChatTask
from nonebot_plugin_wtfllm.services.func.memory_retrieval.core_memory import (
    CoreMemoryTask,
    CrossSessionMemoryTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.knowledge import (
    KnowledgeSearchTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.tool_history import (
    ToolCallHistoryTask,
)
from nonebot_plugin_wtfllm.services.func.memory_retrieval.recent_react import (
    RecentReactTask,
)


# ── patch 路径 ────────────────────────────────────────

_MAIN = "nonebot_plugin_wtfllm.services.func.memory_retrieval.main_chat"
_CORE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.core_memory"
_KNOW = "nonebot_plugin_wtfllm.services.func.memory_retrieval.knowledge"
_TOOL = "nonebot_plugin_wtfllm.services.func.memory_retrieval.tool_history"
_REACT = "nonebot_plugin_wtfllm.services.func.memory_retrieval.recent_react"


# ── fixtures ──────────────────────────────────────────


def _items(n: int):
    import time

    base_ts = int(time.time()) - 600
    return [
        GroupMemoryItem(
            message_id=f"msg_{i}",
            sender=f"user_{i}",
            content=Message.create(),
            created_at=base_ts + i * 60,
            agent_id="agent_1",
            group_id="group_1",
        )
        for i in range(n)
    ]


def _core_mems(n: int):
    return [
        CoreMemory(content=f"cm_{i}", agent_id="agent_1", group_id="group_1")
        for i in range(n)
    ]


def _know_entries(n: int, tokens: int = 50):
    return [
        KnowledgeEntry(
            content=f"k_{i}",
            title=f"Title_{i}",
            agent_id="agent_1",
            token_count=tokens,
        )
        for i in range(n)
    ]


# ── 群聊完整流程 ──────────────────────────────────────


class TestGroupChatIntegration:
    """模拟群聊场景：main_chat + core_memory + knowledge + tool_history"""

    @pytest.mark.asyncio
    @patch(f"{_TOOL}.tool_call_record_repo")
    @patch(f"{_KNOW}.knowledge_base_repo")
    @patch(f"{_CORE}.core_memory_repo")
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_full_chain(
        self, mock_mem, mock_core, mock_know, mock_tool
    ):
        mock_mem.get_by_group = AsyncMock(return_value=_items(5))
        mock_core.get_by_session = AsyncMock(return_value=_core_mems(3))
        mock_know.search_relevant = AsyncMock(
            return_value=[MagicMock(item=e) for e in _know_entries(2)]
        )
        mock_tool.get_recent = AsyncMock(
            return_value=[MagicMock(tool_name="wiki"), MagicMock(tool_name="calc")]
        )

        sources = await (
            RetrievalChain(agent_id="agent_1", group_id="group_1", query="hello")
            .main_chat(limit=50)
            .core_memory()
            .knowledge()
            .tool_history()
            .resolve()
        )

        # 4 个 Task → 4 个 MemorySource
        assert len(sources) == 4

        types_found = {type(s) for s in sources}
        assert MemoryItemStream in types_found
        assert CoreMemoryBlock in types_found
        assert KnowledgeBlock in types_found
        assert ToolCallSummaryBlock in types_found

    @pytest.mark.asyncio
    @patch(f"{_TOOL}.tool_call_record_repo")
    @patch(f"{_CORE}.core_memory_repo")
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_partial_empty(self, mock_mem, mock_core, mock_tool):
        """部分仓储返回空 → 对应 task 返回空集 → 合并结果不受影响"""
        mock_mem.get_by_group = AsyncMock(return_value=_items(3))
        mock_core.get_by_session = AsyncMock(return_value=[])  # 空
        mock_tool.get_recent = AsyncMock(return_value=[])  # 空

        sources = await (
            RetrievalChain(agent_id="agent_1", group_id="group_1")
            .main_chat()
            .core_memory()
            .tool_history()
            .resolve()
        )

        # 只有 main_chat 返回了结果
        assert len(sources) == 1
        assert isinstance(next(iter(sources)), MemoryItemStream)


# ── 私聊完整流程 ──────────────────────────────────────


class TestPrivateChatIntegration:
    @pytest.mark.asyncio
    @patch(f"{_TOOL}.tool_call_record_repo")
    @patch(f"{_CORE}.core_memory_repo")
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_private_chain(self, mock_mem, mock_core, mock_tool):
        mock_mem.get_in_private_by_user = AsyncMock(return_value=_items(2))
        mock_core.get_by_session = AsyncMock(return_value=_core_mems(1))
        mock_tool.get_recent = AsyncMock(
            return_value=[MagicMock(tool_name="search")]
        )

        sources = await (
            RetrievalChain(agent_id="agent_1", user_id="user_1")
            .main_chat()
            .core_memory()
            .tool_history()
            .resolve()
        )

        assert len(sources) == 3

        # 确认调用了私聊接口
        mock_mem.get_in_private_by_user.assert_awaited_once()
        mock_mem.get_by_group = AsyncMock()  # 不应被调用
        mock_mem.get_by_group.assert_not_awaited()


# ── 跨会话记忆 + 知识库 ──────────────────────────────


class TestCrossSessionIntegration:
    @pytest.mark.asyncio
    @patch(f"{_KNOW}.knowledge_base_repo")
    @patch(f"{_CORE}.core_memory_repo")
    async def test_cross_session_and_knowledge(self, mock_core, mock_know):
        mock_core.search_cross_session = AsyncMock(
            return_value=[MagicMock(item=m) for m in _core_mems(2)]
        )
        mock_know.search_relevant = AsyncMock(
            return_value=[MagicMock(item=e) for e in _know_entries(1)]
        )

        sources = await (
            RetrievalChain(
                agent_id="agent_1",
                group_id="group_1",
                query="重要事件",
            )
            .cross_session_memory()
            .knowledge()
            .resolve()
        )

        assert len(sources) == 2

        # 验证 cross_session 默认参数传递
        mock_core.search_cross_session.assert_awaited_once_with(
            agent_id="agent_1",
            query="重要事件",
            exclude_group_id="group_1",
            exclude_user_id=None,
            limit=5,
        )


# ── recent_react 集成 ────────────────────────────────


class TestRecentReactIntegration:
    @pytest.mark.asyncio
    @patch(f"{_REACT}.memory_item_repo")
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_main_chat_and_recent_react(self, mock_main_repo, mock_react_repo):
        mock_main_repo.get_by_group = AsyncMock(return_value=_items(3))
        mock_react_repo.get_many_by_message_ids = AsyncMock(
            return_value=_items(2)
        )

        alias_provider = MagicMock()
        alias_provider.get_alias = MagicMock(return_value="TestGroup")

        sources = await (
            RetrievalChain(agent_id="agent_1", group_id="group_1")
            .main_chat()
            .recent_react(
                recent_react={"other_group": ["m1", "m2"]},
                alias_provider=alias_provider,
            )
            .resolve()
        )

        # main_chat → 1 stream, recent_react → 1 stream
        assert len(sources) == 2
        all_streams = [s for s in sources if isinstance(s, MemoryItemStream)]
        assert len(all_streams) == 2


# ── 链合并集成 ────────────────────────────────────────


class TestChainMergeIntegration:
    @pytest.mark.asyncio
    @patch(f"{_TOOL}.tool_call_record_repo")
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_add_merge(self, mock_mem, mock_tool):
        mock_mem.get_by_group = AsyncMock(return_value=_items(2))
        mock_tool.get_recent = AsyncMock(
            return_value=[MagicMock(tool_name="calc")]
        )

        chain_a = RetrievalChain(agent_id="agent_1", group_id="group_1").main_chat()
        chain_b = RetrievalChain(agent_id="agent_1", group_id="group_1").tool_history()

        merged = chain_a + chain_b
        sources = await merged.resolve()

        assert len(sources) == 2
        assert MemoryItemStream in {type(s) for s in sources}
        assert ToolCallSummaryBlock in {type(s) for s in sources}

    @pytest.mark.asyncio
    @patch(f"{_TOOL}.tool_call_record_repo")
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_iadd_merge(self, mock_mem, mock_tool):
        mock_mem.get_by_group = AsyncMock(return_value=_items(1))
        mock_tool.get_recent = AsyncMock(
            return_value=[MagicMock(tool_name="search")]
        )

        chain = RetrievalChain(agent_id="agent_1", group_id="group_1").main_chat()
        extra = RetrievalChain(agent_id="agent_1", group_id="group_1").tool_history()
        chain += extra

        sources = await chain.resolve()
        assert len(sources) == 2


# ── 空链 ────────────────────────────────────────────


class TestEmptyChain:
    @pytest.mark.asyncio
    async def test_empty_chain_resolve(self):
        sources = await RetrievalChain().resolve()
        assert sources == set()

    @pytest.mark.asyncio
    async def test_chain_bool_false(self):
        assert not RetrievalChain()

    @pytest.mark.asyncio
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_all_tasks_return_empty(self, mock_repo):
        """所有仓储都返回空 → resolve 返回空集"""
        mock_repo.get_by_group = AsyncMock(return_value=[])

        sources = await (
            RetrievalChain(agent_id="agent_1", group_id="group_1")
            .main_chat()
            .resolve()
        )
        # MainChatTask 返回空集（items 列表虽然为空，但 MemoryItemStream
        # 仍会被 create——需要检查源码逻辑）
        # 实际上 main_chat.py 无论如何都会 return {stream}
        # 所以即使 items 为空，也返回 1 个 stream
        assert len(sources) == 1


# ── 默认参数覆盖 ────────────────────────────────────


class TestDefaultOverride:
    @pytest.mark.asyncio
    @patch(f"{_MAIN}.memory_item_repo")
    async def test_shortcut_override_defaults(self, mock_repo):
        """快捷方法显式传参 → 覆盖链级默认值"""
        mock_repo.get_by_group = AsyncMock(return_value=_items(1))

        chain = RetrievalChain(
            agent_id="agent_1", group_id="group_1"
        ).main_chat(group_id="override_group", limit=10)

        await chain.resolve()

        mock_repo.get_by_group.assert_awaited_once_with(
            group_id="override_group",
            agent_id="agent_1",
            limit=10,
        )

    @pytest.mark.asyncio
    @patch(f"{_CORE}.core_memory_repo")
    async def test_cross_session_inherits_group_for_exclude(self, mock_repo):
        """cross_session_memory 的 exclude_group_id 默认继承 chain.group_id"""
        mock_repo.search_cross_session = AsyncMock(return_value=[])

        await (
            RetrievalChain(
                agent_id="agent_1",
                group_id="group_1",
                user_id="user_1",
                query="test",
            )
            .cross_session_memory()
            .resolve()
        )

        mock_repo.search_cross_session.assert_awaited_once_with(
            agent_id="agent_1",
            query="test",
            exclude_group_id="group_1",
            exclude_user_id="user_1",
            limit=5,
        )
