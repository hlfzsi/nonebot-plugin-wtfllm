"""各 RetrievalTask 子类单元测试

Mock 仓储层，验证每个 Task 的 execute() 逻辑。
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nonebot_plugin_wtfllm.memory.items.storages import MemoryItemStream
from nonebot_plugin_wtfllm.memory.items.core_memory import CoreMemoryBlock, CoreMemory
from nonebot_plugin_wtfllm.memory.items.note import Note, NoteBlock
from nonebot_plugin_wtfllm.memory.items.knowledge_base import (
    KnowledgeBlock,
    KnowledgeEntry,
)
from nonebot_plugin_wtfllm.memory.items.tool_call_summary import ToolCallSummaryBlock
from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem
from nonebot_plugin_wtfllm.memory.content import Message

from nonebot_plugin_wtfllm.services.func.memory_retrieval.main_chat import MainChatTask
from nonebot_plugin_wtfllm.services.func.memory_retrieval.note import NoteTask
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


# ── helpers ──────────────────────────────────────────


def _make_group_items(n: int):
    base_ts = int(time.time()) - 600
    items = []
    for i in range(n):
        items.append(
            GroupMemoryItem(
                message_id=f"msg_{i}",
                sender=f"user_{i}",
                content=Message.create().text(f"Hello {i}"),
                created_at=base_ts + i * 60,
                agent_id="agent_1",
                group_id="group_1",
            )
        )
    return items


def _make_core_memories(n: int):
    return [
        CoreMemory(
            content=f"Core memory {i}",
            agent_id="agent_1",
            group_id="group_1",
        )
        for i in range(n)
    ]


def _make_notes(n: int, expires_at: int | None = None):
    base_expiry = expires_at or int(time.time()) + 1800
    return [
        Note(
            content=f"Note {i}",
            agent_id="agent_1",
            group_id="group_1",
            expires_at=base_expiry + i * 60,
        )
        for i in range(n)
    ]


def _make_knowledge_entries(n: int, token_count: int = 100):
    return [
        KnowledgeEntry(
            content=f"Knowledge {i}",
            title=f"Title {i}",
            agent_id="agent_1",
            token_count=token_count,
        )
        for i in range(n)
    ]


# ── MainChatTask ─────────────────────────────────────


MAIN_CHAT_MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.main_chat"


class TestMainChatTask:
    @pytest.mark.asyncio
    @patch(f"{MAIN_CHAT_MODULE}.memory_item_repo")
    async def test_group_chat(self, mock_repo):
        items = _make_group_items(3)
        mock_repo.get_by_group = AsyncMock(return_value=items)

        task = MainChatTask(agent_id="a1", group_id="g1", limit=10)
        result = await task.execute()

        mock_repo.get_by_group.assert_awaited_once_with(
            group_id="g1", agent_id="a1", limit=10
        )
        assert len(result) == 1
        stream = next(iter(result))
        assert isinstance(stream, MemoryItemStream)
        assert stream.role == "main_chat"
        assert stream.priority == pytest.approx(0.1)
        assert len(stream.items) == 3

    @pytest.mark.asyncio
    @patch(f"{MAIN_CHAT_MODULE}.memory_item_repo")
    async def test_private_chat(self, mock_repo):
        items = _make_group_items(2)  # 内容不重要
        mock_repo.get_in_private_by_user = AsyncMock(return_value=items)

        task = MainChatTask(agent_id="a1", user_id="u1", limit=5)
        result = await task.execute()

        mock_repo.get_in_private_by_user.assert_awaited_once_with(
            user_id="u1", agent_id="a1", limit=5
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_no_session_returns_empty(self):
        task = MainChatTask(agent_id="a1")
        result = await task.execute()
        assert result == set()


# ── CoreMemoryTask ───────────────────────────────────


CORE_MEM_MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.core_memory"


class TestCoreMemoryTask:
    @pytest.mark.asyncio
    @patch(f"{CORE_MEM_MODULE}.core_memory_repo")
    async def test_returns_block(self, mock_repo):
        memories = _make_core_memories(3)
        mock_repo.get_by_session = AsyncMock(return_value=memories)

        task = CoreMemoryTask(agent_id="a1", group_id="g1")
        result = await task.execute()

        mock_repo.get_by_session.assert_awaited_once_with(
            agent_id="a1", group_id="g1", user_id=None
        )
        assert len(result) == 1
        block = next(iter(result))
        assert isinstance(block, CoreMemoryBlock)
        assert len(block.memories) == 3
        assert block.prefix == "<core_memory>"
        assert block.priority == pytest.approx(2)

    @pytest.mark.asyncio
    @patch(f"{CORE_MEM_MODULE}.core_memory_repo")
    async def test_empty_memories(self, mock_repo):
        mock_repo.get_by_session = AsyncMock(return_value=[])

        task = CoreMemoryTask(agent_id="a1", group_id="g1")
        result = await task.execute()
        assert result == set()

    @pytest.mark.asyncio
    @patch(f"{CORE_MEM_MODULE}.core_memory_repo")
    async def test_custom_prefix_suffix(self, mock_repo):
        mock_repo.get_by_session = AsyncMock(return_value=_make_core_memories(1))

        task = CoreMemoryTask(
            agent_id="a1", group_id="g1", prefix="<custom>", suffix="</custom>"
        )
        result = await task.execute()
        block = next(iter(result))
        assert block.prefix == "<custom>"
        assert block.suffix == "</custom>"


# ── NoteTask ─────────────────────────────────────────


NOTE_MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.note"


class TestNoteTask:
    @pytest.mark.asyncio
    @patch(f"{NOTE_MODULE}.note_memory_repo")
    @patch(f"{NOTE_MODULE}.time.time", return_value=1_700_000_000)
    async def test_returns_block(self, mock_time, mock_repo):
        notes = _make_notes(2, expires_at=1_700_001_800)
        mock_repo.delete_expired_by_session = AsyncMock(return_value=0)
        mock_repo.get_by_session = AsyncMock(return_value=notes)

        task = NoteTask(agent_id="a1", group_id="g1")
        result = await task.execute()

        mock_repo.delete_expired_by_session.assert_awaited_once_with(
            agent_id="a1", group_id="g1", user_id=None
        )
        mock_repo.get_by_session.assert_awaited_once_with(
            agent_id="a1",
            group_id="g1",
            user_id=None,
            include_expired=False,
        )
        assert len(result) == 1
        block = next(iter(result))
        assert isinstance(block, NoteBlock)
        assert len(block.notes) == 2
        assert block.prefix == "<note_memory>"

    @pytest.mark.asyncio
    @patch(f"{NOTE_MODULE}.note_memory_repo")
    @patch(f"{NOTE_MODULE}.time.time", return_value=1_700_000_000)
    async def test_filters_expired_notes(self, mock_time, mock_repo):
        notes = _make_notes(1, expires_at=1_699_999_999)
        mock_repo.delete_expired_by_session = AsyncMock(return_value=1)
        mock_repo.get_by_session = AsyncMock(return_value=notes)

        task = NoteTask(agent_id="a1", group_id="g1")
        result = await task.execute()

        assert result == set()


# ── CrossSessionMemoryTask ───────────────────────────


class TestCrossSessionMemoryTask:
    @pytest.mark.asyncio
    @patch(f"{CORE_MEM_MODULE}.core_memory_repo")
    async def test_returns_block(self, mock_repo):
        search_results = [MagicMock(item=m) for m in _make_core_memories(2)]
        mock_repo.search_cross_session = AsyncMock(return_value=search_results)

        task = CrossSessionMemoryTask(
            agent_id="a1", query="test", exclude_group_id="g1"
        )
        result = await task.execute()

        mock_repo.search_cross_session.assert_awaited_once_with(
            agent_id="a1",
            query="test",
            exclude_group_id="g1",
            exclude_user_id=None,
            limit=5,
        )
        assert len(result) == 1
        block = next(iter(result))
        assert isinstance(block, CoreMemoryBlock)
        assert block.prefix == "<cross_session_memory>"

    @pytest.mark.asyncio
    @patch(f"{CORE_MEM_MODULE}.core_memory_repo")
    async def test_empty_results(self, mock_repo):
        mock_repo.search_cross_session = AsyncMock(return_value=[])

        task = CrossSessionMemoryTask(agent_id="a1", query="test")
        result = await task.execute()
        assert result == set()


# ── KnowledgeSearchTask ──────────────────────────────


KNOWLEDGE_MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.knowledge"


class TestKnowledgeSearchTask:
    @pytest.mark.asyncio
    @patch(f"{KNOWLEDGE_MODULE}.knowledge_base_repo")
    async def test_returns_block(self, mock_repo):
        entries = _make_knowledge_entries(3, token_count=100)
        search_results = [MagicMock(item=e) for e in entries]
        mock_repo.search_relevant = AsyncMock(return_value=search_results)

        task = KnowledgeSearchTask(
            agent_id="a1", query="test", limit=5, max_tokens=4000
        )
        result = await task.execute()

        assert len(result) == 1
        block = next(iter(result))
        assert isinstance(block, KnowledgeBlock)
        assert len(block.entries) == 3
        assert block.priority == pytest.approx(3)

    @pytest.mark.asyncio
    @patch(f"{KNOWLEDGE_MODULE}.knowledge_base_repo")
    async def test_token_truncation(self, mock_repo):
        """token 数超出限制时截断"""
        entries = _make_knowledge_entries(5, token_count=300)
        search_results = [MagicMock(item=e) for e in entries]
        mock_repo.search_relevant = AsyncMock(return_value=search_results)

        task = KnowledgeSearchTask(
            agent_id="a1", query="test", limit=5, max_tokens=800
        )
        result = await task.execute()

        block = next(iter(result))
        # 800 / 300 = 2.66, 所以只有 2 个 (300 + 300 = 600 <= 800, 600+300=900 > 800)
        assert len(block.entries) == 2

    @pytest.mark.asyncio
    @patch(f"{KNOWLEDGE_MODULE}.knowledge_base_repo")
    async def test_empty_results(self, mock_repo):
        mock_repo.search_relevant = AsyncMock(return_value=[])

        task = KnowledgeSearchTask(agent_id="a1", query="test")
        result = await task.execute()
        assert result == set()

    @pytest.mark.asyncio
    @patch(f"{KNOWLEDGE_MODULE}.knowledge_base_repo")
    async def test_all_entries_exceed_max_tokens(self, mock_repo):
        """所有条目的 token 都超过 max_tokens 时返回空"""
        entries = _make_knowledge_entries(3, token_count=5000)
        search_results = [MagicMock(item=e) for e in entries]
        mock_repo.search_relevant = AsyncMock(return_value=search_results)

        task = KnowledgeSearchTask(
            agent_id="a1", query="test", max_tokens=100
        )
        result = await task.execute()
        assert result == set()


# ── ToolCallHistoryTask ──────────────────────────────


TOOL_MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.tool_history"


class TestToolCallHistoryTask:
    @pytest.mark.asyncio
    @patch(f"{TOOL_MODULE}.tool_call_record_repo")
    async def test_returns_block(self, mock_repo):
        records = [MagicMock(tool_name="search"), MagicMock(tool_name="save")]
        mock_repo.get_recent = AsyncMock(return_value=records)

        task = ToolCallHistoryTask(agent_id="a1", group_id="g1", limit=5)
        result = await task.execute()

        mock_repo.get_recent.assert_awaited_once_with(
            agent_id="a1", group_id="g1", user_id=None, limit=5
        )
        assert len(result) == 1
        block = next(iter(result))
        assert isinstance(block, ToolCallSummaryBlock)
        assert set(block.tool_names) == {"search", "save"}
        assert block.priority == pytest.approx(1)

    @pytest.mark.asyncio
    @patch(f"{TOOL_MODULE}.tool_call_record_repo")
    async def test_empty_records(self, mock_repo):
        mock_repo.get_recent = AsyncMock(return_value=[])

        task = ToolCallHistoryTask(agent_id="a1", group_id="g1")
        result = await task.execute()
        assert result == set()


# ── RecentReactTask ──────────────────────────────────


RECENT_MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.recent_react"


class TestRecentReactTask:
    @pytest.mark.asyncio
    @patch(f"{RECENT_MODULE}.memory_item_repo")
    async def test_creates_streams_per_group(self, mock_repo):
        items_g1 = _make_group_items(2)
        items_g2 = _make_group_items(1)
        mock_repo.get_many_by_message_ids = AsyncMock(
            side_effect=[items_g1, items_g2]
        )

        alias_provider = MagicMock()
        alias_provider.get_alias = MagicMock(
            side_effect=lambda gid: f"Group_{gid}"
        )

        task = RecentReactTask(
            recent_react={"g1": ["m1", "m2"], "g2": ["m3"]},
            alias_provider=alias_provider,
        )
        result = await task.execute()

        assert mock_repo.get_many_by_message_ids.await_count == 2
        assert len(result) == 2
        for stream in result:
            assert isinstance(stream, MemoryItemStream)
            assert stream.priority == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_empty_recent_react(self):
        alias_provider = MagicMock()
        task = RecentReactTask(
            recent_react={}, alias_provider=alias_provider
        )
        result = await task.execute()
        assert result == set()

    @pytest.mark.asyncio
    @patch(f"{RECENT_MODULE}.memory_item_repo")
    async def test_skips_empty_mid_lists(self, mock_repo):
        items = _make_group_items(1)
        mock_repo.get_many_by_message_ids = AsyncMock(return_value=items)

        alias_provider = MagicMock()
        alias_provider.get_alias = MagicMock(return_value="Group_g1")

        task = RecentReactTask(
            recent_react={"g1": ["m1"], "g2": []},
            alias_provider=alias_provider,
        )
        result = await task.execute()

        # 只调用一次（g2 的空列表被跳过）
        mock_repo.get_many_by_message_ids.assert_awaited_once_with(["m1"])
        assert len(result) == 1

    @pytest.mark.asyncio
    @patch(f"{RECENT_MODULE}.memory_item_repo")
    async def test_private_scene_name(self, mock_repo):
        """gid 为空字符串时，scene 应为"私聊" """
        items = _make_group_items(1)
        mock_repo.get_many_by_message_ids = AsyncMock(return_value=items)

        alias_provider = MagicMock()

        task = RecentReactTask(
            recent_react={"": ["m1"]},
            alias_provider=alias_provider,
        )
        result = await task.execute()

        stream = next(iter(result))
        assert '私聊' in stream.prefix


class TestMemorySourcePriorityContract:
    def test_global_order_matches_product_expectation(self):
        """排序契约: 知识 > 核心记忆 > 工具 > 最近交互 > 主题补充 > 主对话流"""
        knowledge = KnowledgeBlock(entries=_make_knowledge_entries(1))
        core = CoreMemoryBlock(memories=_make_core_memories(1))
        tool = ToolCallSummaryBlock(tool_names=["search"])

        items = _make_group_items(1)
        recent = MemoryItemStream.create(items=items, priority=0.3)
        topic = MemoryItemStream.create(items=items, role="topic_context", priority=0.2)
        main_chat = MemoryItemStream.create(items=items, role="main_chat", priority=0.1)

        sources = [main_chat, topic, recent, tool, core, knowledge]
        sorted_sources = sorted(
            sources, key=lambda x: (-x.priority, x.sort_key[0], x.sort_key[1])
        )

        assert sorted_sources == [knowledge, core, tool, recent, topic, main_chat]
