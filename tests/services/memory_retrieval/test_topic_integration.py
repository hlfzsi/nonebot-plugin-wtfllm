"""集成测试: TopicManager -> TopicContextTask 全链路

验证话题摄入后，TopicContextTask 能正确检索出同话题的旧消息。
"""

import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from nonebot_plugin_wtfllm.memory.content import Message
from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem
from nonebot_plugin_wtfllm.memory.items import MemoryItemStream
from nonebot_plugin_wtfllm.topic.manager import TopicManager
from nonebot_plugin_wtfllm.services.func.memory_retrieval.topic_context import (
    TopicContextTask,
)


MODULE = "nonebot_plugin_wtfllm.services.func.memory_retrieval.topic_context"


def _make_items(message_ids: list[str], base_ts: int):
    """创建真实的 GroupMemoryItem 列表"""
    return [
        GroupMemoryItem(
            message_id=mid,
            sender="u1",
            content=Message.create(),
            created_at=base_ts + i * 10,
            agent_id="a1",
            group_id="g1",
        )
        for i, mid in enumerate(message_ids)
    ]


class TestTopicIntegration:
    """TopicManager + TopicContextTask 集成测试"""

    @pytest.fixture
    def manager(self):
        return TopicManager(maxsize=10, cluster_threshold=0.65, max_clusters=20)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    @patch(f"{MODULE}.topic_manager")
    async def test_full_pipeline_returns_topic_messages(
        self, mock_tm, mock_repo, manager: TopicManager
    ):
        """摄入消息后，TopicContextTask 应返回同话题的旧消息"""
        msgs = [
            ("今天天气真好适合出门散步", "u1"),
            ("天气好的时候心情也好", "u2"),
            ("这个天气适合去公园", "u1"),
        ]
        for i, (text, sender) in enumerate(msgs):
            await manager.ingest("a1", "g1", None, f"msg_{i}", text)

        # 手动设置 message_entries 的时间戳为过去，模拟已滑出窗口
        old_ts = int(time.time()) - 200
        state = manager._sessions["a1:g:g1"].state
        for cluster in state.clusters.values():
            cluster.message_entries = [
                (mid, old_ts - i * 10)
                for i, (mid, _) in enumerate(cluster.message_entries)
            ]

        query_text = "今天天气怎么样"
        label, msg_ids = await manager.query_topic(
            "a1", "g1", None, query_text, max_count=10, before_timestamp=time.time() - 50,
        )
        mock_tm.query_topic = AsyncMock(return_value=(label, msg_ids))
        mock_repo.get_many_by_message_ids = AsyncMock(
            return_value=_make_items(msg_ids, old_ts)
        )

        task = TopicContextTask(
            agent_id="a1",
            group_id="g1",
            query=query_text,
        )
        result = await task.execute()

        assert len(result) == 1
        stream = result.pop()
        assert isinstance(stream, MemoryItemStream)
        assert stream._role == "topic_context"
        assert stream.priority == pytest.approx(0.2)
        assert len(stream.items) == len(msg_ids)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    @patch(f"{MODULE}.topic_manager")
    async def test_empty_when_no_topic_data(self, mock_tm, mock_repo):
        """没有话题数据时应返回空集合"""
        mock_tm.query_topic = AsyncMock(return_value=(-1, []))

        task = TopicContextTask(agent_id="a1", group_id="g1", query="任意查询")
        result = await task.execute()
        assert result == set()
        mock_repo.get_many_by_message_ids.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    @patch(f"{MODULE}.topic_manager")
    async def test_empty_when_db_returns_nothing(self, mock_tm, mock_repo):
        """DB 查不到消息时应返回空集合"""
        mock_tm.query_topic = AsyncMock(return_value=(0, ["msg_0", "msg_1"]))
        mock_repo.get_many_by_message_ids = AsyncMock(return_value=[])

        task = TopicContextTask(agent_id="a1", group_id="g1", query="天气怎么样")
        result = await task.execute()
        assert result == set()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    @patch(f"{MODULE}.topic_manager")
    async def test_empty_query_returns_empty(self, mock_tm, mock_repo):
        """空 query 应直接返回空集合"""
        task = TopicContextTask(agent_id="a1", group_id="g1", query="")
        result = await task.execute()
        assert result == set()
        mock_tm.query_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_manager_query_topic_returns_correct_messages(self, manager: TopicManager):
        """摄入消息后，query_topic 应返回正确的消息 ID"""
        msgs = ["今天吃了红烧肉很好吃", "晚饭做了糖醋排骨", "午餐去吃了麻辣火锅"]
        for i, msg in enumerate(msgs):
            await manager.ingest("a1", "g1", None, f"food_{i}", msg)

        _, ids = await manager.query_topic("a1", "g1", None, "今天吃了什么")
        assert len(ids) > 0
        assert all(mid.startswith("food_") for mid in ids)
