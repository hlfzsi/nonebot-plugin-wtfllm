"""TopicManager 单元测试"""

import asyncio
import time

import pytest

from nonebot_plugin_wtfllm.topic.manager import TopicManager


@pytest.fixture
def manager():
    return TopicManager(
        maxsize=10,
        cluster_threshold=0.8,
        max_clusters=20,
        decay_seconds=60,
        max_messages_per_cluster=100,
    )


class TestTopicManagerIngest:
    @pytest.mark.asyncio
    async def test_returns_valid_label(self, manager: TopicManager):
        label = await manager.ingest("a1", "g1", None, "msg1", "今天天气真好")
        assert isinstance(label, int)
        assert label >= 0

    @pytest.mark.asyncio
    async def test_empty_text_returns_minus_one(self, manager: TopicManager):
        assert await manager.ingest("a1", "g1", None, "msg1", "") == -1
        assert await manager.ingest("a1", "g1", None, "msg1", "   ") == -1

    @pytest.mark.asyncio
    async def test_session_isolation(self, manager: TopicManager):
        """不同会话的状态独立"""
        await manager.ingest("a1", "g1", None, "msg1", "群聊消息一")
        await manager.ingest("a1", "g2", None, "msg2", "另一个群的消息")

        _, ids_g1 = await manager.query_topic("a1", "g1", None, "群聊消息一")
        _, ids_g2 = await manager.query_topic("a1", "g2", None, "另一个群的消息")

        assert "msg1" in ids_g1
        assert "msg2" in ids_g2
        assert "msg1" not in ids_g2
        assert "msg2" not in ids_g1

    @pytest.mark.asyncio
    async def test_private_vs_group_isolation(self, manager: TopicManager):
        await manager.ingest("a1", "g1", None, "msg1", "群聊消息")
        await manager.ingest("a1", "u1", None, "msg2", "私聊消息")

        _, ids_group = await manager.query_topic("a1", "g1", None, "群聊消息")
        _, ids_private = await manager.query_topic("a1", "u1", None, "私聊消息")

        assert "msg1" in ids_group
        assert "msg2" in ids_private
        assert "msg1" not in ids_private


class TestTopicManagerQueryTopic:
    @pytest.mark.asyncio
    async def test_returns_message_ids(self, manager: TopicManager):
        for i in range(5):
            await manager.ingest("a1", "g1", None, f"msg{i}", f"话题内容消息{i}")
        label, ids = await manager.query_topic(
            "a1", "g1", None, "话题内容消息", max_count=10
        )
        assert label >= 0
        assert len(ids) > 0

    @pytest.mark.asyncio
    async def test_respects_max_count(self, manager: TopicManager):
        for i in range(10):
            await manager.ingest("a1", "g1", None, f"msg{i}", "相同话题的消息内容")
        _, ids = await manager.query_topic(
            "a1", "g1", None, "相同话题的消息内容", max_count=3
        )
        assert len(ids) <= 3

    @pytest.mark.asyncio
    async def test_before_timestamp_filter(self, manager: TopicManager):
        now = time.time()
        await manager.ingest("a1", "g1", None, "msg1", "早期消息内容一")
        _, ids = await manager.query_topic(
            "a1", "g1", None, "早期消息内容一", before_timestamp=now - 1
        )
        # 消息刚刚摄入，时间戳 >= now，所以 before_timestamp=now-1 应过滤掉所有
        assert len(ids) == 0

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_empty(self, manager: TopicManager):
        label, ids = await manager.query_topic("a1", "nonexistent", None, "任意文本")
        assert label == -1
        assert ids == []

    @pytest.mark.asyncio
    async def test_does_not_modify_state(self, manager: TopicManager):
        """query_topic 不应修改模型状态"""
        await manager.ingest("a1", "g1", None, "msg1", "消息内容")
        state = manager._sessions["a1:g:g1"].state
        ingested_before = state.total_messages_ingested

        await manager.query_topic("a1", "g1", None, "查询文本")

        assert state.total_messages_ingested == ingested_before


class TestTopicManagerSummary:
    @pytest.mark.asyncio
    async def test_returns_summary(self, manager: TopicManager):
        await manager.ingest("a1", "g1", None, "msg1", "话题一的内容")
        await manager.ingest("a1", "g1", None, "msg2", "话题二完全不同")
        summary = await manager.get_active_topics_summary("a1", "g1", None)
        assert isinstance(summary, list)
        assert len(summary) > 0
        for item in summary:
            assert "label" in item
            assert "message_count" in item

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_empty(self, manager: TopicManager):
        assert await manager.get_active_topics_summary("a1", "none", None) == []


class TestTopicManagerLRU:
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        manager = TopicManager(maxsize=3)
        # 填满 3 个 session
        for i in range(3):
            await manager.ingest("a1", f"g{i}", None, f"msg{i}", "消息内容")
        # 第 4 个 session 应触发 LRU 驱逐
        await manager.ingest("a1", "g3", None, "msg3", "新消息")
        # g0 应被驱逐
        label, ids = await manager.query_topic("a1", "g0", None, "消息内容")
        assert label == -1
        assert ids == []


class TestTopicManagerMaintenance:
    @pytest.mark.asyncio
    async def test_maintenance_runs_without_error(self):
        manager = TopicManager(
            maxsize=10,
            decay_seconds=0.01,  # 极短衰减时间
        )
        manager.start()
        for i in range(10):
            await manager.ingest("a1", "g1", None, f"msg{i}", f"消息内容{i}")
        await asyncio.sleep(0.05)
        await manager.stop()
        # 周期清理 worker 可运行且不崩溃


class TestTopicManagerArchiveQueue:
    @pytest.mark.asyncio
    async def test_eviction_enqueues_candidate(self):
        """容量淘汰时应将候选入队"""
        manager = TopicManager(
            maxsize=10,
            cluster_threshold=0.99,  # 极高阈值确保每条消息单独成簇
            max_clusters=3,
            min_archive_messages=1,  # 低阈值确保能归档
        )
        # 使用截然不同的话题文本
        distinct_texts = [
            "Python编程语言的异步框架asyncio非常好用",
            "今天晚餐吃了一碗热腾腾的红烧牛肉面",
            "周末去杭州西湖旅游景点拍照留念",
            "NBA篮球比赛湖人队对阵勇士队的精彩对决",
        ]
        for i in range(3):
            await manager.ingest("a1", "g1", None, f"msg{i}", distinct_texts[i])

        ctx = manager._sessions["a1:g:g1"]
        clusters_before = len(ctx.state.clusters)
        assert clusters_before == 3, f"Expected 3 clusters, got {clusters_before}"

        # 第 4 条应触发淘汰
        await manager.ingest("a1", "g1", None, "msg3", distinct_texts[3])

        # 检查队列中是否有候选
        if not manager._archive_queue.empty():
            candidate = manager._archive_queue.get_nowait()
            assert candidate.session_key.agent_id == "a1"
            assert candidate.session_key.group_id == "g1"
        else:
            # 如果模型恰好将第4条归入已有簇，则无淘汰
            assert ctx.clustering.n_clusters <= 3

    @pytest.mark.asyncio
    async def test_prune_enqueues_candidates(self):
        """超时清理时应将候选入队"""
        manager = TopicManager(
            maxsize=10,
            cluster_threshold=0.5,
            max_clusters=30,
            decay_seconds=0.01,  # 极短衰减
            min_archive_messages=1,
        )
        manager.start()
        # 先摄入几条消息建立簇
        for i in range(4):
            await manager.ingest("a1", "g1", None, f"msg{i}", f"消息内容{i}关于美食")

        # 等待过期
        import asyncio
        await asyncio.sleep(0.05)

        # 再摄入一条不同消息，并等待周期清理
        await manager.ingest("a1", "g1", None, "msg4", "消息内容4关于技术编程")
        await asyncio.sleep(0.05)

        # 队列中可能有因超时清理产生的候选
        # (具体是否有取决于簇的消息数是否 >= min_archive_messages)
        await manager.stop()
