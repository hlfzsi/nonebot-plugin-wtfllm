"""TopicManager 单元测试"""

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
    def test_returns_valid_label(self, manager: TopicManager):
        label = manager.ingest("a1", "g1", None, "msg1", "今天天气真好")
        assert isinstance(label, int)
        assert label >= 0

    def test_empty_text_returns_minus_one(self, manager: TopicManager):
        assert manager.ingest("a1", "g1", None, "msg1", "") == -1
        assert manager.ingest("a1", "g1", None, "msg1", "   ") == -1

    def test_session_isolation(self, manager: TopicManager):
        """不同会话的状态独立"""
        manager.ingest("a1", "g1", None, "msg1", "群聊消息一")
        manager.ingest("a1", "g2", None, "msg2", "另一个群的消息")

        _, ids_g1 = manager.query_topic("a1", "g1", None, "群聊消息一")
        _, ids_g2 = manager.query_topic("a1", "g2", None, "另一个群的消息")

        assert "msg1" in ids_g1
        assert "msg2" in ids_g2
        assert "msg1" not in ids_g2
        assert "msg2" not in ids_g1

    def test_private_vs_group_isolation(self, manager: TopicManager):
        manager.ingest("a1", "g1", None, "msg1", "群聊消息")
        manager.ingest("a1", "u1", None, "msg2", "私聊消息")

        _, ids_group = manager.query_topic("a1", "g1", None, "群聊消息")
        _, ids_private = manager.query_topic("a1", "u1", None, "私聊消息")

        assert "msg1" in ids_group
        assert "msg2" in ids_private
        assert "msg1" not in ids_private


class TestTopicManagerQueryTopic:
    def test_returns_message_ids(self, manager: TopicManager):
        for i in range(5):
            manager.ingest("a1", "g1", None, f"msg{i}", f"话题内容消息{i}")
        label, ids = manager.query_topic(
            "a1", "g1", None, "话题内容消息", max_count=10
        )
        assert label >= 0
        assert len(ids) > 0

    def test_respects_max_count(self, manager: TopicManager):
        for i in range(10):
            manager.ingest("a1", "g1", None, f"msg{i}", "相同话题的消息内容")
        _, ids = manager.query_topic(
            "a1", "g1", None, "相同话题的消息内容", max_count=3
        )
        assert len(ids) <= 3

    def test_before_timestamp_filter(self, manager: TopicManager):
        now = time.time()
        manager.ingest("a1", "g1", None, "msg1", "早期消息内容一")
        _, ids = manager.query_topic(
            "a1", "g1", None, "早期消息内容一", before_timestamp=now - 1
        )
        # 消息刚刚摄入，时间戳 >= now，所以 before_timestamp=now-1 应过滤掉所有
        assert len(ids) == 0

    def test_nonexistent_session_returns_empty(self, manager: TopicManager):
        label, ids = manager.query_topic("a1", "nonexistent", None, "任意文本")
        assert label == -1
        assert ids == []

    def test_does_not_modify_state(self, manager: TopicManager):
        """query_topic 不应修改模型状态"""
        manager.ingest("a1", "g1", None, "msg1", "消息内容")
        state = manager._sessions["a1:g:g1"].state
        ingested_before = state.total_messages_ingested

        manager.query_topic("a1", "g1", None, "查询文本")

        assert state.total_messages_ingested == ingested_before


class TestTopicManagerSummary:
    def test_returns_summary(self, manager: TopicManager):
        manager.ingest("a1", "g1", None, "msg1", "话题一的内容")
        manager.ingest("a1", "g1", None, "msg2", "话题二完全不同")
        summary = manager.get_active_topics_summary("a1", "g1", None)
        assert isinstance(summary, list)
        assert len(summary) > 0
        for item in summary:
            assert "label" in item
            assert "message_count" in item

    def test_nonexistent_session_returns_empty(self, manager: TopicManager):
        assert manager.get_active_topics_summary("a1", "none", None) == []


class TestTopicManagerLRU:
    def test_lru_eviction(self):
        manager = TopicManager(maxsize=3)
        # 填满 3 个 session
        for i in range(3):
            manager.ingest("a1", f"g{i}", None, f"msg{i}", "消息内容")
        # 第 4 个 session 应触发 LRU 驱逐
        manager.ingest("a1", "g3", None, "msg3", "新消息")
        # g0 应被驱逐
        label, ids = manager.query_topic("a1", "g0", None, "消息内容")
        assert label == -1
        assert ids == []


class TestTopicManagerMaintenance:
    def test_maintenance_runs_without_error(self):
        manager = TopicManager(
            maxsize=10,
            maintenance_interval=5,  # 每 5 条消息触发一次维护
            decay_seconds=0.01,  # 极短衰减时间
        )
        for i in range(10):
            manager.ingest("a1", "g1", None, f"msg{i}", f"消息内容{i}")
        # 应该跑过至少一次 _maintenance 而不崩溃
