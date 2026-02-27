"""话题聚类数据模型单元测试"""

import time

import pytest

from nonebot_plugin_wtfllm.topic._types import (
    SessionKey,
    TopicCluster,
    TopicSessionState,
)


# ===================== SessionKey =====================


class TestSessionKey:

    def test_group_session_cache_key(self):
        sk = SessionKey(agent_id="a1", group_id="g1")
        assert sk.cache_key == "a1:g:g1"

    def test_private_session_cache_key(self):
        sk = SessionKey(agent_id="a1", user_id="u1")
        assert sk.cache_key == "a1:u:u1"

    def test_hash_eq_same(self):
        a = SessionKey(agent_id="a1", group_id="g1")
        b = SessionKey(agent_id="a1", group_id="g1")
        assert a == b
        assert hash(a) == hash(b)

    def test_hash_neq_different_group(self):
        a = SessionKey(agent_id="a1", group_id="g1")
        b = SessionKey(agent_id="a1", group_id="g2")
        assert a != b

    def test_hash_neq_group_vs_private(self):
        a = SessionKey(agent_id="a1", group_id="g1")
        b = SessionKey(agent_id="a1", user_id="u1")
        assert a != b

    def test_usable_as_dict_key(self):
        sk = SessionKey(agent_id="a1", group_id="g1")
        d = {sk: 42}
        assert d[SessionKey(agent_id="a1", group_id="g1")] == 42

    def test_eq_with_non_session_key(self):
        sk = SessionKey(agent_id="a1", group_id="g1")
        assert sk != "not a session key"


# ===================== TopicCluster =====================


class TestTopicCluster:

    def test_construction_defaults(self):
        cluster = TopicCluster(label=0)
        assert cluster.label == 0
        assert cluster.message_entries == []
        assert cluster.message_count == 0

    def test_message_entries(self):
        cluster = TopicCluster(label=1)
        cluster.message_entries.append(("msg1", time.time()))
        cluster.message_count += 1
        assert len(cluster.message_entries) == 1
        assert cluster.message_count == 1


# ===================== TopicSessionState =====================


class TestTopicSessionState:

    def test_construction_defaults(self):
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)
        assert state.total_messages_ingested == 0
        assert state.clusters == {}

    def test_add_cluster(self):
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)
        cluster = TopicCluster(label=0)
        state.clusters[0] = cluster
        assert 0 in state.clusters
        assert state.clusters[0].label == 0
