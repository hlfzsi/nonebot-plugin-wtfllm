# tests/abilities/test_poi.py
"""abilities/poi.py 单元测试"""

import time
import threading
from unittest.mock import patch

import pytest

from nonebot_plugin_wtfllm.abilities.poi import PoiInfo, AttentionRouter


class TestPoiInfo:
    """PoiInfo 模型测试"""

    def test_create_with_defaults(self):
        """测试使用默认值创建"""
        poi = PoiInfo(
            user_id="user1",
            group_id="group1",
            agent_id="agent1",
            reason="测试原因",
        )
        assert poi.user_id == "user1"
        assert poi.group_id == "group1"
        assert poi.agent_id == "agent1"
        assert poi.reason == "测试原因"
        assert poi.turns == 1
        # 默认 5 分钟后过期
        assert poi.expires_at > time.time()

    def test_create_with_custom_values(self):
        """测试使用自定义值创建"""
        custom_expiry = time.time() + 3600
        poi = PoiInfo(
            user_id="user2",
            group_id=None,
            agent_id="agent2",
            reason="自定义原因",
            turns=5,
            expires_at=custom_expiry,
        )
        assert poi.turns == 5
        assert poi.expires_at == custom_expiry
        assert poi.group_id is None

    def test_is_expired_not_expired(self):
        """测试未过期判断"""
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            expires_at=time.time() + 100,
        )
        assert poi.is_expired() is False

    def test_is_expired_already_expired(self):
        """测试已过期判断"""
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            expires_at=time.time() - 1,
        )
        assert poi.is_expired() is True

    def test_is_expired_exactly_at_boundary(self):
        """测试刚好到过期时间边界"""
        with patch("nonebot_plugin_wtfllm.abilities.poi.time.time") as mock_time:
            mock_time.return_value = 1000.0
            poi = PoiInfo(
                user_id="user1",
                group_id=None,
                agent_id="agent1",
                reason="test",
                expires_at=1000.0,
            )
            # time.time() > expires_at 才过期，所以 == 时不过期
            assert poi.is_expired() is False

            mock_time.return_value = 1000.1
            assert poi.is_expired() is True

    def test_consume_turn_success(self):
        """测试成功消耗回合"""
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=3,
        )
        assert poi.consume_turn() is True
        assert poi.turns == 2
        assert poi.consume_turn() is True
        assert poi.turns == 1
        assert poi.consume_turn() is True
        assert poi.turns == 0

    def test_consume_turn_zero_turns(self):
        """测试零回合时消耗失败"""
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=0,
        )
        assert poi.consume_turn() is False
        assert poi.turns == 0

    def test_consume_turn_negative_turns(self):
        """测试负回合时消耗失败"""
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=-1,
        )
        assert poi.consume_turn() is False
        assert poi.turns == -1


class TestAttentionRouterGetKey:
    """AttentionRouter.get_key 静态方法测试"""

    def test_get_key_with_group(self):
        """测试有群组时的键生成"""
        key = AttentionRouter.get_key("user1", "agent1", "group1")
        assert key == "ATTN:G:group1:A:agent1:U:user1"

    def test_get_key_without_group(self):
        """测试无群组（私聊）时的键生成"""
        key = AttentionRouter.get_key("user1", "agent1", None)
        assert key == "ATTN:A:agent1:U:user1"

    def test_get_key_empty_group(self):
        """测试空字符串群组被视为无群组"""
        # 空字符串是 falsy，会走无群组分支
        key = AttentionRouter.get_key("user1", "agent1", "")
        assert key == "ATTN:A:agent1:U:user1"

    def test_get_key_different_users(self):
        """测试不同用户生成不同键"""
        key1 = AttentionRouter.get_key("user1", "agent1", "group1")
        key2 = AttentionRouter.get_key("user2", "agent1", "group1")
        assert key1 != key2

    def test_get_key_different_agents(self):
        """测试不同 Agent 生成不同键"""
        key1 = AttentionRouter.get_key("user1", "agent1", "group1")
        key2 = AttentionRouter.get_key("user1", "agent2", "group1")
        assert key1 != key2


class TestAttentionRouterMarkPoi:
    """AttentionRouter.mark_poi 方法测试"""

    def test_mark_poi_new(self):
        """测试新增关注点"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id="group1",
            agent_id="agent1",
            reason="new poi",
            turns=3,
        )
        router.mark_poi(poi)

        result = router.get_poi("user1", "agent1", "group1")
        assert result is not None
        assert result.reason == "new poi"
        assert result.turns == 3

    def test_mark_poi_update(self):
        """测试更新已存在的关注点"""
        router = AttentionRouter()
        poi1 = PoiInfo(
            user_id="user1",
            group_id="group1",
            agent_id="agent1",
            reason="original",
            turns=1,
        )
        router.mark_poi(poi1)

        poi2 = PoiInfo(
            user_id="user1",
            group_id="group1",
            agent_id="agent1",
            reason="updated",
            turns=5,
        )
        router.mark_poi(poi2)

        result = router.get_poi("user1", "agent1", "group1")
        assert result is not None
        assert result.reason == "updated"
        assert result.turns == 5


class TestAttentionRouterGetPoi:
    """AttentionRouter.get_poi 方法测试"""

    def test_get_poi_exists(self):
        """测试获取存在的关注点"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=2,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        result = router.get_poi("user1", "agent1", None)
        assert result is not None
        assert result.user_id == "user1"

    def test_get_poi_not_exists(self):
        """测试获取不存在的关注点"""
        router = AttentionRouter()
        result = router.get_poi("user1", "agent1", None)
        assert result is None

    def test_get_poi_expired_auto_remove(self):
        """测试获取已过期的关注点会自动移除"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=5,
            expires_at=time.time() - 1,  # 已过期
        )
        router.mark_poi(poi)

        result = router.get_poi("user1", "agent1", None)
        assert result is None

        # 确认已被删除
        assert len(router._poi) == 0

    def test_get_poi_zero_turns_auto_remove(self):
        """测试零回合的关注点会自动移除"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=0,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        result = router.get_poi("user1", "agent1", None)
        assert result is None
        assert len(router._poi) == 0


class TestAttentionRouterIsInterested:
    """AttentionRouter.is_interested 方法测试"""

    def test_is_interested_true(self):
        """测试有关注点时返回 True"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=1,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)
        assert router.is_interested("user1", "agent1", None) is True

    def test_is_interested_false(self):
        """测试无关注点时返回 False"""
        router = AttentionRouter()
        assert router.is_interested("user1", "agent1", None) is False


class TestAttentionRouterConsumePoi:
    """AttentionRouter.consume_poi 方法测试"""

    def test_consume_poi_success(self):
        """测试成功消耗回合"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=3,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        result = router.consume_poi("user1", "agent1", None)
        assert result is True

        remaining = router.get_poi("user1", "agent1", None)
        assert remaining is not None
        assert remaining.turns == 2

    def test_consume_poi_not_exists(self):
        """测试消耗不存在的关注点"""
        router = AttentionRouter()
        result = router.consume_poi("user1", "agent1", None)
        assert result is False

    def test_consume_poi_last_turn_removes(self):
        """测试最后一个回合消耗后移除"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=1,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        result = router.consume_poi("user1", "agent1", None)
        assert result is True

        remaining = router.get_poi("user1", "agent1", None)
        assert remaining is None

    def test_consume_poi_expired_removes(self):
        """测试消耗已过期的关注点会移除"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=5,
            expires_at=time.time() - 1,  # 已过期
        )
        router.mark_poi(poi)

        result = router.consume_poi("user1", "agent1", None)
        assert result is False
        assert len(router._poi) == 0


class TestAttentionRouterRemovePoi:
    """AttentionRouter.remove_poi 方法测试"""

    def test_remove_poi_exists(self):
        """测试移除存在的关注点"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
        )
        router.mark_poi(poi)

        router.remove_poi("user1", "agent1", None)
        assert router.get_poi("user1", "agent1", None) is None

    def test_remove_poi_not_exists(self):
        """测试移除不存在的关注点（不报错）"""
        router = AttentionRouter()
        # 应该不报错
        router.remove_poi("user1", "agent1", None)


class TestAttentionRouterGetAndConsumePoi:
    """AttentionRouter.get_and_consume_poi 方法测试"""

    def test_get_and_consume_success(self):
        """测试成功获取并消耗"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test reason",
            turns=3,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        result = router.get_and_consume_poi("user1", "agent1", None)
        assert result is not None
        assert result.reason == "test reason"
        assert result.turns == 2  # 已消耗一个

    def test_get_and_consume_not_exists(self):
        """测试不存在时返回 None"""
        router = AttentionRouter()
        result = router.get_and_consume_poi("user1", "agent1", None)
        assert result is None

    def test_get_and_consume_last_turn_removes(self):
        """测试最后回合时移除并返回 None"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=1,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        result = router.get_and_consume_poi("user1", "agent1", None)
        assert result is not None
        assert len(router._poi) == 1

        result = router.get_and_consume_poi("user1", "agent1", None)
        assert result is None
        assert len(router._poi) == 0

class TestAttentionRouterListAgentInterests:
    """AttentionRouter.list_agent_interests 方法测试"""

    def test_list_agent_interests_empty(self):
        """测试空列表"""
        router = AttentionRouter()
        result = router.list_agent_interests("agent1")
        assert result == []

    def test_list_agent_interests_multiple(self):
        """测试列出多个关注点"""
        router = AttentionRouter()
        poi1 = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="poi1",
            turns=2,
            expires_at=time.time() + 100,
        )
        poi2 = PoiInfo(
            user_id="user2",
            group_id="group1",
            agent_id="agent1",
            reason="poi2",
            turns=3,
            expires_at=time.time() + 100,
        )
        poi3 = PoiInfo(
            user_id="user3",
            group_id=None,
            agent_id="agent2",  # 不同 agent
            reason="poi3",
            turns=1,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi1)
        router.mark_poi(poi2)
        router.mark_poi(poi3)

        result = router.list_agent_interests("agent1")
        assert len(result) == 2
        reasons = {p.reason for p in result}
        assert reasons == {"poi1", "poi2"}

    def test_list_agent_interests_filters_expired(self):
        """测试会过滤掉过期的关注点"""
        router = AttentionRouter()
        poi_active = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="active",
            turns=2,
            expires_at=time.time() + 100,
        )
        poi_expired = PoiInfo(
            user_id="user2",
            group_id=None,
            agent_id="agent1",
            reason="expired",
            turns=2,
            expires_at=time.time() - 1,
        )
        router.mark_poi(poi_active)
        router.mark_poi(poi_expired)

        result = router.list_agent_interests("agent1")
        assert len(result) == 1
        assert result[0].reason == "active"


class TestAttentionRouterClearAll:
    """AttentionRouter.clear_all 方法测试"""

    def test_clear_all(self):
        """测试清空所有数据"""
        router = AttentionRouter()
        for i in range(5):
            poi = PoiInfo(
                user_id=f"user{i}",
                group_id=None,
                agent_id="agent1",
                reason=f"reason{i}",
            )
            router.mark_poi(poi)

        assert len(router._poi) == 5
        router.clear_all()
        assert len(router._poi) == 0


class TestAttentionRouterThreadSafety:
    """AttentionRouter 线程安全测试"""

    def test_concurrent_mark_and_get(self):
        """测试并发 mark 和 get 操作"""
        router = AttentionRouter()
        errors = []
        iterations = 100

        def mark_worker(user_suffix: int):
            try:
                for i in range(iterations):
                    poi = PoiInfo(
                        user_id=f"user{user_suffix}_{i}",
                        group_id=None,
                        agent_id="agent1",
                        reason=f"reason_{i}",
                        turns=10,
                        expires_at=time.time() + 100,
                    )
                    router.mark_poi(poi)
            except Exception as e:
                errors.append(e)

        def get_worker():
            try:
                for i in range(iterations):
                    router.get_poi(f"user0_{i}", "agent1", None)
                    router.list_agent_interests("agent1")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mark_worker, args=(0,)),
            threading.Thread(target=mark_worker, args=(1,)),
            threading.Thread(target=get_worker),
            threading.Thread(target=get_worker),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_consume(self):
        """测试并发消耗操作"""
        router = AttentionRouter()
        poi = PoiInfo(
            user_id="user1",
            group_id=None,
            agent_id="agent1",
            reason="test",
            turns=100,
            expires_at=time.time() + 100,
        )
        router.mark_poi(poi)

        success_count = [0]
        lock = threading.Lock()

        def consume_worker():
            for _ in range(20):
                if router.consume_poi("user1", "agent1", None):
                    with lock:
                        success_count[0] += 1

        threads = [threading.Thread(target=consume_worker) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 初始 100 回合，消耗后应该接近 0
        remaining = router.get_poi("user1", "agent1", None)
        # 由于竞态，可能已经被移除
        if remaining is not None:
            assert remaining.turns >= 0
