"""message_queue 和 agent_cache 队列集成测试"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.services.func.message_queue import (
    get_conversation_key,
    create_queue,
    get_queue,
    remove_queue,
    _pending_messages,
)
from nonebot_plugin_wtfllm.services.func.agent_cache import (
    handle_control,
    try_enqueue_message,
    get_conv_key,
)


# ===================== message_queue 单元测试 =====================


class TestGetConversationKey:
    """get_conversation_key 测试"""

    def test_group_key(self):
        """群聊使用 group_id 作为键"""
        key = get_conversation_key("qq", "bot1", "group123", "user456")
        assert key == "qq:bot1:group123"

    def test_private_key(self):
        """私聊使用 p:user_id 作为键"""
        key = get_conversation_key("qq", "bot1", None, "user456")
        assert key == "qq:bot1:p:user456"

    def test_different_groups_different_keys(self):
        """不同群组产生不同的键"""
        k1 = get_conversation_key("qq", "bot1", "g1", "u1")
        k2 = get_conversation_key("qq", "bot1", "g2", "u1")
        assert k1 != k2

    def test_same_group_different_users_same_key(self):
        """同群组不同用户产生相同的键"""
        k1 = get_conversation_key("qq", "bot1", "g1", "u1")
        k2 = get_conversation_key("qq", "bot1", "g1", "u2")
        assert k1 == k2


class TestQueueLifecycle:
    """队列创建/获取/删除生命周期测试"""

    def setup_method(self):
        _pending_messages.clear()

    def teardown_method(self):
        _pending_messages.clear()

    def test_create_queue_returns_empty_list(self):
        """创建队列返回空列表"""
        q = create_queue("test_key")
        assert q == []
        assert isinstance(q, list)

    def test_create_queue_idempotent(self):
        """重复创建返回同一对象"""
        q1 = create_queue("test_key")
        q1.append("item")
        q2 = create_queue("test_key")
        assert q1 is q2
        assert len(q2) == 1

    def test_get_queue_exists(self):
        """获取已存在的队列"""
        created = create_queue("test_key")
        got = get_queue("test_key")
        assert got is created

    def test_get_queue_not_exists(self):
        """获取不存在的队列返回 None"""
        assert get_queue("nonexistent") is None

    def test_remove_queue(self):
        """删除队列后不可再获取"""
        create_queue("test_key")
        removed = remove_queue("test_key")
        assert isinstance(removed, list)
        assert get_queue("test_key") is None

    def test_remove_queue_not_exists(self):
        """删除不存在的队列返回 None"""
        assert remove_queue("nonexistent") is None

    def test_shared_reference(self):
        """create_queue 和 get_queue 返回同一个 list 对象引用"""
        q = create_queue("key")
        ref = get_queue("key")
        q.append("x")
        assert ref == ["x"]


# ===================== agent_cache 队列集成测试 =====================


def _make_bot_and_session(adapter="qq", bot_id="bot1", user_id="u1", group_id="g1"):
    """创建 mock Bot 和 Session"""
    bot = MagicMock()
    bot.adapter.get_name.return_value = adapter
    bot.self_id = bot_id

    session = MagicMock()
    session.user.id = user_id
    if group_id:
        session.group.id = group_id
    else:
        session.group = None

    return bot, session


class TestHandleControlQueue:
    """handle_control 队列生命周期测试"""

    def setup_method(self):
        _pending_messages.clear()

    def teardown_method(self):
        _pending_messages.clear()

    def test_handle_control_creates_queue(self):
        """进入 handle_control 时创建队列"""
        bot, session = _make_bot_and_session()
        conv_key = get_conv_key(bot, session)

        with handle_control(bot, session) as can_handle:
            assert can_handle is True
            assert get_queue(conv_key) is not None

    def test_handle_control_removes_queue_on_exit(self):
        """退出 handle_control 时清理队列"""
        bot, session = _make_bot_and_session()
        conv_key = get_conv_key(bot, session)

        with handle_control(bot, session) as can_handle:
            assert can_handle is True

        assert get_queue(conv_key) is None

    def test_handle_control_removes_queue_on_exception(self):
        """异常退出时也清理队列"""
        bot, session = _make_bot_and_session()
        conv_key = get_conv_key(bot, session)

        with pytest.raises(RuntimeError):
            with handle_control(bot, session):
                raise RuntimeError("boom")

        assert get_queue(conv_key) is None

    def test_handle_control_rejects_concurrent(self):
        """并发请求被拒绝但不创建额外队列"""
        bot, session = _make_bot_and_session()

        with handle_control(bot, session) as first:
            assert first is True

            with handle_control(bot, session) as second:
                assert second is False


class TestTryEnqueueMessage:
    """try_enqueue_message 测试"""

    def setup_method(self):
        _pending_messages.clear()

    def teardown_method(self):
        _pending_messages.clear()

    def test_enqueue_when_agent_running(self):
        """Agent 运行时消息成功入队"""
        bot, session = _make_bot_and_session()
        item = MagicMock()

        with handle_control(bot, session):
            result = try_enqueue_message(bot, session, item)
            assert result is True

            conv_key = get_conv_key(bot, session)
            queue = get_queue(conv_key)
            assert item in queue

    def test_enqueue_when_agent_not_running(self):
        """Agent 未运行时消息不入队"""
        bot, session = _make_bot_and_session()
        item = MagicMock()

        result = try_enqueue_message(bot, session, item)
        assert result is False

    def test_enqueue_multiple_items(self):
        """多条消息按顺序入队"""
        bot, session = _make_bot_and_session()
        items = [MagicMock() for _ in range(3)]

        with handle_control(bot, session):
            for item in items:
                try_enqueue_message(bot, session, item)

            conv_key = get_conv_key(bot, session)
            queue = get_queue(conv_key)
            assert len(queue) == 3
            assert queue == items

    def test_enqueue_from_different_user_same_group(self):
        """同群组不同用户的消息入同一队列"""
        bot_a, session_a = _make_bot_and_session(user_id="user_a", group_id="g1")
        bot_b, session_b = _make_bot_and_session(user_id="user_b", group_id="g1")
        item_a = MagicMock(name="msg_from_a")
        item_b = MagicMock(name="msg_from_b")

        with handle_control(bot_a, session_a):
            try_enqueue_message(bot_a, session_a, item_a)
            # user_b 的消息也应该进入同一群组的队列
            result = try_enqueue_message(bot_b, session_b, item_b)
            assert result is True

            conv_key = get_conv_key(bot_a, session_a)
            queue = get_queue(conv_key)
            assert len(queue) == 2
            assert item_a in queue
            assert item_b in queue

    def test_enqueue_private_chat_isolated(self):
        """私聊队列互相隔离"""
        bot_a, session_a = _make_bot_and_session(user_id="u1", group_id=None)
        bot_b, session_b = _make_bot_and_session(user_id="u2", group_id=None)
        item = MagicMock()

        with handle_control(bot_a, session_a):
            # u2 的私聊没有运行中的 Agent
            result = try_enqueue_message(bot_b, session_b, item)
            assert result is False
