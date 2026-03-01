"""services/func/agent_cache.py 单元测试"""

import pytest
from unittest.mock import MagicMock, patch

from nonebot_plugin_wtfllm.services.func.agent_cache import (
    get_handle_key,
    get_conv_key,
    handle_control,
    try_enqueue_message,
    get_alias_cache_main_key,
    get_alias_from_cache,
    set_alias_to_cache,
    _in_handle,
    _alias_cache,
    _group_alias_cache,
)


def _make_bot_session(
    adapter="test", bot_id="bot1", user_id="u1",
    group_id=None, nick=None, name=None, group_name=None,
):
    bot = MagicMock()
    bot.adapter.get_name.return_value = adapter
    bot.self_id = bot_id
    session = MagicMock()
    session.user.id = user_id
    session.user.nick = nick
    session.user.name = name
    if group_id:
        session.scene.is_private = False
        session.scene_path = group_id
        session.scene.name = group_name
        session.group.id = group_id
        session.group.name = group_name
    else:
        session.scene.is_private = True
        session.scene_path = user_id
        session.group = None
    return bot, session


@pytest.fixture(autouse=True)
def _clean_caches():
    _in_handle.clear()
    _alias_cache.clear()
    _group_alias_cache.clear()
    yield
    _in_handle.clear()
    _alias_cache.clear()
    _group_alias_cache.clear()


MODULE = "nonebot_plugin_wtfllm.services.func.agent_cache"


class TestGetHandleKey:
    def test_group_chat(self):
        bot, session = _make_bot_session(group_id="g1")
        key = get_handle_key(bot, session)
        assert key == "test:bot1:u1:g1"

    def test_private_chat(self):
        bot, session = _make_bot_session()
        key = get_handle_key(bot, session)
        assert key == "test:bot1:u1:private"


class TestGetConvKey:
    @patch(f"{MODULE}.get_conversation_key", return_value="conv_key_1")
    def test_group_chat(self, mock_conv):
        bot, session = _make_bot_session(group_id="g1")
        result = get_conv_key(bot, session)
        assert result == "conv_key_1"
        mock_conv.assert_called_once_with("test", "bot1", "g1", "u1")

    @patch(f"{MODULE}.get_conversation_key", return_value="conv_key_2")
    def test_private_chat(self, mock_conv):
        bot, session = _make_bot_session()
        result = get_conv_key(bot, session)
        assert result == "conv_key_2"
        mock_conv.assert_called_once_with("test", "bot1", None, "u1")


class TestHandleControl:
    @patch(f"{MODULE}.remove_queue")
    @patch(f"{MODULE}.create_queue")
    @patch(f"{MODULE}.get_conv_key", return_value="conv1")
    def test_first_entry_yields_true(self, mock_conv, mock_create, mock_remove):
        bot, session = _make_bot_session()
        with handle_control(bot, session) as entered:
            assert entered is True
            handle_key = get_handle_key(bot, session)
            assert handle_key in _in_handle
        assert handle_key not in _in_handle
        mock_create.assert_called_once_with("conv1")
        mock_remove.assert_called_once_with("conv1")

    @patch(f"{MODULE}.remove_queue")
    @patch(f"{MODULE}.create_queue")
    @patch(f"{MODULE}.get_conv_key", return_value="conv1")
    def test_concurrent_entry_yields_false(self, mock_conv, mock_create, mock_remove):
        bot, session = _make_bot_session()
        handle_key = get_handle_key(bot, session)
        _in_handle.add(handle_key)
        with handle_control(bot, session) as entered:
            assert entered is False
        # Key should still be in _in_handle (not removed by the False branch)
        assert handle_key in _in_handle


class TestTryEnqueueMessage:
    @patch(f"{MODULE}.get_queue")
    @patch(f"{MODULE}.get_conv_key", return_value="conv1")
    def test_queue_exists(self, mock_conv, mock_get_queue):
        queue = MagicMock()
        mock_get_queue.return_value = queue
        bot, session = _make_bot_session()
        item = MagicMock()
        result = try_enqueue_message(bot, session, item)
        assert result is True
        queue.append.assert_called_once_with(item)

    @patch(f"{MODULE}.get_queue", return_value=None)
    @patch(f"{MODULE}.get_conv_key", return_value="conv1")
    def test_no_queue(self, mock_conv, mock_get_queue):
        bot, session = _make_bot_session()
        result = try_enqueue_message(bot, session, MagicMock())
        assert result is False


class TestGetAliasCacheMainKey:
    def test_group(self):
        bot, session = _make_bot_session(group_id="g1")
        key = get_alias_cache_main_key(bot, session)
        assert key == "test:bot1:g:g1"

    def test_private(self):
        bot, session = _make_bot_session()
        key = get_alias_cache_main_key(bot, session)
        assert key == "test:bot1:p:u1"


class TestAliasCacheOps:
    def test_set_and_get_user_alias_by_nick(self):
        bot, session = _make_bot_session(nick="Alice")
        set_alias_to_cache(bot, session)
        result = get_alias_from_cache(bot, session)
        assert result["u1"] == "Alice"

    def test_set_user_alias_by_name(self):
        bot, session = _make_bot_session(name="Bob")
        set_alias_to_cache(bot, session)
        result = get_alias_from_cache(bot, session)
        assert result["u1"] == "Bob"

    def test_nick_takes_priority(self):
        bot, session = _make_bot_session(nick="Nick", name="Name")
        set_alias_to_cache(bot, session)
        result = get_alias_from_cache(bot, session)
        assert result["u1"] == "Nick"

    def test_no_alias(self):
        bot, session = _make_bot_session(nick=None, name=None)
        set_alias_to_cache(bot, session)
        result = get_alias_from_cache(bot, session)
        assert "u1" not in result

    def test_group_alias_cached(self):
        bot, session = _make_bot_session(group_id="g1", group_name="TestGroup")
        set_alias_to_cache(bot, session)
        result = get_alias_from_cache(bot, session)
        assert result.get("g1") == "TestGroup"

    def test_group_alias_shared_across_sessions(self):
        # Set group alias from one session
        bot1, session1 = _make_bot_session(
            user_id="u1", group_id="g1", group_name="MyGroup", nick="A"
        )
        set_alias_to_cache(bot1, session1)

        # Get from different user in same group
        bot2, session2 = _make_bot_session(
            user_id="u2", group_id="g1", group_name="MyGroup", nick="B"
        )
        set_alias_to_cache(bot2, session2)

        result = get_alias_from_cache(bot2, session2)
        # Should see own user alias and shared group alias
        assert result.get("u2") == "B"
        assert result.get("g1") == "MyGroup"
