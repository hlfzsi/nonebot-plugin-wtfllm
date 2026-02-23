"""services/store.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.services.store import handle


MODULE = "nonebot_plugin_wtfllm.services.store"


def _make_session(user_id="u1", group_id=None):
    s = MagicMock()
    s.user.id = user_id
    if group_id:
        s.group.id = group_id
    else:
        s.group = None
    return s


class TestStoreHandler:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.try_enqueue_message")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.like_command", return_value=False)
    @patch(f"{MODULE}.set_alias_to_cache")
    @patch(f"{MODULE}.get_agent_id_from_bot", return_value="agent1")
    async def test_normal_flow(
        self, mock_agent_id, mock_set_alias, mock_like, mock_convert, mock_enqueue
    ):
        bot = MagicMock()
        uni_msg = MagicMock()
        session = _make_session(user_id="u1", group_id="g1")
        msg_id = "msg_1"

        mock_item = MagicMock()
        mock_convert.return_value = mock_item

        await handle(bot, uni_msg, session, msg_id)

        mock_set_alias.assert_called_once_with(bot=bot, session=session)
        mock_convert.assert_called_once()
        mock_enqueue.assert_called_once_with(bot, session, mock_item)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.try_enqueue_message")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.like_command", return_value=True)
    @patch(f"{MODULE}.set_alias_to_cache")
    @patch(f"{MODULE}.get_agent_id_from_bot", return_value="agent1")
    async def test_like_command_returns_early(
        self, mock_agent_id, mock_set_alias, mock_like, mock_convert, mock_enqueue
    ):
        bot = MagicMock()
        uni_msg = MagicMock()
        session = _make_session()
        msg_id = "msg_1"

        await handle(bot, uni_msg, session, msg_id)

        mock_set_alias.assert_called_once()
        mock_convert.assert_not_called()
        mock_enqueue.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.try_enqueue_message")
    @patch(f"{MODULE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE}.like_command", return_value=False)
    @patch(f"{MODULE}.set_alias_to_cache")
    @patch(f"{MODULE}.get_agent_id_from_bot", return_value="agent1")
    async def test_private_chat_no_group_id(
        self, mock_agent_id, mock_set_alias, mock_like, mock_convert, mock_enqueue
    ):
        bot = MagicMock()
        uni_msg = MagicMock()
        session = _make_session(user_id="u1", group_id=None)
        msg_id = "msg_1"

        mock_convert.return_value = MagicMock()
        await handle(bot, uni_msg, session, msg_id)

        call_kwargs = mock_convert.call_args.kwargs
        assert call_kwargs["group_id"] is None
        assert call_kwargs["user_id"] == "u1"
