"""utils.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.utils import (
    get_agent_id_from_bot,
    ensure_msgid_from_receipt,
    get_http_client,
    init_http_client,
    shutdown_http_client,
    count_tokens,
)


# Create a real class to stand in for Session so isinstance() works
class _FakeSession:
    def __init__(self, self_id: str):
        self.self_id = self_id


class TestGetAgentIdFromBot:
    def test_strips_llonebot_prefix(self):
        bot = MagicMock()
        bot.self_id = "llonebot:12345"
        with patch("nonebot_plugin_wtfllm.utils.Session", _FakeSession):
            assert get_agent_id_from_bot(bot) == "12345"

    def test_strips_napcat_prefix(self):
        bot = MagicMock()
        bot.self_id = "napcat:67890"
        with patch("nonebot_plugin_wtfllm.utils.Session", _FakeSession):
            assert get_agent_id_from_bot(bot) == "67890"

    def test_no_prefix(self):
        bot = MagicMock()
        bot.self_id = "plain_id"
        with patch("nonebot_plugin_wtfllm.utils.Session", _FakeSession):
            assert get_agent_id_from_bot(bot) == "plain_id"

    def test_session_with_llonebot_prefix(self):
        session = _FakeSession("llonebot:session123")
        with patch("nonebot_plugin_wtfllm.utils.Session", _FakeSession):
            assert get_agent_id_from_bot(session) == "session123"

    def test_session_with_napcat_prefix(self):
        session = _FakeSession("napcat:session456")
        with patch("nonebot_plugin_wtfllm.utils.Session", _FakeSession):
            assert get_agent_id_from_bot(session) == "session456"

    def test_session_no_prefix(self):
        session = _FakeSession("plain_session_id")
        with patch("nonebot_plugin_wtfllm.utils.Session", _FakeSession):
            assert get_agent_id_from_bot(session) == "plain_session_id"


class TestEnsureMsgidFromReceipt:
    def test_with_reply(self):
        receipt = MagicMock()
        reply = MagicMock()
        reply.id = "msg_abc123"
        receipt.get_reply = MagicMock(return_value=reply)
        result = ensure_msgid_from_receipt(receipt)
        assert result == "msg_abc123"

    def test_without_reply_returns_fake(self):
        receipt = MagicMock()
        receipt.get_reply = MagicMock(return_value=None)
        result = ensure_msgid_from_receipt(receipt)
        assert result.startswith("fake_")
        assert len(result) > 10

    def test_unique_fake_ids(self):
        receipt = MagicMock()
        receipt.get_reply = MagicMock(return_value=None)
        ids = {ensure_msgid_from_receipt(receipt) for _ in range(50)}
        assert len(ids) == 50


class TestHttpClient:
    def test_get_raises_when_not_initialized(self):
        import nonebot_plugin_wtfllm.utils as utils_mod
        original = utils_mod._http_client
        try:
            utils_mod._http_client = None
            with pytest.raises(RuntimeError, match="not initialized"):
                get_http_client()
        finally:
            utils_mod._http_client = original

    def test_get_returns_client(self):
        import nonebot_plugin_wtfllm.utils as utils_mod
        original = utils_mod._http_client
        mock_client = MagicMock()
        try:
            utils_mod._http_client = mock_client
            assert get_http_client() is mock_client
        finally:
            utils_mod._http_client = original

    @pytest.mark.asyncio
    async def test_init_creates_client(self):
        import nonebot_plugin_wtfllm.utils as utils_mod
        original = utils_mod._http_client
        try:
            utils_mod._http_client = None
            with patch("nonebot_plugin_wtfllm.utils.httpx.AsyncClient") as mock_cls:
                mock_instance = MagicMock()
                mock_cls.return_value = mock_instance
                await init_http_client()
                assert utils_mod._http_client is mock_instance
                mock_cls.assert_called_once_with(follow_redirects=True, timeout=60)
        finally:
            utils_mod._http_client = original

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self):
        import nonebot_plugin_wtfllm.utils as utils_mod
        original = utils_mod._http_client
        mock_client = AsyncMock()
        try:
            utils_mod._http_client = mock_client
            await shutdown_http_client()
            mock_client.aclose.assert_called_once()
            assert utils_mod._http_client is None
        finally:
            utils_mod._http_client = original

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_none(self):
        import nonebot_plugin_wtfllm.utils as utils_mod
        original = utils_mod._http_client
        try:
            utils_mod._http_client = None
            await shutdown_http_client()  # Should not raise
            assert utils_mod._http_client is None
        finally:
            utils_mod._http_client = original


class TestCountTokens:
    def test_basic(self):
        result = count_tokens("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string(self):
        assert count_tokens("") == 0
