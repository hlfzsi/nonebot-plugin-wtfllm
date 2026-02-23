"""services/summary.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.services.summary import (
    _format_ts,
    _clean_entity_placeholders,
    _perm,
    handle_summary,
    summary_cmd,
)


MODULE = "nonebot_plugin_wtfllm.services.summary"


# ===================== 工具函数测试 =====================


class TestFormatTs:
    def test_formats_unix_timestamp(self):
        from datetime import datetime

        ts = int(datetime(2025, 6, 15, 14, 30).timestamp())
        result = _format_ts(ts)
        assert result == "2025-06-15 14:30"

    def test_handles_zero_timestamp(self):
        result = _format_ts(0)
        assert isinstance(result, str)
        assert len(result) > 0


class TestCleanEntityPlaceholders:
    def test_replaces_placeholders(self):
        text = "用户 {{Alice}} 说了什么"
        result = _clean_entity_placeholders(text)
        assert result == "用户 Alice 说了什么"

    def test_multiple_placeholders(self):
        text = "{{A}} 和 {{B}}"
        result = _clean_entity_placeholders(text)
        assert result == "A 和 B"

    def test_no_placeholders_unchanged(self):
        text = "普通文本"
        result = _clean_entity_placeholders(text)
        assert result == "普通文本"


# ===================== 权限测试 =====================


class TestPermission:
    @pytest.mark.asyncio
    async def test_admin_returns_true(self):
        session = MagicMock()
        session.user.id = "admin_1"
        with patch(f"{MODULE}.APP_CONFIG") as mock_config:
            mock_config.admin_users = ["admin_1"]
            result = await _perm(session)
        assert result is True

    @pytest.mark.asyncio
    async def test_non_admin_returns_false(self):
        session = MagicMock()
        session.user.id = "normal_user"
        with patch(f"{MODULE}.APP_CONFIG") as mock_config:
            mock_config.admin_users = ["admin_1"]
            result = await _perm(session)
        assert result is False


# ===================== handler 测试 =====================


class FinishCalled(Exception):
    def __init__(self, msg):
        self.message = msg


@pytest.fixture
def mock_finish(monkeypatch):
    """mock summary_cmd.finish 使其抛出异常以模拟 NoneBot FinishedException"""

    async def _finish(msg):
        raise FinishCalled(msg)

    monkeypatch.setattr(summary_cmd, "finish", _finish)


def _make_match(available, result=None):
    m = MagicMock()
    m.available = available
    m.result = result
    return m


def _make_session(user_id="u1", group_id=None, self_id="llonebot:agent1"):
    s = MagicMock()
    s.user.id = user_id
    s.self_id = self_id
    if group_id:
        s.group.id = group_id
    else:
        s.group = None
    return s


class TestHandleSummary:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_query_and_group_with_results(self, mock_repo, mock_finish):
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.item.content = "记忆内容"
        mock_result.item.created_at = 1700000000
        mock_result.item.updated_at = 1700000000
        mock_result.item.source = "chat"
        mock_repo.search_cross_session = AsyncMock(return_value=[mock_result])

        count = _make_match(True, 5)
        query = _make_match(True, "关键词")
        session = _make_session()
        opt_group_id = _make_match(True, "g1")

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        assert "核心记忆 #1" in exc_info.value.message
        assert "0.95" in exc_info.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_query_and_group_no_results(self, mock_repo, mock_finish):
        mock_repo.search_cross_session = AsyncMock(return_value=[])

        count = _make_match(True, 10)
        query = _make_match(True, "不存在")
        session = _make_session()
        opt_group_id = _make_match(True, "g1")

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        assert "未找到" in exc_info.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_group_only_with_results(self, mock_repo, mock_finish):
        mock_memory = MagicMock()
        mock_memory.content = "核心记忆"
        mock_memory.created_at = 1700000000
        mock_memory.updated_at = 1700000000
        mock_memory.source = "chat"
        mock_memory.token_count = 100
        mock_repo.get_by_session = AsyncMock(return_value=[mock_memory])

        count = _make_match(False)
        query = _make_match(False)
        session = _make_session()
        opt_group_id = _make_match(True, "g1")

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        assert "核心记忆 #1" in exc_info.value.message
        assert "共 1 条核心记忆" in exc_info.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_group_only_no_results(self, mock_repo, mock_finish):
        mock_repo.get_by_session = AsyncMock(return_value=[])

        count = _make_match(False)
        query = _make_match(False)
        session = _make_session()
        opt_group_id = _make_match(True, "g1")

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        assert "暂无核心记忆" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_no_group_finishes_with_prompt(self, mock_finish):
        count = _make_match(False)
        query = _make_match(False)
        session = _make_session()
        opt_group_id = _make_match(False)

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        assert "群组" in exc_info.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_group_from_session_default(self, mock_repo, mock_finish):
        mock_repo.get_by_session = AsyncMock(return_value=[])

        count = _make_match(False)
        query = _make_match(False)
        session = _make_session(group_id="g_from_session")
        opt_group_id = _make_match(False)

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        # 调用了 get_by_session，说明从 session 获取了 group_id
        mock_repo.get_by_session.assert_called_once()
        call_kwargs = mock_repo.get_by_session.call_args
        assert call_kwargs.kwargs.get("group_id") == "g_from_session"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.core_memory_repo")
    async def test_total_tokens_displayed(self, mock_repo, mock_finish):
        m1 = MagicMock()
        m1.content = "记忆1"
        m1.created_at = 1700000000
        m1.updated_at = 1700000000
        m1.source = "chat"
        m1.token_count = 50
        m2 = MagicMock()
        m2.content = "记忆2"
        m2.created_at = 1700000000
        m2.updated_at = 1700000000
        m2.source = "chat"
        m2.token_count = 30
        mock_repo.get_by_session = AsyncMock(return_value=[m1, m2])

        count = _make_match(False)
        query = _make_match(False)
        session = _make_session()
        opt_group_id = _make_match(True, "g1")

        with pytest.raises(FinishCalled) as exc_info:
            await handle_summary(count, query, session, opt_group_id)

        assert "总 tokens: 80" in exc_info.value.message
