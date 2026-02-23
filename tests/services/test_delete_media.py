"""services/delete_media.py 单元测试"""

import time

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.services.delete_media import _unbound, _perm


MODULE = "nonebot_plugin_wtfllm.services.delete_media"


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


# ===================== _unbound 测试 =====================


class TestUnbound:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_no_items_returns_zero(self, mock_repo):
        mock_repo.get_by_timestamp_before = AsyncMock(return_value=[])
        mock_repo.save_many = AsyncMock(return_value=0)

        count = await _unbound(agent_id="a1", expiry_days=7)
        assert count == 0

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_items_without_media_returns_zero(self, mock_repo):
        item = MagicMock()
        item.content.deep_get = MagicMock(return_value=[])
        mock_repo.get_by_timestamp_before = AsyncMock(return_value=[item])
        mock_repo.save_many = AsyncMock(return_value=0)

        count = await _unbound(agent_id="a1", expiry_days=7)
        assert count == 0
        mock_repo.save_many.assert_called_once_with([])

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_items_with_media_calls_unbound_local(self, mock_repo):
        seg = MagicMock()
        item = MagicMock()
        item.content.deep_get = MagicMock(return_value=[seg])
        mock_repo.get_by_timestamp_before = AsyncMock(return_value=[item])
        mock_repo.save_many = AsyncMock(return_value=1)

        count = await _unbound(agent_id="a1", expiry_days=7)
        assert count == 1
        seg.unbound_local.assert_called_once_with(expired=True)
        mock_repo.save_many.assert_called_once_with([item])

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_correct_expiry_timestamp(self, mock_repo):
        mock_repo.get_by_timestamp_before = AsyncMock(return_value=[])
        mock_repo.save_many = AsyncMock(return_value=0)

        now = int(time.time())
        await _unbound(agent_id="a1", expiry_days=3)

        call_args = mock_repo.get_by_timestamp_before.call_args
        actual_ts = call_args[0][0]
        expected_ts = now - 3 * 24 * 3600
        assert abs(actual_ts - expected_ts) <= 2

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_saves_only_media_items(self, mock_repo):
        seg = MagicMock()
        item_with_media = MagicMock()
        item_with_media.content.deep_get = MagicMock(return_value=[seg])

        item_no_media = MagicMock()
        item_no_media.content.deep_get = MagicMock(return_value=[])

        mock_repo.get_by_timestamp_before = AsyncMock(
            return_value=[item_with_media, item_no_media]
        )
        mock_repo.save_many = AsyncMock(return_value=1)

        await _unbound(agent_id="a1", expiry_days=7)
        mock_repo.save_many.assert_called_once_with([item_with_media])
