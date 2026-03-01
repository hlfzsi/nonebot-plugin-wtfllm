"""services/easy_ban.py handler 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.services.easy_ban import (
    _perm,
    easyban_cmd,
    handle_user_add,
    handle_user_remove,
    handle_user_list,
    handle_group_add,
    handle_group_remove,
    handle_group_list,
)


MODULE = "nonebot_plugin_wtfllm.services.easy_ban"


class FinishCalled(Exception):
    def __init__(self, msg):
        self.message = msg


@pytest.fixture(autouse=True)
def mock_finish(monkeypatch):
    async def _finish(msg):
        raise FinishCalled(msg)

    monkeypatch.setattr(easyban_cmd, "finish", _finish)


def _make_match(available, result=None):
    m = MagicMock()
    m.available = available
    m.result = result
    return m


def _make_session(user_id="u1", group_id=None):
    s = MagicMock()
    s.user.id = user_id
    if group_id:
        s.scene.is_private = False
        s.scene_path = group_id
        s.group.id = group_id
    else:
        s.scene.is_private = True
        s.scene_path = user_id
        s.group = None
    return s


# ===================== 权限测试 =====================


class TestPermission:
    @pytest.mark.asyncio
    async def test_admin_returns_true(self):
        session = _make_session(user_id="admin_1")
        with patch(f"{MODULE}.APP_CONFIG") as mock_config:
            mock_config.admin_users = ["admin_1"]
            result = await _perm(session)
        assert result is True

    @pytest.mark.asyncio
    async def test_non_admin_returns_false(self):
        session = _make_session(user_id="normal")
        with patch(f"{MODULE}.APP_CONFIG") as mock_config:
            mock_config.admin_users = ["admin_1"]
            result = await _perm(session)
        assert result is False


# ===================== user add =====================


class TestHandleUserAdd:
    @pytest.mark.asyncio
    async def test_no_user_id(self):
        user_id = _make_match(False)
        session = _make_session()
        with pytest.raises(FinishCalled) as exc:
            await handle_user_add(user_id, session)
        assert "提供" in exc.value.message

    @pytest.mark.asyncio
    async def test_self_ban(self):
        user_id = _make_match(True, "u1")
        session = _make_session(user_id="u1")
        with pytest.raises(FinishCalled) as exc:
            await handle_user_add(user_id, session)
        assert "自己" in exc.value.message

    @pytest.mark.asyncio
    async def test_admin_immunity(self):
        user_id = _make_match(True, "admin_1")
        session = _make_session(user_id="u1")
        with patch(f"{MODULE}.APP_CONFIG") as mock_config:
            mock_config.admin_users = ["admin_1"]
            with pytest.raises(FinishCalled) as exc:
                await handle_user_add(user_id, session)
        assert "管理员" in exc.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.add_banned_user", new_callable=AsyncMock)
    async def test_success(self, mock_add):
        user_id = _make_match(True, "target_user")
        session = _make_session(user_id="admin")
        with patch(f"{MODULE}.APP_CONFIG") as mock_config:
            mock_config.admin_users = ["admin"]
            with pytest.raises(FinishCalled) as exc:
                await handle_user_add(user_id, session)
        mock_add.assert_called_once_with("target_user")
        assert "target_user" in exc.value.message
        assert "限制列表" in exc.value.message


# ===================== user remove =====================


class TestHandleUserRemove:
    @pytest.mark.asyncio
    async def test_no_user_id(self):
        user_id = _make_match(False)
        with pytest.raises(FinishCalled) as exc:
            await handle_user_remove(user_id)
        assert "提供" in exc.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.remove_banned_user", new_callable=AsyncMock)
    async def test_success(self, mock_remove):
        user_id = _make_match(True, "u2")
        with pytest.raises(FinishCalled) as exc:
            await handle_user_remove(user_id)
        mock_remove.assert_called_once_with("u2")
        assert "解除限制" in exc.value.message


# ===================== user list =====================


class TestHandleUserList:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.get_banned_users", new_callable=AsyncMock, return_value=set())
    async def test_empty(self, mock_get):
        with pytest.raises(FinishCalled) as exc:
            await handle_user_list()
        assert "没有" in exc.value.message

    @pytest.mark.asyncio
    @patch(
        f"{MODULE}.get_banned_users",
        new_callable=AsyncMock,
        return_value={"u1", "u2"},
    )
    async def test_non_empty(self, mock_get):
        with pytest.raises(FinishCalled) as exc:
            await handle_user_list()
        assert "u1" in exc.value.message
        assert "u2" in exc.value.message


# ===================== group add =====================


class TestHandleGroupAdd:
    @pytest.mark.asyncio
    async def test_no_group(self):
        group_id = _make_match(False)
        session = _make_session()
        with pytest.raises(FinishCalled) as exc:
            await handle_group_add(group_id, session)
        assert "提供" in exc.value.message or "群组" in exc.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.add_banned_group", new_callable=AsyncMock)
    async def test_from_session(self, mock_add):
        group_id = _make_match(False)
        session = _make_session(group_id="g_session")
        with pytest.raises(FinishCalled) as exc:
            await handle_group_add(group_id, session)
        mock_add.assert_called_once_with("g_session")
        assert "g_session" in exc.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.add_banned_group", new_callable=AsyncMock)
    async def test_explicit(self, mock_add):
        group_id = _make_match(True, "g_explicit")
        session = _make_session()
        with pytest.raises(FinishCalled) as exc:
            await handle_group_add(group_id, session)
        mock_add.assert_called_once_with("g_explicit")


# ===================== group remove =====================


class TestHandleGroupRemove:
    @pytest.mark.asyncio
    async def test_no_group(self):
        group_id = _make_match(False)
        session = _make_session()
        with pytest.raises(FinishCalled) as exc:
            await handle_group_remove(group_id, session)
        assert "提供" in exc.value.message or "群组" in exc.value.message

    @pytest.mark.asyncio
    @patch(f"{MODULE}.remove_banned_group", new_callable=AsyncMock)
    async def test_success(self, mock_remove):
        group_id = _make_match(True, "g1")
        session = _make_session()
        with pytest.raises(FinishCalled) as exc:
            await handle_group_remove(group_id, session)
        mock_remove.assert_called_once_with("g1")
        assert "解除限制" in exc.value.message


# ===================== group list =====================


class TestHandleGroupList:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.get_banned_groups", new_callable=AsyncMock, return_value=set())
    async def test_empty(self, mock_get):
        with pytest.raises(FinishCalled) as exc:
            await handle_group_list()
        assert "没有" in exc.value.message

    @pytest.mark.asyncio
    @patch(
        f"{MODULE}.get_banned_groups",
        new_callable=AsyncMock,
        return_value={"g1", "g2"},
    )
    async def test_non_empty(self, mock_get):
        with pytest.raises(FinishCalled) as exc:
            await handle_group_list()
        assert "g1" in exc.value.message
        assert "g2" in exc.value.message
