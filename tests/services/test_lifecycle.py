"""services/lifecycle.py 单元测试"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import nonebot_plugin_wtfllm.services.lifecycle as lifecycle_mod


MODULE = "nonebot_plugin_wtfllm.services.lifecycle"


@pytest.fixture(autouse=True)
def reset_tasks(monkeypatch):
    """每个测试前重置模块级 _tasks"""
    monkeypatch.setattr(lifecycle_mod, "_tasks", [])


# ===================== setup / shutdown 测试 =====================


class TestSetupLifecycleTasks:
    @patch(f"{MODULE}.asyncio.create_task")
    @patch(f"{MODULE}.APP_CONFIG")
    def test_creates_task_when_enabled(self, mock_config, mock_create):
        mock_config.media_auto_unbind = True
        mock_task = MagicMock()
        mock_create.return_value = mock_task

        lifecycle_mod.setup_lifecycle_tasks()
        mock_create.assert_called_once()
        assert mock_task in lifecycle_mod._tasks
        # 关闭未 await 的协程，避免 RuntimeWarning
        mock_create.call_args[0][0].close()

    @patch(f"{MODULE}.asyncio.create_task")
    @patch(f"{MODULE}.APP_CONFIG")
    def test_no_task_when_disabled(self, mock_config, mock_create):
        mock_config.media_auto_unbind = False

        lifecycle_mod.setup_lifecycle_tasks()
        mock_create.assert_not_called()


class TestShutdownLifecycleTasks:
    def test_cancels_all_tasks(self, monkeypatch):
        t1 = MagicMock()
        t2 = MagicMock()
        monkeypatch.setattr(lifecycle_mod, "_tasks", [t1, t2])

        lifecycle_mod.shutdown_lifecycle_tasks()
        t1.cancel.assert_called_once()
        t2.cancel.assert_called_once()


# ===================== auto_unbind_expired_media 测试 =====================


class TestAutoUnbindExpiredMedia:
    @pytest.mark.asyncio
    @patch(f"{MODULE}._unbound", new_callable=AsyncMock, return_value=5)
    @patch(f"{MODULE}.get_agent_id_from_bot", return_value="agent1")
    @patch(f"{MODULE}.get_bots")
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    async def test_calls_unbound_for_each_bot(
        self, mock_sleep, mock_config, mock_get_bots, mock_get_agent, mock_unbound
    ):
        mock_config.media_lifecycle_days = 7
        mock_bot = MagicMock()
        mock_get_bots.return_value = {"bot1": mock_bot}

        # 让 sleep 第二次抛出 CancelledError 停止循环
        mock_sleep.side_effect = [None, asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            await lifecycle_mod.auto_unbind_expired_media()

        mock_unbound.assert_called_once_with(agent_id="agent1", expiry_days=7)

    @pytest.mark.asyncio
    @patch(f"{MODULE}._unbound", new_callable=AsyncMock)
    @patch(f"{MODULE}.get_agent_id_from_bot", side_effect=RuntimeError("bot err"))
    @patch(f"{MODULE}.get_bots")
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    async def test_handles_runtime_error(
        self, mock_sleep, mock_config, mock_get_bots, mock_get_agent, mock_unbound
    ):
        mock_config.media_lifecycle_days = 7
        mock_get_bots.return_value = {"bot1": MagicMock()}

        # 第一次 sleep 返回正常，RuntimeError 被捕获后继续循环
        # 第二次 sleep 抛 CancelledError 停止
        mock_sleep.side_effect = [None, asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            await lifecycle_mod.auto_unbind_expired_media()

        # unbound 不会被调用，因为 get_agent_id_from_bot 在列表推导中就抛异常了
        mock_unbound.assert_not_called()
