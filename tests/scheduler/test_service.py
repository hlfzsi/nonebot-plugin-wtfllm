"""scheduler/ 模块单元测试

覆盖: service.py, executor.py, recovery.py
"""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.scheduler.service import (
    schedule_message,
    cancel_message,
    short_uuid,
)
from nonebot_plugin_wtfllm.scheduler.executor import (
    get_handle_func_by_type,
    should_skip_scheduled_message,
    execute_scheduled_static_message,
)
from nonebot_plugin_wtfllm.scheduler.recovery import recover_pending_jobs
from nonebot_plugin_wtfllm.db.models.scheduled_message import ScheduledFunctionType


MODULE_SVC = "nonebot_plugin_wtfllm.scheduler.service"
MODULE_EXE = "nonebot_plugin_wtfllm.scheduler.executor"
MODULE_REC = "nonebot_plugin_wtfllm.scheduler.recovery"


# ===================== short_uuid =====================


class TestShortUuid:
    def test_starts_with_sched(self):
        result = short_uuid()
        assert result.startswith("sched_")

    def test_unique(self):
        ids = {short_uuid() for _ in range(100)}
        assert len(ids) == 100


# ===================== get_handle_func_by_type =====================


class TestGetHandleFuncByType:
    def test_static_message(self):
        func = get_handle_func_by_type(ScheduledFunctionType.STATIC_MESSAGE)
        assert func is execute_scheduled_static_message

    def test_dynamic_message(self):
        from nonebot_plugin_wtfllm.scheduler.executor import (
            execute_scheduled_dynamic_message,
        )
        func = get_handle_func_by_type(ScheduledFunctionType.DYNAMIC_MESSAGE)
        assert func is execute_scheduled_dynamic_message

    def test_unknown_type(self):
        func = get_handle_func_by_type("unknown_type")
        assert func is None


# ===================== should_skip_scheduled_message =====================


class TestShouldSkip:
    @pytest.mark.asyncio
    async def test_skip_non_pending(self):
        record = MagicMock()
        record.status = "completed"
        record.job_id = "j1"
        assert await should_skip_scheduled_message(record) is True

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.is_banned", new_callable=AsyncMock, return_value=True)
    @patch(f"{MODULE_EXE}.scheduled_message_repo")
    async def test_skip_banned(self, mock_repo, mock_banned):
        mock_repo.mark_failed = AsyncMock()
        record = MagicMock()
        record.status = "pending"
        record.user_id = "u1"
        record.group_id = "g1"
        record.job_id = "j2"
        assert await should_skip_scheduled_message(record) is True
        mock_repo.mark_failed.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.is_banned", new_callable=AsyncMock, return_value=False)
    async def test_not_skip(self, mock_banned):
        record = MagicMock()
        record.status = "pending"
        record.user_id = "u1"
        record.group_id = None
        assert await should_skip_scheduled_message(record) is False


# ===================== schedule_message (service) =====================


class TestScheduleMessage:
    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_message_repo")
    @patch(f"{MODULE_SVC}.get_handle_func_by_type")
    @patch(f"{MODULE_SVC}.ScheduledMessage")
    async def test_schedule_creates_and_adds_job(
        self, mock_model, mock_get_func, mock_repo, mock_scheduler
    ):
        mock_record = MagicMock()
        mock_record.job_id = "sched_test123"
        mock_model.create.return_value = mock_record
        mock_repo.save = AsyncMock(return_value=mock_record)
        mock_get_func.return_value = MagicMock()

        target = MagicMock()
        session = MagicMock()
        unimsg = MagicMock()

        result = await schedule_message(
            target=target,
            session=session,
            unimsg=unimsg,
            trigger_time=int(time.time()) + 3600,
        )
        assert result is mock_record
        mock_repo.save.assert_called_once()
        mock_scheduler.add_job.assert_called_once()


# ===================== cancel_message (service) =====================


class TestCancelMessage:
    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_message_repo")
    async def test_cancel_existing_job(self, mock_repo, mock_scheduler):
        mock_scheduler.get_job.return_value = MagicMock()
        mock_repo.mark_canceled = AsyncMock(return_value=MagicMock())
        result = await cancel_message("job_1")
        assert result is True
        mock_scheduler.remove_job.assert_called_once_with("job_1")
        mock_repo.mark_canceled.assert_called_once_with("job_1")

    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_message_repo")
    async def test_cancel_no_scheduler_job(self, mock_repo, mock_scheduler):
        mock_scheduler.get_job.return_value = None
        mock_repo.mark_canceled = AsyncMock(return_value=MagicMock())
        result = await cancel_message("job_2")
        assert result is True
        mock_scheduler.remove_job.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_message_repo")
    async def test_cancel_record_not_found(self, mock_repo, mock_scheduler):
        mock_scheduler.get_job.return_value = None
        mock_repo.mark_canceled = AsyncMock(return_value=None)
        result = await cancel_message("ghost")
        assert result is False


# ===================== execute_scheduled_static_message =====================


class TestExecuteStaticMessage:
    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.msg_tracker")
    @patch(f"{MODULE_EXE}.convert_and_store_item", new_callable=AsyncMock)
    @patch(f"{MODULE_EXE}.ensure_msgid_from_receipt", return_value="sent_123")
    @patch(f"{MODULE_EXE}.should_skip_scheduled_message", new_callable=AsyncMock, return_value=False)
    @patch(f"{MODULE_EXE}.scheduled_message_repo")
    async def test_success_flow(
        self, mock_repo, mock_skip, mock_ensure, mock_convert, mock_tracker
    ):
        record = MagicMock()
        record.job_id = "j1"
        record.target_data = {"some": "data"}
        record.messages = [{"type": "text", "data": {"text": "hi"}}]
        record.agent_id = "a1"
        record.user_id = "u1"
        record.group_id = None
        mock_repo.get_by_job_id = AsyncMock(return_value=record)
        mock_repo.mark_completed = AsyncMock()

        # Patch Target.load and UniMessage.load
        with patch(f"{MODULE_EXE}.Target") as mock_target_cls, \
             patch(f"{MODULE_EXE}.UniMessage") as mock_unimsg_cls:
            mock_target = MagicMock()
            mock_target_cls.load.return_value = mock_target
            mock_unimsg = MagicMock()
            mock_receipt = MagicMock()
            mock_unimsg.send = AsyncMock(return_value=mock_receipt)
            mock_unimsg_cls.load.return_value = mock_unimsg

            await execute_scheduled_static_message("j1")

        mock_repo.mark_completed.assert_called_once_with("j1")
        mock_convert.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.scheduled_message_repo")
    async def test_record_not_found(self, mock_repo):
        mock_repo.get_by_job_id = AsyncMock(return_value=None)
        # Should not raise
        await execute_scheduled_static_message("ghost")

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.should_skip_scheduled_message", new_callable=AsyncMock, return_value=True)
    @patch(f"{MODULE_EXE}.scheduled_message_repo")
    async def test_skip_message(self, mock_repo, mock_skip):
        record = MagicMock()
        mock_repo.get_by_job_id = AsyncMock(return_value=record)
        mock_repo.mark_completed = AsyncMock()
        await execute_scheduled_static_message("j_skip")
        mock_repo.mark_completed.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.should_skip_scheduled_message", new_callable=AsyncMock, return_value=False)
    @patch(f"{MODULE_EXE}.scheduled_message_repo")
    async def test_send_failure_marks_failed(self, mock_repo, mock_skip):
        record = MagicMock()
        record.job_id = "j_fail"
        record.target_data = {}
        record.messages = []
        record.agent_id = "a1"
        record.user_id = "u1"
        mock_repo.get_by_job_id = AsyncMock(return_value=record)
        mock_repo.mark_failed = AsyncMock()

        with patch(f"{MODULE_EXE}.Target") as mock_target_cls, \
             patch(f"{MODULE_EXE}.UniMessage") as mock_unimsg_cls:
            mock_unimsg = MagicMock()
            mock_unimsg.send = AsyncMock(side_effect=RuntimeError("send failed"))
            mock_unimsg_cls.load.return_value = mock_unimsg
            mock_target_cls.load.return_value = MagicMock()

            await execute_scheduled_static_message("j_fail")

        mock_repo.mark_failed.assert_called_once()


# ===================== recover_pending_jobs =====================


class TestRecoverPendingJobs:
    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_message_repo")
    @patch(f"{MODULE_REC}.get_handle_func_by_type")
    async def test_recover_future_jobs(self, mock_get_func, mock_repo, mock_scheduler):
        now = int(time.time())
        mock_repo.batch_mark_missed = AsyncMock(return_value=2)
        pending_record = MagicMock()
        pending_record.trigger_time = now + 3600
        pending_record.job_id = "future_job"
        pending_record.func_type = ScheduledFunctionType.STATIC_MESSAGE
        mock_repo.list_pending = AsyncMock(return_value=[pending_record])
        mock_get_func.return_value = MagicMock()

        await recover_pending_jobs()

        mock_repo.batch_mark_missed.assert_called_once()
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_message_repo")
    async def test_skip_past_pending(self, mock_repo, mock_scheduler):
        now = int(time.time())
        mock_repo.batch_mark_missed = AsyncMock(return_value=0)
        past_record = MagicMock()
        past_record.trigger_time = now - 100
        past_record.job_id = "past_job"
        mock_repo.list_pending = AsyncMock(return_value=[past_record])

        await recover_pending_jobs()

        mock_scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_message_repo")
    async def test_no_pending_jobs(self, mock_repo, mock_scheduler):
        mock_repo.batch_mark_missed = AsyncMock(return_value=0)
        mock_repo.list_pending = AsyncMock(return_value=[])

        await recover_pending_jobs()

        mock_scheduler.add_job.assert_not_called()
