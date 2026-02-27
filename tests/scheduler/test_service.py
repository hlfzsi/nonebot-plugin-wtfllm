"""scheduler/ 模块单元测试

覆盖: registry.py, triggers.py, service.py, executor.py, recovery.py, tasks/send_message.py
"""

import time

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import BaseModel

from nonebot_plugin_wtfllm.scheduler.registry import (
    scheduled_task,
    get_task_handler,
    get_task_params_model,
    list_registered_tasks,
    _TASK_REGISTRY,
)
from nonebot_plugin_wtfllm.scheduler.triggers import (
    DateTriggerConfig,
    IntervalTriggerConfig,
    CronTriggerConfig,
)
from nonebot_plugin_wtfllm.scheduler.service import (
    schedule_job,
    cancel_job,
    _generate_job_id,
)
from nonebot_plugin_wtfllm.scheduler.executor import execute_scheduled_job
from nonebot_plugin_wtfllm.scheduler.recovery import recover_pending_jobs

MODULE_SVC = "nonebot_plugin_wtfllm.scheduler.service"
MODULE_EXE = "nonebot_plugin_wtfllm.scheduler.executor"
MODULE_REC = "nonebot_plugin_wtfllm.scheduler.recovery"
MODULE_TASK = "nonebot_plugin_wtfllm.scheduler.tasks.send_static_message"


# ===================== _generate_job_id =====================


class TestGenerateJobId:
    def test_starts_with_sched(self):
        result = _generate_job_id()
        assert result.startswith("sched_")

    def test_unique(self):
        ids = {_generate_job_id() for _ in range(100)}
        assert len(ids) == 100


# ===================== registry =====================


class TestRegistry:
    def setup_method(self):
        """每个测试前清空注册表，避免测试间互相干扰"""
        self._backup = dict(_TASK_REGISTRY)
        _TASK_REGISTRY.clear()

    def teardown_method(self):
        """恢复注册表"""
        _TASK_REGISTRY.clear()
        _TASK_REGISTRY.update(self._backup)

    def test_register_and_lookup(self):
        class TestParams(BaseModel):
            x: int

        @scheduled_task("test_task", TestParams)
        async def handler(params: TestParams) -> None:
            pass

        assert get_task_handler("test_task") is handler
        assert get_task_params_model("test_task") is TestParams
        assert "test_task" in list_registered_tasks()

    def test_duplicate_registration_raises(self):
        class P(BaseModel):
            pass

        @scheduled_task("dup_task", P)
        async def handler1(params): ...

        with pytest.raises(ValueError, match="already registered"):

            @scheduled_task("dup_task", P)
            async def handler2(params): ...

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError, match="Unknown task type"):
            get_task_handler("nonexistent")

        with pytest.raises(KeyError, match="Unknown task type"):
            get_task_params_model("nonexistent")


# ===================== triggers =====================


class TestTriggerConfigs:
    def test_date_trigger_config(self):
        trigger = DateTriggerConfig(run_timestamp=1700000000)
        assert trigger.type == "date"
        assert trigger.run_timestamp == 1700000000
        assert trigger.run_date.timestamp() == 1700000000

        dumped = trigger.model_dump()
        assert dumped["type"] == "date"
        assert dumped["run_timestamp"] == 1700000000

    def test_interval_trigger_config(self):
        trigger = IntervalTriggerConfig(hours=2, minutes=30)
        assert trigger.type == "interval"
        assert trigger.hours == 2
        assert trigger.minutes == 30

    def test_cron_trigger_config(self):
        trigger = CronTriggerConfig(minute="0", hour="*/2")
        assert trigger.type == "cron"
        assert trigger.minute == "0"
        assert trigger.hour == "*/2"

    def test_trigger_serialization_roundtrip(self):
        trigger = DateTriggerConfig(run_timestamp=1700000000)
        dumped = trigger.model_dump()
        restored = DateTriggerConfig.model_validate(dumped)
        assert restored.run_timestamp == trigger.run_timestamp


# ===================== schedule_job (service) =====================


class TestScheduleJob:
    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_job_repo")
    @patch(f"{MODULE_SVC}.get_task_handler")
    async def test_schedule_creates_and_adds_job(
        self, mock_get_handler, mock_repo, mock_scheduler
    ):
        mock_get_handler.return_value = MagicMock()
        mock_record = MagicMock()
        mock_record.job_id = "sched_test123"
        mock_repo.save = AsyncMock(return_value=mock_record)

        params = MagicMock(spec=BaseModel)
        params.model_dump.return_value = {"key": "value"}
        trigger = DateTriggerConfig(run_timestamp=int(time.time()) + 3600)

        result = await schedule_job(
            task_name="send_static_message",
            task_params=params,
            trigger=trigger,
            user_id="u1",
            agent_id="a1",
        )
        assert result is mock_record
        mock_repo.save.assert_called_once()
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.get_task_handler", side_effect=KeyError("Unknown"))
    async def test_schedule_unknown_task_raises(self, mock_get):
        params = MagicMock(spec=BaseModel)
        params.model_dump.return_value = {}
        trigger = DateTriggerConfig(run_timestamp=int(time.time()) + 3600)

        with pytest.raises(KeyError):
            await schedule_job(
                task_name="nonexistent",
                task_params=params,
                trigger=trigger,
            )


# ===================== cancel_job (service) =====================


class TestCancelJob:
    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_job_repo")
    async def test_cancel_existing_job(self, mock_repo, mock_scheduler):
        mock_scheduler.get_job.return_value = MagicMock()
        mock_repo.mark_canceled = AsyncMock(return_value=MagicMock())
        result = await cancel_job("job_1")
        assert result is True
        mock_scheduler.remove_job.assert_called_once_with("job_1")
        mock_repo.mark_canceled.assert_called_once_with("job_1")

    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_job_repo")
    async def test_cancel_no_scheduler_job(self, mock_repo, mock_scheduler):
        mock_scheduler.get_job.return_value = None
        mock_repo.mark_canceled = AsyncMock(return_value=MagicMock())
        result = await cancel_job("job_2")
        assert result is True
        mock_scheduler.remove_job.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_SVC}.scheduler")
    @patch(f"{MODULE_SVC}.scheduled_job_repo")
    async def test_cancel_record_not_found(self, mock_repo, mock_scheduler):
        mock_scheduler.get_job.return_value = None
        mock_repo.mark_canceled = AsyncMock(return_value=None)
        result = await cancel_job("ghost")
        assert result is False


# ===================== execute_scheduled_job =====================


class TestExecuteScheduledJob:
    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.get_task_params_model")
    @patch(f"{MODULE_EXE}.get_task_handler")
    @patch(f"{MODULE_EXE}.scheduled_job_repo")
    async def test_success_flow(self, mock_repo, mock_get_handler, mock_get_model):
        record = MagicMock()
        record.job_id = "j1"
        record.task_name = "send_static_message"
        record.task_params = {"user_id": "u1", "agent_id": "a1"}
        record.status = "pending"
        mock_repo.get_by_job_id = AsyncMock(return_value=record)
        mock_repo.mark_completed = AsyncMock()

        mock_handler = AsyncMock()
        mock_get_handler.return_value = mock_handler

        mock_params_cls = MagicMock()
        mock_params_instance = MagicMock()
        mock_params_cls.model_validate.return_value = mock_params_instance
        mock_get_model.return_value = mock_params_cls

        await execute_scheduled_job("j1")

        mock_handler.assert_called_once_with(mock_params_instance)
        mock_repo.mark_completed.assert_called_once_with("j1")

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.scheduled_job_repo")
    async def test_record_not_found(self, mock_repo):
        mock_repo.get_by_job_id = AsyncMock(return_value=None)
        # Should not raise
        await execute_scheduled_job("ghost")

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.scheduled_job_repo")
    async def test_skip_non_pending(self, mock_repo):
        record = MagicMock()
        record.status = "completed"
        record.job_id = "j2"
        mock_repo.get_by_job_id = AsyncMock(return_value=record)
        mock_repo.mark_completed = AsyncMock()
        await execute_scheduled_job("j2")
        mock_repo.mark_completed.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_EXE}.get_task_params_model")
    @patch(f"{MODULE_EXE}.get_task_handler")
    @patch(f"{MODULE_EXE}.scheduled_job_repo")
    async def test_handler_failure_marks_failed(
        self, mock_repo, mock_get_handler, mock_get_model
    ):
        record = MagicMock()
        record.job_id = "j_fail"
        record.task_name = "send_static_message"
        record.task_params = {}
        record.status = "pending"
        mock_repo.get_by_job_id = AsyncMock(return_value=record)
        mock_repo.mark_failed = AsyncMock()

        mock_handler = AsyncMock(side_effect=RuntimeError("boom"))
        mock_get_handler.return_value = mock_handler

        mock_params_cls = MagicMock()
        mock_params_cls.model_validate.return_value = MagicMock()
        mock_get_model.return_value = mock_params_cls

        await execute_scheduled_job("j_fail")

        mock_repo.mark_failed.assert_called_once()
        args = mock_repo.mark_failed.call_args
        assert args[0][0] == "j_fail"
        assert "boom" in args[0][1]


# ===================== recover_pending_jobs =====================


class TestRecoverPendingJobs:
    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_job_repo")
    async def test_recover_future_date_jobs(self, mock_repo, mock_scheduler):
        now = int(time.time())
        mock_repo.batch_mark_missed_date_jobs = AsyncMock(return_value=2)

        pending_record = MagicMock()
        pending_record.trigger_config = {
            "type": "date",
            "run_timestamp": now + 3600,
        }
        pending_record.job_id = "future_job"
        pending_record.task_name = "send_static_message"

        async def _iter_batched():
            yield [pending_record]

        mock_repo.iter_pending_batched = _iter_batched

        await recover_pending_jobs()

        mock_repo.batch_mark_missed_date_jobs.assert_called_once()
        mock_scheduler.add_job.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_job_repo")
    async def test_skip_past_date_jobs(self, mock_repo, mock_scheduler):
        now = int(time.time())
        mock_repo.batch_mark_missed_date_jobs = AsyncMock(return_value=0)

        past_record = MagicMock()
        past_record.trigger_config = {
            "type": "date",
            "run_timestamp": now - 100,
        }
        past_record.job_id = "past_job"

        async def _iter_batched():
            yield [past_record]

        mock_repo.iter_pending_batched = _iter_batched

        await recover_pending_jobs()

        mock_scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_job_repo")
    async def test_no_pending_jobs(self, mock_repo, mock_scheduler):
        mock_repo.batch_mark_missed_date_jobs = AsyncMock(return_value=0)

        async def _iter_batched():
            return
            yield  # noqa: unreachable – makes this an async generator

        mock_repo.iter_pending_batched = _iter_batched

        await recover_pending_jobs()

        mock_scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{MODULE_REC}.scheduler")
    @patch(f"{MODULE_REC}.scheduled_job_repo")
    async def test_recover_interval_jobs(self, mock_repo, mock_scheduler):
        mock_repo.batch_mark_missed_date_jobs = AsyncMock(return_value=0)

        interval_record = MagicMock()
        interval_record.trigger_config = {
            "type": "interval",
            "hours": 2,
            "minutes": 0,
            "seconds": 0,
            "days": 0,
        }
        interval_record.job_id = "interval_job"
        interval_record.task_name = "some_task"

        async def _iter_batched():
            yield [interval_record]

        mock_repo.iter_pending_batched = _iter_batched

        await recover_pending_jobs()

        mock_scheduler.add_job.assert_called_once()
        call_kwargs = mock_scheduler.add_job.call_args
        assert (
            call_kwargs.kwargs.get("trigger") == "interval"
            or call_kwargs[1].get("trigger") == "interval"
        )


# ===================== tasks/send_message.py =====================


# 延迟导入后, patch 路径指向各函数/类的源模块
_SRC_BAN = "nonebot_plugin_wtfllm.services.func.easy_ban.is_banned"
_SRC_ENSURE = "nonebot_plugin_wtfllm.utils.ensure_msgid_from_receipt"
_SRC_STORE_WITH_CONTEXT = (
    "nonebot_plugin_wtfllm.stream_processing.store_message_with_context"
)
_SRC_TARGET = "nonebot_plugin_alconna.Target"
_SRC_UNIMSG = "nonebot_plugin_alconna.UniMessage"


class TestSendStaticMessageHandler:
    @pytest.mark.asyncio
    @patch(_SRC_STORE_WITH_CONTEXT, new_callable=AsyncMock)
    @patch(_SRC_ENSURE, return_value="sent_123")
    @patch(_SRC_BAN, new_callable=AsyncMock, return_value=False)
    async def test_success_flow(
        self, mock_banned, mock_ensure, mock_store
    ):
        from nonebot_plugin_wtfllm.scheduler.tasks.send_static_message import (
            handle_send_static_message,
            SendStaticMessageParams,
        )

        with (
            patch(_SRC_TARGET) as mock_target_cls,
            patch(_SRC_UNIMSG) as mock_unimsg_cls,
        ):
            mock_target = MagicMock()
            mock_target_cls.load.return_value = mock_target
            mock_unimsg = MagicMock()
            mock_receipt = MagicMock()
            mock_unimsg.send = AsyncMock(return_value=mock_receipt)
            mock_unimsg_cls.load.return_value = mock_unimsg

            params = SendStaticMessageParams(
                target_data={"some": "data"},
                messages=[{"type": "text", "data": {"text": "hi"}}],
                user_id="u1",
                group_id=None,
                agent_id="a1",
            )
            await handle_send_static_message(params)

        mock_unimsg.send.assert_called_once_with(target=mock_target)
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    @patch(_SRC_BAN, new_callable=AsyncMock, return_value=True)
    async def test_banned_user_raises(self, mock_banned):
        from nonebot_plugin_wtfllm.scheduler.tasks.send_static_message import (
            handle_send_static_message,
            SendStaticMessageParams,
        )

        params = SendStaticMessageParams(
            target_data={},
            messages=[],
            user_id="banned_user",
            group_id="g1",
            agent_id="a1",
        )

        with pytest.raises(RuntimeError, match="banned"):
            await handle_send_static_message(params)

    @pytest.mark.asyncio
    @patch(_SRC_BAN, new_callable=AsyncMock, return_value=False)
    async def test_send_failure_propagates(self, mock_banned):
        from nonebot_plugin_wtfllm.scheduler.tasks.send_static_message import (
            handle_send_static_message,
            SendStaticMessageParams,
        )

        with (
            patch(_SRC_TARGET) as mock_target_cls,
            patch(_SRC_UNIMSG) as mock_unimsg_cls,
        ):
            mock_target_cls.load.return_value = MagicMock()
            mock_unimsg = MagicMock()
            mock_unimsg.send = AsyncMock(side_effect=RuntimeError("send failed"))
            mock_unimsg_cls.load.return_value = mock_unimsg

            params = SendStaticMessageParams(
                target_data={},
                messages=[],
                user_id="u1",
                group_id=None,
                agent_id="a1",
            )

            with pytest.raises(RuntimeError, match="send failed"):
                await handle_send_static_message(params)
