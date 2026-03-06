__all__ = ["schedule_job", "cancel_job"]

import uuid

import pybase64
from pydantic import BaseModel

from .engine import scheduler
from .executor import execute_scheduled_job
from .registry import get_task_handler
from .triggers import TriggerConfig, DateTriggerConfig, IntervalTriggerConfig, CronTriggerConfig
from ..db import scheduled_job_repo
from ..db.models.scheduled_job import ScheduledJob
from ..utils import logger


def _generate_job_id() -> str:
    uuid_bytes = uuid.uuid4().bytes
    encoded = pybase64.urlsafe_b64encode(uuid_bytes).decode().rstrip("=")
    return f"sched_{encoded[:11]}"


def _register_apscheduler_job(job_id: str, trigger: TriggerConfig) -> None:
    """根据 trigger 类型向 APScheduler 注册作业。"""
    if isinstance(trigger, DateTriggerConfig):
        scheduler.add_job(
            execute_scheduled_job,
            trigger="date",
            run_date=trigger.run_date,
            id=job_id,
            args=[job_id],
            replace_existing=True,
            misfire_grace_time=60 * 5,
        )
    elif isinstance(trigger, IntervalTriggerConfig):
        kwargs = {}
        if trigger.seconds:
            kwargs["seconds"] = trigger.seconds
        if trigger.minutes:
            kwargs["minutes"] = trigger.minutes
        if trigger.hours:
            kwargs["hours"] = trigger.hours
        if trigger.days:
            kwargs["days"] = trigger.days
        scheduler.add_job(
            execute_scheduled_job,
            trigger="interval",
            id=job_id,
            args=[job_id],
            replace_existing=True,
            misfire_grace_time=60 * 5,
            **kwargs,
        )
    elif isinstance(trigger, CronTriggerConfig):
        scheduler.add_job(
            execute_scheduled_job,
            trigger="cron",
            minute=trigger.minute,
            hour=trigger.hour,
            day=trigger.day,
            month=trigger.month,
            day_of_week=trigger.day_of_week,
            id=job_id,
            args=[job_id],
            replace_existing=True,
            misfire_grace_time=60 * 5,
        )
    else:
        raise ValueError(f"Unsupported trigger type: {type(trigger)}")


async def schedule_job(
    task_name: str,
    task_params: BaseModel,
    trigger: TriggerConfig,
    *,
    user_id: str | None = None,
    group_id: str | None = None,
    agent_id: str | None = None,
    description: str | None = None,
) -> ScheduledJob:
    """创建调度任务：写入数据库并注册 APScheduler 作业。

    Args:
        task_name: 注册表中的任务类型名称（必须已通过 @scheduled_task 注册）
        task_params: 任务参数 (Pydantic BaseModel)
        trigger: 触发器配置
        user_id: 可选，所属用户
        group_id: 可选，关联群组
        agent_id: 可选，Bot/Agent
        description: 可选，人类可读描述

    Returns:
        持久化后的 ScheduledJob 记录
    """
    get_task_handler(task_name)

    job_id = _generate_job_id()

    record = ScheduledJob(
        job_id=job_id,
        task_name=task_name,
        task_params=task_params.model_dump(),
        trigger_config=trigger.model_dump(),
        user_id=user_id,
        group_id=group_id,
        agent_id=agent_id,
        description=description,
    )
    record = await scheduled_job_repo.save(record)

    _register_apscheduler_job(job_id, trigger)

    logger.info(f"Scheduled job created: job_id={job_id}, task={task_name}")
    return record


async def cancel_job(job_id: str) -> bool:
    """取消调度任务：从 APScheduler 移除并更新数据库状态。

    Args:
        job_id: 作业 ID

    Returns:
        是否成功取消
    """
    existing_job = scheduler.get_job(job_id)
    if existing_job:
        scheduler.remove_job(job_id)

    record = await scheduled_job_repo.mark_canceled(job_id)
    if record:
        logger.info(f"Scheduled job canceled: job_id={job_id}")
        return True

    logger.warning(f"Cancel failed, record not found: job_id={job_id}")
    return False
