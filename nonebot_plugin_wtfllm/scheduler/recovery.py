__all__ = ["recover_pending_jobs"]

import time

from .engine import scheduler
from .executor import execute_scheduled_job
from .triggers import DateTriggerConfig, IntervalTriggerConfig, CronTriggerConfig
from ..db import scheduled_job_repo
from ..utils import logger


async def recover_pending_jobs() -> None:
    """恢复数据库中的 pending 任务到 APScheduler。"""
    now = int(time.time())

    missed_count = await scheduled_job_repo.batch_mark_missed_date_jobs(cutoff=now)
    if missed_count:
        logger.info(f"Marked {missed_count} missed scheduled jobs")

    restored = 0

    async for batch in scheduled_job_repo.iter_pending_batched():
        for record in batch:
            trigger_cfg = record.trigger_config
            trigger_type = trigger_cfg.get("type")

            try:
                if trigger_type == "date":
                    run_ts = trigger_cfg.get("run_timestamp", 0)
                    if run_ts <= now:
                        continue
                    trigger = DateTriggerConfig.model_validate(trigger_cfg)
                    scheduler.add_job(
                        execute_scheduled_job,
                        trigger="date",
                        run_date=trigger.run_date,
                        id=record.job_id,
                        args=[record.job_id],
                        replace_existing=True,
                        misfire_grace_time=60 * 5,
                    )
                    restored += 1

                elif trigger_type == "interval":
                    trigger = IntervalTriggerConfig.model_validate(trigger_cfg)
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
                        id=record.job_id,
                        args=[record.job_id],
                        replace_existing=True,
                        misfire_grace_time=60 * 5,
                        **kwargs,
                    )
                    restored += 1

                elif trigger_type == "cron":
                    trigger = CronTriggerConfig.model_validate(trigger_cfg)
                    scheduler.add_job(
                        execute_scheduled_job,
                        trigger="cron",
                        minute=trigger.minute,
                        hour=trigger.hour,
                        day=trigger.day,
                        month=trigger.month,
                        day_of_week=trigger.day_of_week,
                        id=record.job_id,
                        args=[record.job_id],
                        replace_existing=True,
                        misfire_grace_time=60 * 5,
                    )
                    restored += 1

                else:
                    logger.warning(
                        f"Unknown trigger type '{trigger_type}' for job_id={record.job_id}, skipping"
                    )

            except Exception:
                logger.exception(
                    f"Failed to recover job_id={record.job_id}, task={record.task_name}"
                )

    if restored:
        logger.info(f"Restored {restored} pending jobs to scheduler")
