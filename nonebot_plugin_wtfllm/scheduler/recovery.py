"""调度器启动恢复

服务重启后，将数据库中未过期的 pending 消息重新注册到 APScheduler，
并将已过期的 pending 消息标记为 missed。
"""

__all__ = ["recover_pending_jobs"]

import time
from datetime import datetime, timezone

from .engine import scheduler
from .executor import get_handle_func_by_type
from ..db import scheduled_message_repo
from ..utils import logger


async def recover_pending_jobs() -> None:
    """恢复数据库中的 pending 任务到 APScheduler"""
    now = int(time.time())

    missed_count = await scheduled_message_repo.batch_mark_missed(cutoff=now)
    if missed_count:
        logger.info(f"Marked {missed_count} missed scheduled messages")

    pending = await scheduled_message_repo.list_pending()
    restored = 0
    for record in pending:
        if record.trigger_time <= now:
            continue

        run_date = datetime.fromtimestamp(record.trigger_time, tz=timezone.utc)
        scheduler.add_job(
            get_handle_func_by_type(record.func_type),
            trigger="date",
            run_date=run_date,
            id=record.job_id,
            args=[record.job_id],
            replace_existing=True,
            misfire_grace_time=60 * 5,
        )
        restored += 1

    if restored:
        logger.info(f"Restored {restored} pending scheduled messages to scheduler")
