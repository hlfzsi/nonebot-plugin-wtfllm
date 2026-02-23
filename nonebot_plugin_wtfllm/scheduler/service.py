"""定时消息调度服务

提供创建和取消定时消息的高层接口，同时操作 APScheduler 与数据库。
"""

__all__ = ["schedule_message", "cancel_message"]

import time
import uuid
from datetime import datetime, timezone

import pybase64
from nonebot_plugin_uninfo import Uninfo
from nonebot_plugin_alconna import UniMessage, Target

from .engine import scheduler
from .executor import get_handle_func_by_type
from ..db import scheduled_message_repo
from ..db.models import ScheduledMessage
from ..db.models.scheduled_message import ScheduledFunctionType
from ..utils import logger


def short_uuid():
    uuid_bytes = uuid.uuid4().bytes
    encoded = pybase64.urlsafe_b64encode(uuid_bytes).decode().rstrip("=")
    return f"sched_{encoded[:11]}"


async def schedule_message(
    target: Target,
    session: Uninfo,
    unimsg: UniMessage,
    trigger_time: int,
    func_type: ScheduledFunctionType = ScheduledFunctionType.STATIC_MESSAGE,
) -> ScheduledMessage:
    """创建定时消息：同时写入数据库并注册 APScheduler 作业

    Args:
        target: 消息发送目标
        session: 当前会话信息
        unimsg: 待发送的消息内容
        trigger_time: 触发时间（Unix 时间戳）

    Returns:
        持久化后的 ScheduledMessage 记录
    """
    job_id = short_uuid()

    record = ScheduledMessage.create(
        job_id=job_id,
        target=target,
        session=session,
        unimsg=unimsg,
        trigger_time=trigger_time,
        func_type=func_type,
        created_at=int(time.time()),
    )
    record = await scheduled_message_repo.save(record)

    handle_func = get_handle_func_by_type(func_type)

    run_date = datetime.fromtimestamp(trigger_time, tz=timezone.utc)
    scheduler.add_job(
        handle_func,
        trigger="date",
        run_date=run_date,
        id=job_id,
        args=[job_id],
        replace_existing=True,
        misfire_grace_time=60 * 5,
    )

    logger.info(
        f"Scheduled message created: job_id={job_id}, trigger_time={trigger_time}"
    )
    return record


async def cancel_message(job_id: str) -> bool:
    """取消定时消息：同时从 APScheduler 移除并更新数据库状态

    Args:
        job_id: APScheduler 作业 ID

    Returns:
        是否成功取消
    """
    existing_job = scheduler.get_job(job_id)
    if existing_job:
        scheduler.remove_job(job_id)

    record = await scheduled_message_repo.mark_canceled(job_id)
    if record:
        logger.info(f"Scheduled message canceled: job_id={job_id}")
        return True

    logger.warning(f"Cancel failed, record not found: job_id={job_id}")
    return False
