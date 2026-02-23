import asyncio

from ....deps import Context
from .....utils import logger


def reschedule_deadline(ctx: Context, added_time: float) -> None:
    """
    为当前对话管理器重新调度截止时间

    Args:
        ctx (Context): 上下文对象
        added_time (float): 需要增加的时间（秒）
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("No running event loop, cannot reschedule deadline.")
        return
    if ctx.deps.cm is not None:
        former_deadline = ctx.deps.cm.when() or 0
        new_deadline = former_deadline + added_time
        ctx.deps.cm.reschedule(new_deadline)
