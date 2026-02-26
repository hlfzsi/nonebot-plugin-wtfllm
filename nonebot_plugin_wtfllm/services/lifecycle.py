__all__ = ["setup_lifecycle_tasks", "shutdown_lifecycle_tasks"]

import asyncio

from nonebot import get_bots

from .delete_media import _unbound
from ..utils import APP_CONFIG, get_agent_id_from_bot, logger

_tasks: list[asyncio.Task] = []


def setup_lifecycle_tasks():
    if APP_CONFIG.media_auto_unbind:
        _tasks.append(asyncio.create_task(auto_unbind_expired_media()))


def shutdown_lifecycle_tasks():
    for task in _tasks:
        task.cancel()


async def auto_unbind_expired_media():
    # 低精度任务，不用aps了
    while True:
        await asyncio.sleep(24 * 3600)  # 每24小时执行一次
        try:
            bots = [get_agent_id_from_bot(bot) for bot in get_bots().values()]
            _tasks = [
                _unbound(agent_id=agent_id, expiry_days=APP_CONFIG.media_lifecycle_days)
                for agent_id in bots
            ]
            count = sum(await asyncio.gather(*_tasks))
            logger.info(f"自动清理了{count}条过期媒体记录")
        except asyncio.CancelledError:
            logger.info("自动清理过期媒体任务已取消")
            break
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"自动清理过期媒体时发生错误: {e} ")
