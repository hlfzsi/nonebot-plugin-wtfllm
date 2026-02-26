"""APScheduler 引擎与生命周期管理"""

__all__ = ["scheduler", "init_scheduler", "shutdown_scheduler"]

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..utils import logger

scheduler = AsyncIOScheduler()


async def init_scheduler() -> None:
    """启动调度器，并恢复漏执行的任务"""
    from .recovery import recover_pending_jobs
    from .tasks import setup as setup_task_functions

    setup_task_functions()
    await recover_pending_jobs()
    scheduler.start()
    logger.info("Scheduler started")


async def shutdown_scheduler() -> None:
    """关闭调度器"""
    scheduler.shutdown(wait=True)
    logger.info("Scheduler shutdown")
