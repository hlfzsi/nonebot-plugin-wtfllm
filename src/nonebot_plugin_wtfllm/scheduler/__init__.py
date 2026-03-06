"""通用调度模块

提供 APScheduler 集成，管理调度任务的注册、执行和生命周期。
"""

__all__ = [
    "scheduler",
    "init_scheduler",
    "shutdown_scheduler",
    "schedule_job",
    "cancel_job",
    "scheduled_task",
]

from .engine import scheduler, init_scheduler, shutdown_scheduler
from .service import schedule_job, cancel_job
from .registry import scheduled_task
