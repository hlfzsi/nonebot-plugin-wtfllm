__all__ = ["execute_scheduled_job"]

from ..db import scheduled_job_repo
from ..db.models.scheduled_job import ScheduledJobStatus
from ..utils import logger
from .registry import get_task_handler, get_task_params_model


async def execute_scheduled_job(job_id: str) -> None:
    """APScheduler 通用作业回调：查找注册表、反序列化参数、调用 handler。

    Args:
        job_id: 对应 ScheduledJob 记录的 job_id
    """
    record = await scheduled_job_repo.get_by_job_id(job_id)
    if not record:
        logger.warning(f"Scheduled job not found: job_id={job_id}")
        return

    if record.status != ScheduledJobStatus.PENDING:
        logger.debug(
            f"Skipped non-pending job: job_id={job_id} (status={record.status})"
        )
        return

    try:
        handler = get_task_handler(record.task_name)
        params_model = get_task_params_model(record.task_name)
        params = params_model.model_validate(record.task_params)

        await handler(params)

        await scheduled_job_repo.mark_completed(job_id)
        logger.info(
            f"Scheduled job completed: job_id={job_id}, task={record.task_name}"
        )
    except Exception as e:
        await scheduled_job_repo.mark_failed(job_id, str(e))
        logger.error(
            f"Scheduled job failed: job_id={job_id}, task={record.task_name}, error={e}"
        )
