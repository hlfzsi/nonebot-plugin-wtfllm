"""通用调度任务 Repository

封装 ScheduledJob 的数据访问操作。
"""

__all__ = ["ScheduledJobRepository"]

import time
from collections.abc import AsyncIterator
from typing import List, Optional

from sqlmodel import select, desc, col

from .base import BaseRepository
from ..models.scheduled_job import ScheduledJob, ScheduledJobStatus
from ..engine import SESSION_MAKER, WRITE_LOCK


class ScheduledJobRepository(BaseRepository[ScheduledJob]):
    """通用调度任务数据访问层"""

    def __init__(self):
        super().__init__(ScheduledJob)

    async def get_by_job_id(self, job_id: str) -> Optional[ScheduledJob]:
        """根据 job_id 获取任务记录"""
        async with SESSION_MAKER() as session:
            stmt = select(ScheduledJob).where(ScheduledJob.job_id == job_id)
            return (await session.exec(stmt)).first()

    async def list_by_status(
        self,
        status: ScheduledJobStatus,
        limit: int = 100,
    ) -> List[ScheduledJob]:
        """按状态查询任务列表"""
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledJob)
                .where(ScheduledJob.status == status)
                .order_by(col(ScheduledJob.created_at))
                .limit(limit)
            )
            return list((await session.exec(stmt)).all())

    async def list_pending(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ScheduledJob]:
        """获取待执行的任务列表"""
        async with SESSION_MAKER() as session:
            stmt = select(ScheduledJob).where(
                ScheduledJob.status == ScheduledJobStatus.PENDING
            )
            if agent_id is not None:
                stmt = stmt.where(ScheduledJob.agent_id == agent_id)
            stmt = stmt.order_by(col(ScheduledJob.created_at)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def iter_pending_batched(
        self,
        batch_size: int = 200,
    ) -> AsyncIterator[list[ScheduledJob]]:
        """分批迭代 pending 任务（按 id 游标分页）

        每次从数据库取 batch_size 条记录，避免一次性加载全部到内存。

        Yields:
            每批 ScheduledJob 列表
        """
        last_id = 0
        while True:
            async with SESSION_MAKER() as session:
                stmt = (
                    select(ScheduledJob)
                    .where(
                        ScheduledJob.status == ScheduledJobStatus.PENDING,
                        ScheduledJob.id > last_id,  # type: ignore[operator]
                    )
                    .order_by(col(ScheduledJob.id))
                    .limit(batch_size)
                )
                batch = list((await session.exec(stmt)).all())
            if not batch:
                break
            last_id = batch[-1].id  # type: ignore[assignment]
            yield batch

    async def list_by_group(
        self,
        group_id: str,
        agent_id: str,
        status: Optional[ScheduledJobStatus] = None,
        limit: int = 50,
    ) -> List[ScheduledJob]:
        """按群组查询任务"""
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledJob)
                .where(ScheduledJob.group_id == group_id)
                .where(ScheduledJob.agent_id == agent_id)
            )
            if status is not None:
                stmt = stmt.where(ScheduledJob.status == status)
            stmt = stmt.order_by(desc(ScheduledJob.created_at)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def list_by_user(
        self,
        user_id: str,
        agent_id: str,
        status: Optional[ScheduledJobStatus] = None,
        limit: int = 50,
    ) -> List[ScheduledJob]:
        """按用户查询任务"""
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledJob)
                .where(ScheduledJob.user_id == user_id)
                .where(ScheduledJob.agent_id == agent_id)
            )
            if status is not None:
                stmt = stmt.where(ScheduledJob.status == status)
            stmt = stmt.order_by(desc(ScheduledJob.created_at)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def list_by_task_name(
        self,
        task_name: str,
        status: Optional[ScheduledJobStatus] = None,
        limit: int = 50,
    ) -> List[ScheduledJob]:
        """按任务类型查询"""
        async with SESSION_MAKER() as session:
            stmt = select(ScheduledJob).where(ScheduledJob.task_name == task_name)
            if status is not None:
                stmt = stmt.where(ScheduledJob.status == status)
            stmt = stmt.order_by(desc(ScheduledJob.created_at)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def mark_completed(self, job_id: str) -> Optional[ScheduledJob]:
        """标记为已完成"""
        return await self._update_status(job_id, ScheduledJobStatus.COMPLETED)

    async def mark_failed(
        self, job_id: str, error_message: str
    ) -> Optional[ScheduledJob]:
        """标记为失败并记录错误信息"""
        return await self._update_status(
            job_id, ScheduledJobStatus.FAILED, error_message=error_message
        )

    async def mark_missed(self, job_id: str) -> Optional[ScheduledJob]:
        """标记为错过"""
        return await self._update_status(job_id, ScheduledJobStatus.MISSED)

    async def mark_canceled(self, job_id: str) -> Optional[ScheduledJob]:
        """标记为已取消"""
        return await self._update_status(job_id, ScheduledJobStatus.CANCELED)

    async def batch_mark_missed_date_jobs(self, cutoff: Optional[int] = None) -> int:
        """批量标记过期的 date 类型 pending 任务为 missed

        仅处理 trigger_config.type == "date" 且 run_timestamp < cutoff 的记录。

        Args:
            cutoff: 截止时间戳，默认当前时间

        Returns:
            标记数量
        """
        if cutoff is None:
            cutoff = int(time.time())

        missed_ids: list[int] = []
        async for batch in self.iter_pending_batched():
            for record in batch:
                trigger_type = record.trigger_config.get("type")
                if trigger_type != "date":
                    continue
                run_ts = record.trigger_config.get("run_timestamp", 0)
                if run_ts < cutoff:
                    missed_ids.append(record.id)  # type: ignore[arg-type]

        if not missed_ids:
            return 0

        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                now = int(time.time())
                for record_id in missed_ids:
                    db_record = await session.get(ScheduledJob, record_id)
                    if db_record and db_record.status == ScheduledJobStatus.PENDING:
                        db_record.status = ScheduledJobStatus.MISSED  # type: ignore[assignment]
                        db_record.executed_at = now
                await session.commit()

        return len(missed_ids)

    async def delete_by_job_id(self, job_id: str) -> bool:
        """根据 job_id 删除任务记录"""
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    stmt = select(ScheduledJob).where(ScheduledJob.job_id == job_id)
                    record = (await session.exec(stmt)).first()
                    if record:
                        await session.delete(record)
                        return True
        return False

    async def cleanup_completed(self, before: int) -> int:
        """清理指定时间之前的已完成/已取消记录

        Args:
            before: 时间戳，清理此时间之前执行的记录

        Returns:
            删除数量
        """
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    stmt = select(ScheduledJob).where(
                        col(ScheduledJob.status).in_(
                            [ScheduledJobStatus.COMPLETED, ScheduledJobStatus.CANCELED]
                        ),
                        ScheduledJob.executed_at < before,  # type: ignore[operator]
                    )
                    records = (await session.exec(stmt)).all()
                    for record in records:
                        await session.delete(record)
                    return len(records)

    async def _update_status(
        self,
        job_id: str,
        status: ScheduledJobStatus,
        error_message: Optional[str] = None,
    ) -> Optional[ScheduledJob]:
        """统一状态更新逻辑"""
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                stmt = select(ScheduledJob).where(ScheduledJob.job_id == job_id)
                record = (await session.exec(stmt)).first()
                if not record:
                    return None

                record.status = status  # type: ignore[assignment]
                record.executed_at = int(time.time())
                if error_message is not None:
                    record.error_message = error_message

                await session.flush()
                await session.refresh(record)
                await session.commit()

        return record
