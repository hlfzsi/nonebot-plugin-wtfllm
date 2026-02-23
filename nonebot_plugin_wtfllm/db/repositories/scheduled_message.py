"""定时消息 Repository

封装定时消息的数据访问操作，与 APScheduler 集成。
"""

__all__ = ["ScheduledMessageRepository"]

import time
from typing import List, Literal, Optional

from sqlmodel import select, desc, col
from .base import BaseRepository
from ..models import ScheduledMessage
from ..engine import SESSION_MAKER, WRITE_LOCK


class ScheduledMessageRepository(BaseRepository[ScheduledMessage]):
    """定时消息数据访问层"""

    def __init__(self):
        super().__init__(ScheduledMessage)

    async def get_by_job_id(self, job_id: str) -> Optional[ScheduledMessage]:
        """根据 APScheduler job_id 获取定时消息"""
        async with SESSION_MAKER() as session:
            stmt = select(ScheduledMessage).where(ScheduledMessage.job_id == job_id)
            return (await session.exec(stmt)).first()

    async def list_by_status(
        self,
        status: Literal["pending", "completed", "failed", "missed", "canceled"],
        limit: int = 100,
    ) -> List[ScheduledMessage]:
        """按状态查询定时消息列表"""
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledMessage)
                .where(ScheduledMessage.status == status)
                .order_by(col(ScheduledMessage.trigger_time))
                .limit(limit)
            )
            return list((await session.exec(stmt)).all())

    async def list_pending(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ScheduledMessage]:
        """获取待执行的定时消息，按触发时间排序"""
        async with SESSION_MAKER() as session:
            stmt = select(ScheduledMessage).where(ScheduledMessage.status == "pending")
            if agent_id is not None:
                stmt = stmt.where(ScheduledMessage.agent_id == agent_id)
            stmt = stmt.order_by(col(ScheduledMessage.trigger_time)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def list_by_group(
        self,
        group_id: str,
        agent_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[ScheduledMessage]:
        """按群组查询定时消息"""
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledMessage)
                .where(ScheduledMessage.group_id == group_id)
                .where(ScheduledMessage.agent_id == agent_id)
            )
            if status is not None:
                stmt = stmt.where(ScheduledMessage.status == status)
            stmt = stmt.order_by(desc(ScheduledMessage.created_at)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def list_by_user(
        self,
        user_id: str,
        agent_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[ScheduledMessage]:
        """按用户查询定时消息"""
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledMessage)
                .where(ScheduledMessage.user_id == user_id)
                .where(ScheduledMessage.agent_id == agent_id)
            )
            if status is not None:
                stmt = stmt.where(ScheduledMessage.status == status)
            stmt = stmt.order_by(desc(ScheduledMessage.created_at)).limit(limit)
            return list((await session.exec(stmt)).all())

    async def list_missed(self, cutoff: Optional[int] = None) -> List[ScheduledMessage]:
        """获取过期未执行的 pending 消息

        Args:
            cutoff: 截止时间戳，默认当前时间
        """
        if cutoff is None:
            cutoff = int(time.time())
        async with SESSION_MAKER() as session:
            stmt = (
                select(ScheduledMessage)
                .where(ScheduledMessage.status == "pending")
                .where(col(ScheduledMessage.trigger_time) < cutoff)
                .order_by(col(ScheduledMessage.trigger_time))
            )
            return list((await session.exec(stmt)).all())

    async def mark_completed(self, job_id: str) -> Optional[ScheduledMessage]:
        """标记为已完成"""
        return await self._update_status(job_id, "completed")

    async def mark_failed(
        self, job_id: str, error_message: str
    ) -> Optional[ScheduledMessage]:
        """标记为失败并记录错误信息"""
        return await self._update_status(job_id, "failed", error_message=error_message)

    async def mark_missed(self, job_id: str) -> Optional[ScheduledMessage]:
        """标记为错过"""
        return await self._update_status(job_id, "missed")

    async def mark_canceled(self, job_id: str) -> Optional[ScheduledMessage]:
        """标记为已取消"""
        return await self._update_status(job_id, "canceled")

    async def batch_mark_missed(self, cutoff: Optional[int] = None) -> int:
        """批量标记过期 pending 消息为 missed

        Args:
            cutoff: 截止时间戳，默认当前时间

        Returns:
            标记数量
        """
        missed = await self.list_missed(cutoff)
        if not missed:
            return 0

        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                now = int(time.time())
                for msg in missed:
                    db_msg = await session.get(ScheduledMessage, msg.id)
                    if db_msg and db_msg.status == "pending":
                        db_msg.status = "missed"
                        db_msg.executed_at = now
                await session.commit()

        return len(missed)

    async def delete_by_job_id(self, job_id: str) -> bool:
        """根据 job_id 删除定时消息"""
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    stmt = select(ScheduledMessage).where(
                        ScheduledMessage.job_id == job_id
                    )
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
                    stmt = select(ScheduledMessage).where(
                        col(ScheduledMessage.status).in_(["completed", "canceled"]),
                        ScheduledMessage.executed_at < before,  # type: ignore[operator]
                    )
                    records = (await session.exec(stmt)).all()
                    for record in records:
                        await session.delete(record)
                    return len(records)

    async def _update_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional[ScheduledMessage]:
        """统一状态更新逻辑"""
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                stmt = select(ScheduledMessage).where(ScheduledMessage.job_id == job_id)
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
