"""工具调用记录 Repository

封装工具调用记录的数据访问操作。
"""

__all__ = ["ToolCallRecordRepository"]

import uuid
from time import time
from typing import List, Optional, TYPE_CHECKING

from sqlmodel import select, desc, col
from sqlalchemy import func as sa_func, select as sa_select
from .base import BaseRepository
from ..models.tool_call_record import ToolCallRecordTable
from ..engine import SESSION_MAKER, WRITE_LOCK

if TYPE_CHECKING:
    from ...llm.deps import ToolCallInfo


class ToolCallRecordRepository(BaseRepository[ToolCallRecordTable]):
    """工具调用记录数据访问层"""

    def __init__(self):
        super().__init__(ToolCallRecordTable)

    async def save_empty_record(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """保存一条空的工具调用记录，主要用于占位"""
        record = ToolCallRecordTable(
            run_id=str(uuid.uuid4()),
            run_step=0,
            agent_id=agent_id,
            group_id=group_id,
            user_id=user_id,
            tool_name="No Tool was called",
            kwargs={},
            timestamp=int(time()),
        )
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                session.add(record)
                await session.commit()
                return 1

    async def save_batch(self, records: List[ToolCallRecordTable]) -> int:
        """批量保存工具调用记录

        Args:
            records: ToolCallRecordTable 实例列表

        Returns:
            成功存储的记录数量
        """
        if not records:
            return 0
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                session.add_all(records)
                await session.commit()
                return len(records)

    async def get_recent(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 3,
    ) -> List[ToolCallRecordTable]:
        """获取指定会话最近 N 次 Agent.run() 的工具调用记录

        Args:
            agent_id: 智能体ID
            group_id: 群组ID (群聊时)
            user_id: 用户ID (私聊时)
            limit: 返回最近几次 Agent.run() 的记录

        Returns:
            ToolCallRecordTable 列表，按 timestamp ASC 排序
        """
        if limit <= 0:
            return []
        async with SESSION_MAKER() as session:
            run_id_col = col(ToolCallRecordTable.run_id)
            agent_id_col = col(ToolCallRecordTable.agent_id)
            group_id_col = col(ToolCallRecordTable.group_id)
            user_id_col = col(ToolCallRecordTable.user_id)
            timestamp_col = col(ToolCallRecordTable.timestamp)

            sub = sa_select(run_id_col).where(agent_id_col == agent_id)
            if group_id is not None:
                sub = sub.where(group_id_col == group_id)
            else:
                sub = sub.where(group_id_col == None)  # noqa: E711
                sub = sub.where(user_id_col == user_id)
            sub = (
                sub.group_by(run_id_col)
                .order_by(desc(sa_func.max(timestamp_col)))
                .limit(limit)
            )
            sub = sub.subquery()

            stmt = (
                select(ToolCallRecordTable)
                .where(run_id_col.in_(sa_select(sub.c.run_id)))
                .order_by(timestamp_col)
            )
            return list((await session.exec(stmt)).all())

    async def save_batch_from_tool_call_info(
        self,
        infos: List["ToolCallInfo"],
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """从 ToolCallInfo 列表批量保存工具调用记录

        Args:
            infos: ToolCallInfo 实例列表
            agent_id: 智能体ID
            group_id: 群组ID (群聊时)
            user_id: 用户ID (私聊时)

        Returns:
            成功存储的记录数量
        """
        records = [
            ToolCallRecordTable.from_tool_call_info(
                info=info,
                agent_id=agent_id,
                group_id=group_id,
                user_id=user_id,
            )
            for info in infos
        ]
        return await self.save_batch(records)
