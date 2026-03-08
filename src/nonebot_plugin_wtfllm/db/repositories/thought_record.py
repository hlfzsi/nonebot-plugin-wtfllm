"""显式思考记录 Repository"""

__all__ = ["ThoughtRecordRepository"]

from time import time
from typing import List, Optional

from sqlalchemy import  select as sa_select
from sqlmodel import col, desc, select

from .base import BaseRepository
from ..engine import SESSION_MAKER, WRITE_LOCK
from ..models.thought_record import ThoughtRecordTable


class ThoughtRecordRepository(BaseRepository[ThoughtRecordTable]):
    """显式思考记录数据访问层"""

    def __init__(self):
        super().__init__(ThoughtRecordTable)

    async def save_record(
        self,
        thought_of_chain: str,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ThoughtRecordTable:
        record = ThoughtRecordTable(
            run_id=run_id,
            agent_id=agent_id,
            group_id=group_id,
            user_id=user_id,
            thought_of_chain=thought_of_chain,
            timestamp=int(time()),
        )
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
        return record

    async def get_recent(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 3,
    ) -> List[ThoughtRecordTable]:
        if limit <= 0:
            return []

        async with SESSION_MAKER() as session:
            id_col = col(ThoughtRecordTable.id)
            agent_id_col = col(ThoughtRecordTable.agent_id)
            group_id_col = col(ThoughtRecordTable.group_id)
            user_id_col = col(ThoughtRecordTable.user_id)
            timestamp_col = col(ThoughtRecordTable.timestamp)

            sub = sa_select(id_col).where(agent_id_col == agent_id)
            if group_id is not None:
                sub = sub.where(group_id_col == group_id)
            else:
                sub = sub.where(group_id_col == None)  # noqa: E711
                sub = sub.where(user_id_col == user_id)

            sub = sub.order_by(desc(timestamp_col), desc(id_col)).limit(limit).subquery()

            stmt = (
                select(ThoughtRecordTable)
                .where(id_col.in_(sa_select(sub.c.id)))
                .order_by(timestamp_col, id_col)
            )
            return list((await session.exec(stmt)).all())