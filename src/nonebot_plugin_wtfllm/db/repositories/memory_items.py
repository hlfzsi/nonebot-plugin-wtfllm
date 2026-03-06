"""记忆项 Repository

封装记忆项的数据访问操作。
"""

__all__ = ["MemoryItemRepository"]

from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import literal_column, select as sa_select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlmodel import delete, select, desc, col, func
from .base import BaseRepository
from ..models import MemoryItemTable
from ..engine import SESSION_MAKER, WRITE_LOCK

if TYPE_CHECKING:
    from ...memory import MemoryItemUnion


class MemoryItemRepository(BaseRepository[MemoryItemTable]):
    """记忆项数据访问层"""

    def __init__(self):
        super().__init__(MemoryItemTable)

    async def get_by_message_id(self, message_id: str) -> Optional["MemoryItemUnion"]:
        """根据 message_id 获取记忆项

        Args:
            message_id: 消息ID

        Returns:
            MemoryItem 实例，不存在则返回 None
        """
        async with SESSION_MAKER() as session:
            record = await session.get(MemoryItemTable, message_id)
            if record:
                return record.to_MemoryItem()
            return None

    async def save_memory_item(self, item: "MemoryItemUnion") -> MemoryItemTable:
        """存储单个记忆项

        Args:
            item: MemoryItem 实例

        Returns:
            存储后的 MemoryItemTable 对象
        """
        record = MemoryItemTable.from_MemoryItem(item)
        return await self.save(record)

    async def save_many(self, items: List["MemoryItemUnion"]) -> int:
        """批量快速存储记忆项

        Args:
            items: MemoryItem 实例列表

        Returns:
            成功存储的记录数量
        """
        if not items:
            return 0

        records = [MemoryItemTable.from_MemoryItem(item) for item in items]

        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                try:
                    session.add_all(records)
                    await session.commit()
                    return len(records)

                except IntegrityError:
                    await session.rollback()
                    saved_count = 0

                    for record in records:
                        try:
                            async with session.begin_nested():
                                await session.merge(record)
                                saved_count += 1
                        except (IntegrityError, SQLAlchemyError):
                            continue

                    await session.commit()
                    return saved_count

                except SQLAlchemyError:
                    await session.rollback()
                    raise

    async def list_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List["MemoryItemUnion"]:
        """按 agent_id 查询记忆列表

        Args:
            agent_id: 智能体ID
            limit: 返回数量限制

        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.agent_id == agent_id)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_by_group(
        self,
        group_id: str,
        agent_id: str,
        limit: int = 30,
    ) -> List["MemoryItemUnion"]:
        """按 group_id 查询记忆列表

        Args:
            group_id: 群组ID
            limit: 返回数量限制

        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.group_id == group_id)
                .where(MemoryItemTable.agent_id == agent_id)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_by_group_after(
        self,
        group_id: str,
        agent_id: str,
        timestamp: int,
        limit: int | None = 30,
    ):
        """按 group_id 和时间戳 查询记忆列表
        Args:
            group_id: 群组ID
            timestamp: 时间戳，获取在此时间戳时和之后的记忆
            limit: 返回数量限制
        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.group_id == group_id)
                .where(MemoryItemTable.agent_id == agent_id)
                .where(MemoryItemTable.created_at >= timestamp)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_by_group_before(
        self,
        group_id: str,
        agent_id: str,
        timestamp: int,
        limit: int | None = 30,
    ):
        """按 group_id 和时间戳 查询记忆列表

        Args:
            group_id: 群组ID
            timestamp: 时间戳，获取在此时间戳时和之前的记忆
            limit: 返回数量限制
        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.group_id == group_id)
                .where(MemoryItemTable.agent_id == agent_id)
                .where(MemoryItemTable.created_at <= timestamp)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_by_user(
        self,
        user_id: str,
        agent_id: str,
        limit: int | None = 30,
    ) -> List["MemoryItemUnion"]:
        """按 user_id 查询记忆列表

        Args:
            user_id: 用户ID
            limit: 返回数量限制

        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.sender == user_id)
                .where(MemoryItemTable.agent_id == agent_id)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_in_private_by_user(
        self,
        user_id: str,
        agent_id: str,
        limit: int | None = 30,
    ) -> List["MemoryItemUnion"]:
        """按 user_id 查询私聊记忆

        Args:
            user_id: 用户ID
            limit: 返回数量限制

        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.group_id == None)  # noqa: E711
                .where(MemoryItemTable.user_id == user_id)
                .where(MemoryItemTable.agent_id == agent_id)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_in_private_by_user_before(
        self,
        user_id: str,
        agent_id: str,
        timestamp: int,
        limit: int | None = 30,
    ) -> List["MemoryItemUnion"]:
        """按 user_id 查询指定时间戳之前的私聊记忆

        Args:
            user_id: 用户ID
            agent_id: AgentID
            timestamp: 时间戳上界（不含），获取此时间戳之前的记忆
            limit: 返回数量限制
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.group_id == None)  # noqa: E711
                .where(MemoryItemTable.user_id == user_id)
                .where(MemoryItemTable.agent_id == agent_id)
                .where(MemoryItemTable.created_at < timestamp)
                .order_by(desc(MemoryItemTable.created_at))
                .limit(limit)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def delete_by_message_id(self, message_id: str) -> bool:
        """删除指定记忆

        Args:
            message_id: 消息ID

        Returns:
            是否成功删除
        """
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    record = await session.get(MemoryItemTable, message_id)
                    if record:
                        await session.delete(record)
                        return True
        return False

    async def delete_many_by_message_ids(self, message_ids: List[str]) -> int:
        """批量删除记忆

        Args:
            message_ids: 消息ID列表

        Returns:
            成功删除的记录数量
        """
        if not message_ids:
            return 0

        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    result = await session.exec(
                        delete(MemoryItemTable).where(
                            col(MemoryItemTable.message_id).in_(message_ids)
                        )
                    )
                    return result.rowcount

    async def get_many_by_message_ids(
        self, message_ids: List[str]
    ) -> List["MemoryItemUnion"]:
        """批量获取记忆项

        Args:
            message_ids: 消息ID列表

        Returns:
            MemoryItem 列表（顺序不保证与输入一致）
        """
        if not message_ids:
            return []

        async with SESSION_MAKER() as session:
            stmt = select(MemoryItemTable).where(
                col(MemoryItemTable.message_id).in_(message_ids)
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_by_timestamp_before(
        self, timestamp: int, agent_id: str
    ) -> List["MemoryItemUnion"]:
        """按时间戳查询记忆项

        Args:
            timestamp: 时间戳，获取在此时间戳时和之前的记忆

        Returns:
            MemoryItem 列表
        """
        async with SESSION_MAKER() as session:
            stmt = (
                select(MemoryItemTable)
                .where(MemoryItemTable.created_at <= timestamp)
                .where(MemoryItemTable.agent_id == agent_id)
                .order_by(desc(MemoryItemTable.created_at))
            )
            records = (await session.exec(stmt)).all()
            return [record.to_MemoryItem() for record in records]

    async def get_chain_by_message_ids(
        self,
        message_ids: List[str],
        max_depth: int = 10,
    ) -> List["MemoryItemUnion"]:
        """获取记忆链：从给定的 message_id 开始，递归获取所有关联的记忆项

        使用递归 CTE

        Args:
            message_ids: 起始消息ID列表
            max_depth: 最大追溯深度，防止无限循环

        Returns:
            包含所有关联记忆的 MemoryItem 列表（按 created_at 排序）
        """
        if not message_ids:
            return []

        col_message_id = col(MemoryItemTable.message_id).label("message_id")
        col_related_message_id = col(MemoryItemTable.related_message_id).label(
            "related_message_id"
        )
        col_sender = col(MemoryItemTable.sender).label("sender")
        col_content = col(MemoryItemTable.content).label("content")
        col_created_at = col(MemoryItemTable.created_at).label("created_at")
        col_agent_id = col(MemoryItemTable.agent_id).label("agent_id")
        col_memory_type = col(MemoryItemTable.memory_type).label("memory_type")
        col_group_id = col(MemoryItemTable.group_id).label("group_id")
        col_user_id = col(MemoryItemTable.user_id).label("user_id")

        base_query = sa_select(
            col_message_id,
            col_related_message_id,
            col_sender,
            col_content,
            col_created_at,
            col_agent_id,
            col_memory_type,
            col_group_id,
            col_user_id,
            literal_column("0").label("depth"),
        ).where(col(MemoryItemTable.message_id).in_(message_ids))

        cte = base_query.cte(name="memory_chain", recursive=True)

        recursive_query = (
            sa_select(
                col(MemoryItemTable.message_id).label("message_id"),
                col(MemoryItemTable.related_message_id).label("related_message_id"),
                col(MemoryItemTable.sender).label("sender"),
                col(MemoryItemTable.content).label("content"),
                col(MemoryItemTable.created_at).label("created_at"),
                col(MemoryItemTable.agent_id).label("agent_id"),
                col(MemoryItemTable.memory_type).label("memory_type"),
                col(MemoryItemTable.group_id).label("group_id"),
                col(MemoryItemTable.user_id).label("user_id"),
                (cte.c.depth + 1).label("depth"),
            )
            .join(cte, col(MemoryItemTable.message_id) == cte.c.related_message_id)
            .where(cte.c.depth < max_depth)
        )

        cte = cte.union_all(recursive_query)

        final_query = (
            sa_select(
                cte.c.message_id,
                cte.c.related_message_id,
                cte.c.sender,
                cte.c.content,
                cte.c.created_at,
                cte.c.agent_id,
                cte.c.memory_type,
                cte.c.group_id,
                cte.c.user_id,
            )
            .distinct()
            .order_by(cte.c.created_at)
        )

        async with SESSION_MAKER() as session:
            rows = (await session.exec(final_query)).all()  # type: ignore[arg-type]

            items: List["MemoryItemUnion"] = []
            for row in rows:
                record = MemoryItemTable(
                    message_id=row.message_id,
                    related_message_id=row.related_message_id,
                    sender=row.sender,
                    content=row.content,
                    created_at=row.created_at,
                    agent_id=row.agent_id,
                    memory_type=row.memory_type,
                    group_id=row.group_id,
                    user_id=row.user_id,
                )
                items.append(record.to_MemoryItem())

            return items

    async def get_active_group_ids(
        self,
        agent_id: str,
        since: int,
    ) -> list[str]:
        """
        获取有活跃记录的去重群组 ID 列表。

        Args:
            agent_id: 智能体ID
            since: 起始时间戳（秒）
        """
        stmt = (
            sa_select(col(MemoryItemTable.group_id))
            .where(col(MemoryItemTable.agent_id) == agent_id)
            .where(col(MemoryItemTable.created_at) >= since)
            .where(col(MemoryItemTable.group_id).is_not(None))
            .group_by(col(MemoryItemTable.group_id))
        )
        async with SESSION_MAKER() as session:
            raw = (await session.exec(stmt)).all()  # type: ignore[arg-type]
            return [str(row[0]) for row in raw if row[0] is not None]

    async def get_group_activity_bins(
        self,
        agent_id: str,
        group_ids: list[str],
        since: int,
        utc_offset: int = 0,
        min_repeat_days: int = 1,
    ) -> list[tuple[str | None, str | None, str, int, int]]:
        """获取指定群聊的活跃分钟分布。

        仅返回「用户级最高活跃天数 ≥ min_repeat_days」的用户全部数据，
        以在 DB 层过滤不可能通过算法前置检查的低活跃用户。

        Args:
            agent_id: 智能体 ID
            group_ids: 要查询的群组 ID 列表
            since: 起始时间戳（秒）
            utc_offset: 本地时区偏移（秒）
            min_repeat_days: 用户最高活跃分钟桶需达到的天数阈值

        Returns:
            list of (group_id, user_id, sender, minute_of_day, active_days)
        """
        offset = int(utc_offset)
        local_ts = f"(memoryitemtable.created_at + {offset})"
        minute_sql = f"({local_ts} % 86400) / 60"
        day_sql = f"({local_ts}) / 86400"

        # 内层：同一天同一分钟去重
        dedup = (
            sa_select(
                col(MemoryItemTable.group_id).label("group_id"),
                col(MemoryItemTable.sender).label("sender"),
                literal_column(minute_sql).label("minute_of_day"),
                literal_column(day_sql).label("day_number"),
            )
            .where(col(MemoryItemTable.agent_id) == agent_id)
            .where(col(MemoryItemTable.created_at) >= since)
            .where(col(MemoryItemTable.group_id).in_(group_ids))
            .group_by(
                col(MemoryItemTable.group_id),
                col(MemoryItemTable.sender),
                literal_column("day_number"),
                literal_column("minute_of_day"),
            )
        ).subquery("dedup")

        # 按分钟聚合活跃天数
        agg = (
            sa_select(
                dedup.c.group_id,
                dedup.c.sender,
                dedup.c.minute_of_day,
                func.count().label("active_days"),
            ).group_by(dedup.c.group_id, dedup.c.sender, dedup.c.minute_of_day)
        ).subquery("agg")

        if min_repeat_days > 1:
            # 窗口函数：取每用户最高活跃天数，过滤不达标用户
            user_max = (
                func.max(agg.c.active_days)
                .over(partition_by=[agg.c.group_id, agg.c.sender])
                .label("user_max_days")
            )

            windowed = (
                sa_select(
                    agg.c.group_id,
                    agg.c.sender,
                    agg.c.minute_of_day,
                    agg.c.active_days,
                    user_max,
                )
            ).subquery("windowed")

            stmt = sa_select(
                windowed.c.group_id,
                literal_column("NULL").label("user_id"),
                windowed.c.sender,
                windowed.c.minute_of_day,
                windowed.c.active_days,
            ).where(windowed.c.user_max_days >= min_repeat_days)
        else:
            stmt = sa_select(
                agg.c.group_id,
                literal_column("NULL").label("user_id"),
                agg.c.sender,
                agg.c.minute_of_day,
                agg.c.active_days,
            )

        async with SESSION_MAKER() as session:
            rows = (await session.exec(stmt)).all()  # type: ignore[arg-type]
            return [
                (
                    row.group_id,
                    row.user_id,
                    row.sender,
                    int(row.minute_of_day),
                    int(row.active_days),
                )
                for row in rows
            ]

    async def get_private_activity_bins(
        self,
        agent_id: str,
        since: int,
        utc_offset: int = 0,
        min_repeat_days: int = 1,
    ) -> list[tuple[str | None, str | None, str, int, int]]:
        """获取私聊的活跃分钟分布。

        仅返回「用户级最高活跃天数 ≥ min_repeat_days」的用户全部数据。

        Args:
            agent_id: 智能体 ID
            since: 起始时间戳（秒）
            utc_offset: 本地时区偏移（秒）
            min_repeat_days: 用户最高活跃分钟桶需达到的天数阈值

        Returns:
            list of (group_id, user_id, sender, minute_of_day, active_days)
        """
        offset = int(utc_offset)
        local_ts = f"(memoryitemtable.created_at + {offset})"
        minute_sql = f"({local_ts} % 86400) / 60"
        day_sql = f"({local_ts}) / 86400"

        dedup = (
            sa_select(
                col(MemoryItemTable.user_id).label("user_id"),
                col(MemoryItemTable.sender).label("sender"),
                literal_column(minute_sql).label("minute_of_day"),
                literal_column(day_sql).label("day_number"),
            )
            .where(col(MemoryItemTable.agent_id) == agent_id)
            .where(col(MemoryItemTable.created_at) >= since)
            .where(col(MemoryItemTable.group_id).is_(None))
            .group_by(
                col(MemoryItemTable.user_id),
                col(MemoryItemTable.sender),
                literal_column("day_number"),
                literal_column("minute_of_day"),
            )
        ).subquery("dedup")

        agg = (
            sa_select(
                dedup.c.user_id,
                dedup.c.sender,
                dedup.c.minute_of_day,
                func.count().label("active_days"),
            ).group_by(dedup.c.user_id, dedup.c.sender, dedup.c.minute_of_day)
        ).subquery("agg")

        if min_repeat_days > 1:
            user_max = (
                func.max(agg.c.active_days)
                .over(partition_by=[agg.c.user_id, agg.c.sender])
                .label("user_max_days")
            )

            windowed = (
                sa_select(
                    agg.c.user_id,
                    agg.c.sender,
                    agg.c.minute_of_day,
                    agg.c.active_days,
                    user_max,
                )
            ).subquery("windowed")

            stmt = sa_select(
                literal_column("NULL").label("group_id"),
                windowed.c.user_id,
                windowed.c.sender,
                windowed.c.minute_of_day,
                windowed.c.active_days,
            ).where(windowed.c.user_max_days >= min_repeat_days)
        else:
            stmt = sa_select(
                literal_column("NULL").label("group_id"),
                agg.c.user_id,
                agg.c.sender,
                agg.c.minute_of_day,
                agg.c.active_days,
            )

        async with SESSION_MAKER() as session:
            rows = (await session.exec(stmt)).all()  # type: ignore[arg-type]
            return [
                (
                    row.group_id,
                    row.user_id,
                    row.sender,
                    int(row.minute_of_day),
                    int(row.active_days),
                )
                for row in rows
            ]
