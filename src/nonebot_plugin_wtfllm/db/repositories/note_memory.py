"""NoteMemory Repository"""

__all__ = ["NoteMemoryRepository"]

import time
from typing import List, Optional

from sqlmodel import select, col

from .base import BaseRepository
from ..engine import SESSION_MAKER, WRITE_LOCK
from ..models.note_memory import NoteMemoryTable
from ...memory.items.note import Note


class NoteMemoryRepository(BaseRepository[NoteMemoryTable]):
    """会话级短期备忘数据访问层。"""

    def __init__(self):
        super().__init__(NoteMemoryTable)

    async def save_note(self, note: Note) -> NoteMemoryTable:
        record = NoteMemoryTable.model_validate(note.model_dump())
        return await self.save(record)

    async def get_note_by_id(self, storage_id: str) -> Optional[Note]:
        async with SESSION_MAKER() as session:
            record = await session.get(NoteMemoryTable, storage_id)
            return Note.model_validate(record.model_dump()) if record else None

    async def get_by_session(
        self,
        agent_id: str,
        group_id: str | None = None,
        user_id: str | None = None,
        *,
        include_expired: bool = False,
        limit: int = 1000,
    ) -> List[Note]:
        async with SESSION_MAKER() as session:
            stmt = select(NoteMemoryTable).where(NoteMemoryTable.agent_id == agent_id)
            if group_id is not None:
                stmt = stmt.where(NoteMemoryTable.group_id == group_id).where(
                    NoteMemoryTable.user_id == None  # noqa: E711
                )
            elif user_id is not None:
                stmt = stmt.where(NoteMemoryTable.user_id == user_id).where(
                    NoteMemoryTable.group_id == None  # noqa: E711
                )
            else:
                raise ValueError("Either group_id or user_id must be provided")

            if not include_expired:
                stmt = stmt.where(NoteMemoryTable.expires_at > int(time.time()))

            stmt = stmt.order_by(col(NoteMemoryTable.expires_at)).limit(limit)
            records = (await session.exec(stmt)).all()
            return [Note.model_validate(record.model_dump()) for record in records]

    async def count_by_session(
        self,
        agent_id: str,
        group_id: str | None = None,
        user_id: str | None = None,
        *,
        include_expired: bool = False,
    ) -> int:
        notes = await self.get_by_session(
            agent_id=agent_id,
            group_id=group_id,
            user_id=user_id,
            include_expired=include_expired,
        )
        return len(notes)

    async def delete_by_storage_ids(self, storage_ids: List[str]) -> int:
        deleted = 0
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    for storage_id in storage_ids:
                        record = await session.get(NoteMemoryTable, storage_id)
                        if record:
                            await session.delete(record)
                            deleted += 1
        return deleted

    async def delete_expired_by_session(
        self,
        agent_id: str,
        group_id: str | None = None,
        user_id: str | None = None,
    ) -> int:
        deleted = 0
        now = int(time.time())
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    stmt = select(NoteMemoryTable).where(
                        NoteMemoryTable.agent_id == agent_id,
                        NoteMemoryTable.expires_at <= now,
                    )
                    if group_id is not None:
                        stmt = stmt.where(NoteMemoryTable.group_id == group_id).where(
                            NoteMemoryTable.user_id == None  # noqa: E711
                        )
                    elif user_id is not None:
                        stmt = stmt.where(NoteMemoryTable.user_id == user_id).where(
                            NoteMemoryTable.group_id == None  # noqa: E711
                        )
                    else:
                        raise ValueError("Either group_id or user_id must be provided")

                    records = (await session.exec(stmt)).all()
                    for record in records:
                        await session.delete(record)
                        deleted += 1
        return deleted

    async def delete_expired_global(self) -> int:
        deleted = 0
        now = int(time.time())
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                async with session.begin():
                    stmt = select(NoteMemoryTable).where(NoteMemoryTable.expires_at <= now)
                    records = (await session.exec(stmt)).all()
                    for record in records:
                        await session.delete(record)
                        deleted += 1
        return deleted