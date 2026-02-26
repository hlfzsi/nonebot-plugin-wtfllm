__all__ = ["ENGINE", "WRITE_LOCK", "SESSION_MAKER"]

import asyncio
import sqlite3
from typing import Any

from orjson import dumps, loads
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import event

from ..utils import DATABASE_DATA_DIR, APP_CONFIG


class _Lock:
    __slots__ = ("_enabled", "_lock")

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        if self._enabled:
            await self._lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._enabled and self._lock.locked():
            self._lock.release()


WRITE_LOCK = _Lock()
_db_path = DATABASE_DATA_DIR / "UserData.db"
_DATABASE_URL = APP_CONFIG.database_url or f"sqlite+aiosqlite:///{_db_path.as_posix()}"

ENGINE = create_async_engine(
    _DATABASE_URL,
    connect_args={"timeout": 30},
    json_serializer=lambda obj: dumps(obj).decode("utf-8"),
    json_deserializer=loads,
)


@event.listens_for(ENGINE.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    if isinstance(dbapi_connection, sqlite3.Connection):
        WRITE_LOCK._enabled = True
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SESSION_MAKER = async_sessionmaker(ENGINE, class_=AsyncSession, expire_on_commit=False)
