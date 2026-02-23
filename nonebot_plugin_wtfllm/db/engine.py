__all__ = ["ENGINE", "WRITE_LOCK", "SESSION_MAKER"]

import asyncio
import sqlite3
from typing import Any

import orjson
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy import event

from ..utils import DATABASE_DATA_DIR

WRITE_LOCK = asyncio.Lock()

_db_path = DATABASE_DATA_DIR / "UserData.db"
_DATABASE_URL = f"sqlite+aiosqlite:///{_db_path.as_posix()}"

ENGINE = create_async_engine(
    _DATABASE_URL,
    connect_args={"timeout": 30},
    json_serializer=lambda obj: orjson.dumps(obj).decode(),
    json_deserializer=orjson.loads,
)


@event.listens_for(ENGINE.sync_engine, "connect")
def set_sqlite_pragma(
    dbapi_connection: sqlite3.Connection, connection_record: Any
) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SESSION_MAKER = async_sessionmaker(ENGINE, class_=AsyncSession, expire_on_commit=False)
