__all__ = ["init_db", "shutdown_db"]

from sqlmodel import SQLModel
from .engine import ENGINE


async def init_db():
    """初始化数据库（创建所有表）

    在应用启动时调用，自动创建所有 SQLModel 定义的表。
    """
    async with ENGINE.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def shutdown_db():
    """关闭数据库连接

    在应用关闭时调用，释放数据库连接资源。
    """
    await ENGINE.dispose()
