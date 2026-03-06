"""Repository 基类

提供通用的数据访问层抽象。
"""

__all__ = ["BaseRepository"]

from abc import ABC
from typing import Generic, TypeVar, Optional

from sqlmodel import SQLModel
from ..engine import SESSION_MAKER, WRITE_LOCK

T = TypeVar("T", bound=SQLModel)


class BaseRepository(ABC, Generic[T]):
    """数据访问层基类

    提供基础的 CRUD 操作，子类可以扩展特定的业务方法。
    """

    def __init__(self, model_class: type[T]):
        self.model_class = model_class

    async def get_by_id(self, id_value: str) -> Optional[T]:
        """根据主键查询实体

        Args:
            id_value: 主键值

        Returns:
            找到的实体对象，不存在则返回 None
        """
        async with SESSION_MAKER() as session:
            return await session.get(self.model_class, id_value)

    async def save(self, entity: T) -> T:
        """保存实体 (INSERT 或 UPDATE)

        Args:
            entity: 要保存的实体对象

        Returns:
            保存后的实体对象（已刷新）
        """
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                merged_entity = await session.merge(entity)
                await session.flush()
                await session.refresh(merged_entity)
                await session.commit()
        return merged_entity
