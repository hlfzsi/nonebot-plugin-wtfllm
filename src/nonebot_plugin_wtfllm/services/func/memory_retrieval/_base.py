from abc import ABC, abstractmethod

from ....memory.items import MemorySource


class RetrievalTask(ABC):
    """记忆检索任务抽象基类

    每个子类封装一种独立的记忆检索逻辑，
    通过 ``execute()`` 返回一组 ``MemorySource``。
    """

    @abstractmethod
    async def execute(self) -> set[MemorySource]:
        """执行检索，返回记忆源集合"""
        ...
