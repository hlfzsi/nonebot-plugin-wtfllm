from dataclasses import dataclass

from ._base import RetrievalTask
from ....memory.items import MemorySource
from ....memory.items.core_memory import CoreMemoryBlock
from ....v_db import core_memory_repo


@dataclass
class CoreMemoryTask(RetrievalTask):
    """当前会话核心记忆检索"""

    agent_id: str
    group_id: str | None = None
    user_id: str | None = None
    prefix: str = "<core_memory>"
    suffix: str = "</core_memory>"

    async def execute(self) -> set[MemorySource]:
        memories = await core_memory_repo.get_by_session(
            agent_id=self.agent_id,
            group_id=self.group_id,
            user_id=self.user_id,
        )
        if not memories:
            return set()

        block = CoreMemoryBlock(
            memories=memories,
            prefix=self.prefix,
            suffix=self.suffix,
        )
        return {block}


@dataclass
class CrossSessionMemoryTask(RetrievalTask):
    """跨会话核心记忆语义搜索"""

    agent_id: str
    query: str
    exclude_group_id: str | None = None
    exclude_user_id: str | None = None
    limit: int = 5
    prefix: str = "<cross_session_memory>"
    suffix: str = "</cross_session_memory>"

    async def execute(self) -> set[MemorySource]:
        results = await core_memory_repo.search_cross_session(
            agent_id=self.agent_id,
            query=self.query,
            exclude_group_id=self.exclude_group_id,
            exclude_user_id=self.exclude_user_id,
            limit=self.limit,
        )
        if not results:
            return set()

        memories = [r.item for r in results]
        block = CoreMemoryBlock(
            memories=memories,
            prefix=self.prefix,
            suffix=self.suffix,
        )
        return {block}


@dataclass
class EntityMemoryTask(RetrievalTask):
    """按实体ID搜索相关核心记忆（语义搜索 + 实体过滤）"""

    agent_id: str
    query: str
    entity_ids: list[str]
    limit: int = 5
    prefix: str = "<core_memory>"
    suffix: str = "</core_memory>"

    async def execute(self) -> set[MemorySource]:
        results = await core_memory_repo.search_by_entities(
            agent_id=self.agent_id,
            query=self.query,
            entity_ids=self.entity_ids,
            limit=self.limit,
        )
        if not results:
            return set()

        block = CoreMemoryBlock(
            memories=[r.item for r in results],
            prefix=self.prefix,
            suffix=self.suffix,
        )
        return {block}
