"""核心记忆 Repository"""

__all__ = ["CoreMemoryRepository"]

from typing import List, Optional, TYPE_CHECKING
from qdrant_client import models

from .base import VectorRepository, SearchResult
from ..models.core_memory import CoreMemoryPayload

if TYPE_CHECKING:
    from ...memory.items.core_memory import CoreMemory


class CoreMemoryRepository(VectorRepository[CoreMemoryPayload]):
    """核心记忆 Repository

    提供 CoreMemory 的存储、检索和语义搜索功能。
    """

    def __init__(self) -> None:
        super().__init__(CoreMemoryPayload, CoreMemoryPayload.collection_name)

    async def save_core_memory(self, memory: "CoreMemory") -> str:
        """保存 CoreMemory

        Args:
            memory: CoreMemory 实例

        Returns:
            存储的 storage_id
        """
        payload = CoreMemoryPayload.from_core_memory(memory)
        return await self.upsert(payload)

    async def save_many_core_memories(self, memories: List["CoreMemory"]) -> List[str]:
        """批量保存 CoreMemory

        Args:
            memories: CoreMemory 列表

        Returns:
            存储的 storage_id 列表
        """
        payloads = [CoreMemoryPayload.from_core_memory(m) for m in memories]
        return await self.upsert_many(payloads)

    async def get_core_memory_by_id(self, storage_id: str) -> Optional["CoreMemory"]:
        """根据 storage_id 获取 CoreMemory

        Args:
            storage_id: 核心记忆存储ID

        Returns:
            CoreMemory 实例，不存在则返回 None
        """
        payload = await self.get_by_id(storage_id)
        return payload.to_core_memory() if payload else None

    async def get_by_session(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List["CoreMemory"]:
        """获取指定会话的所有核心记忆

        Args:
            agent_id: 智能体ID
            group_id: 群组ID（群聊场景）
            user_id: 用户ID（私聊场景）

        Returns:
            按 updated_at 升序排列的 CoreMemory 列表
        """
        conditions: List[models.Condition] = [
            self.match_keyword("agent_id", agent_id),
        ]

        if group_id is not None:
            conditions.append(self.match_keyword("group_id", group_id))
            conditions.append(self.is_null("user_id"))
        elif user_id is not None:
            conditions.append(self.match_keyword("user_id", user_id))
            conditions.append(self.is_null("group_id"))
        else:
            raise ValueError("Either group_id or user_id must be provided")

        filter_ = models.Filter(must=conditions)
        payloads, _ = await self.scroll(
            filter_=filter_,
            limit=1000,
            order_by="updated_at",
            order_type="asc",
        )
        return [p.to_core_memory() for p in payloads]

    async def search_cross_session(
        self,
        agent_id: str,
        query: str,
        exclude_group_id: Optional[str] = None,
        exclude_user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[SearchResult["CoreMemory"]]:
        """跨会话语义检索核心记忆

        排除当前会话的记忆，检索其他会话中与查询相关的核心记忆。

        Args:
            agent_id: 智能体ID
            query: 搜索查询文本
            exclude_group_id: 要排除的群组ID
            exclude_user_id: 要排除的用户ID
            limit: 返回数量限制

        Returns:
            SearchResult 列表
        """
        must_conditions: List[models.Condition] = [
            self.match_keyword("agent_id", agent_id),
        ]

        must_not_conditions: List[models.Condition] = []
        if exclude_group_id is not None:
            must_not_conditions.append(
                self.match_keyword("group_id", exclude_group_id)
            )
        if exclude_user_id is not None:
            must_not_conditions.append(
                self.match_keyword("user_id", exclude_user_id)
            )

        filter_ = models.Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None,
        )

        results = await self.search(
            query=query,
            limit=limit,
            filter_=filter_,
        )

        return [
            SearchResult(
                item=r.item.to_core_memory(),
                score=r.score,
            )
            for r in results
        ]

    async def search_by_entities(
        self,
        agent_id: str,
        entity_ids: List[str],
        query: str,
        limit: int = 5,
    ) -> List[SearchResult["CoreMemory"]]:
        """搜索与指定实体相关的核心记忆

        Args:
            agent_id: 智能体ID
            entity_ids: 实体ID列表
            query: 语义查询文本
            limit: 返回数量限制

        Returns:
            SearchResult 列表
        """
        conditions: List[models.Condition] = [
            self.match_keyword("agent_id", agent_id),
            self.match_any("related_entities", entity_ids),
        ]
        filter_ = models.Filter(must=conditions)

        results = await self.search(
            query=query,
            limit=limit,
            filter_=filter_,
        )
        return [
            SearchResult(
                item=r.item.to_core_memory(),
                score=r.score,
            )
            for r in results
        ]

    async def delete_by_storage_ids(self, storage_ids: List[str]) -> bool:
        """批量删除指定的核心记忆

        Args:
            storage_ids: 要删除的 storage_id 列表

        Returns:
            操作是否成功
        """
        return await self.delete_many_by_ids(storage_ids)

    async def delete_by_session(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """删除指定会话的所有核心记忆

        Args:
            agent_id: 智能体ID
            group_id: 群组ID
            user_id: 用户ID

        Returns:
            操作是否成功
        """
        conditions: List[models.Condition] = [
            self.match_keyword("agent_id", agent_id),
        ]
        if group_id is not None:
            conditions.append(self.match_keyword("group_id", group_id))
        elif user_id is not None:
            conditions.append(self.match_keyword("user_id", user_id))

        filter_ = models.Filter(must=conditions)
        return await self.delete_by_filter(filter_)

    async def count_by_session(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """统计指定会话的核心记忆数量

        Args:
            agent_id: 智能体ID
            group_id: 群组ID
            user_id: 用户ID

        Returns:
            核心记忆数量
        """
        conditions: List[models.Condition] = [
            self.match_keyword("agent_id", agent_id),
        ]
        if group_id is not None:
            conditions.append(self.match_keyword("group_id", group_id))
        elif user_id is not None:
            conditions.append(self.match_keyword("user_id", user_id))

        filter_ = models.Filter(must=conditions)
        return await self.count(filter_)
