"""知识库 Repository"""

__all__ = ["KnowledgeBaseRepository"]

from typing import List, Optional, TYPE_CHECKING

from qdrant_client import models

from .base import VectorRepository, SearchResult
from ..models.knowledge_base import KnowledgeBasePayload

if TYPE_CHECKING:
    from ...memory.items.knowledge_base import KnowledgeEntry


class KnowledgeBaseRepository(VectorRepository[KnowledgeBasePayload]):
    """知识库 Repository

    提供知识条目的存储、检索和语义搜索功能。
    知识条目全局共享，仅按 agent_id 隔离。
    """

    def __init__(self) -> None:
        super().__init__(KnowledgeBasePayload, KnowledgeBasePayload.collection_name)

    async def save_knowledge(self, entry: "KnowledgeEntry") -> str:
        """保存知识条目

        Args:
            entry: KnowledgeEntry 实例

        Returns:
            存储的 storage_id
        """
        payload = KnowledgeBasePayload.from_knowledge_entry(entry)
        return await self.upsert(payload)

    async def save_many_knowledge(self, entries: List["KnowledgeEntry"]) -> List[str]:
        """批量保存知识条目

        Args:
            entries: KnowledgeEntry 列表

        Returns:
            存储的 storage_id 列表
        """
        payloads = [KnowledgeBasePayload.from_knowledge_entry(e) for e in entries]
        return await self.upsert_many(payloads)

    async def get_knowledge_by_id(
        self, storage_id: str
    ) -> Optional["KnowledgeEntry"]:
        """根据 storage_id 获取知识条目

        Args:
            storage_id: 知识条目存储ID

        Returns:
            KnowledgeEntry 实例，不存在则返回 None
        """
        payload = await self.get_by_id(storage_id)
        return payload.to_knowledge_entry() if payload else None

    async def search_relevant(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> List[SearchResult["KnowledgeEntry"]]:
        """语义搜索相关知识

        Args:
            agent_id: 智能体ID
            query: 搜索查询文本
            limit: 返回数量限制
            category: 可选的分类过滤
        """
        conditions: List[models.Condition] = [
            self.match_keyword("agent_id", agent_id),
        ]
        if category:
            conditions.append(self.match_keyword("category", category))

        filter_ = models.Filter(must=conditions)

        results = await self.search(
            query=query,
            limit=limit,
            filter_=filter_,
        )
        return [
            SearchResult(
                item=r.item.to_knowledge_entry(),
                score=r.score,
            )
            for r in results
        ]

    async def find_similar(
        self,
        agent_id: str,
        query: str,
        limit: int = 3,
    ) -> List[SearchResult["KnowledgeEntry"]]:
        """查找语义相似的知识条目（用于去重检测）

        Args:
            agent_id: 智能体ID
            query: 待检测文本
            limit: 返回数量限制

        Returns:
            高相似度的 SearchResult 列表
        """
        filter_ = models.Filter(
            must=[self.match_keyword("agent_id", agent_id)]
        )
        results = await self.search(query=query, limit=limit, filter_=filter_)
        return [
            SearchResult(
                item=r.item.to_knowledge_entry(),
                score=r.score,
            )
            for r in results
        ]

    async def search_by_category(
        self,
        agent_id: str,
        category: str,
        limit: int = 100,
    ) -> List["KnowledgeEntry"]:
        """获取指定分类的知识条目

        Args:
            agent_id: 智能体ID
            category: 分类名称
            limit: 返回数量限制
        """
        filter_ = models.Filter(
            must=[
                self.match_keyword("agent_id", agent_id),
                self.match_keyword("category", category),
            ]
        )
        payloads, _ = await self.scroll(
            filter_=filter_,
            limit=limit,
            order_by="updated_at",
            order_type="desc",
        )
        return [p.to_knowledge_entry() for p in payloads]

    async def search_by_tags(
        self,
        agent_id: str,
        tags: List[str],
        limit: int = 100,
    ) -> List["KnowledgeEntry"]:
        """按标签搜索知识条目

        Args:
            agent_id: 智能体ID
            tags: 标签列表
            limit: 返回数量限制
        """
        filter_ = models.Filter(
            must=[
                self.match_keyword("agent_id", agent_id),
                self.match_any("tags", tags),
            ]
        )
        payloads, _ = await self.scroll(
            filter_=filter_,
            limit=limit,
        )
        return [p.to_knowledge_entry() for p in payloads]

    async def delete_knowledge(self, storage_id: str) -> bool:
        """删除知识条目

        Args:
            storage_id: 要删除的知识条目ID

        Returns:
            操作是否成功
        """
        return await self.delete_by_id(storage_id)

    async def delete_by_agent(self, agent_id: str) -> bool:
        """删除指定 agent 的所有知识条目

        Args:
            agent_id: 智能体ID

        Returns:
            操作是否成功
        """
        filter_ = models.Filter(
            must=[self.match_keyword("agent_id", agent_id)]
        )
        return await self.delete_by_filter(filter_)

    async def count_by_agent(self, agent_id: str) -> int:
        """统计指定 agent 的知识条目数量

        Args:
            agent_id: 智能体ID

        Returns:
            知识条目数量
        """
        filter_ = models.Filter(
            must=[self.match_keyword("agent_id", agent_id)]
        )
        return await self.count(filter_)
