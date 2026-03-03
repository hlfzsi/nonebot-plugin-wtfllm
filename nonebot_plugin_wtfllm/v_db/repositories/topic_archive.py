"""话题归档 Repository"""

__all__ = ["TopicArchiveRepository"]

from typing import List, Optional

from qdrant_client import models

from ..models.topic_archive import TopicArchivePayload
from .base import SearchResult, VectorRepository


class TopicArchiveRepository(VectorRepository[TopicArchivePayload]):
    def __init__(self) -> None:
        super().__init__(TopicArchivePayload, TopicArchivePayload.collection_name)

    async def search_by_session(
        self,
        agent_id: str,
        query: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 3,
        score_threshold: float = 0.3,
    ) -> List[SearchResult[TopicArchivePayload]]:
        """按会话搜索归档话题。"""
        conditions: list[models.Condition] = [self.match_keyword("agent_id", agent_id)]
        if group_id is not None:
            conditions.append(self.match_keyword("group_id", group_id))
        if user_id is not None:
            conditions.append(self.match_keyword("user_id", user_id))

        filter_ = models.Filter(must=conditions)
        return await self.search(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            filter_=filter_,
        )

    async def delete_by_session(
        self,
        agent_id: str,
        group_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """删除某会话的所有归档。"""
        conditions: list[models.Condition] = [self.match_keyword("agent_id", agent_id)]
        if group_id is not None:
            conditions.append(self.match_keyword("group_id", group_id))
        if user_id is not None:
            conditions.append(self.match_keyword("user_id", user_id))

        return await self.delete_by_filter(models.Filter(must=conditions))
