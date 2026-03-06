"""话题归档 Repository"""

__all__ = ["TopicArchiveRepository"]

import math
import time
from typing import List, Optional

from qdrant_client import models

from ..models.topic_archive import TopicArchivePayload
from .base import SearchResult, VectorRepository

# 指数时间衰减系数 λ：adjusted = score × e^(-λ × age_days)
# λ=0.01 对应半衰期 ≈ ln2/0.01 ≈ 69 天
_DECAY_LAMBDA: float = 0.01

# 多拉 N 倍数据，衰减重排后截取
_OVERFETCH_FACTOR: int = 3


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
        """按会话搜索归档话题，结果经指数时间衰减重排。"""
        conditions: list[models.Condition] = [self.match_keyword("agent_id", agent_id)]
        if group_id is not None:
            conditions.append(self.match_keyword("group_id", group_id))
        if user_id is not None:
            conditions.append(self.match_keyword("user_id", user_id))

        filter_ = models.Filter(must=conditions)

        raw = await self.search(
            query=query,
            limit=limit * _OVERFETCH_FACTOR,
            score_threshold=score_threshold,
            filter_=filter_,
        )

        if not raw:
            return []

        now = time.time()
        reranked: list[tuple[float, SearchResult[TopicArchivePayload]]] = []
        for r in raw:
            age_days = max((now - r.item.created_at) / 86400.0, 0.0)
            adjusted = r.score * math.exp(-_DECAY_LAMBDA * age_days)
            reranked.append((adjusted, r))

        reranked.sort(key=lambda p: p[0], reverse=True)

        return [
            SearchResult[TopicArchivePayload](item=r.item, score=adj)
            for adj, r in reranked[:limit]
        ]

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
