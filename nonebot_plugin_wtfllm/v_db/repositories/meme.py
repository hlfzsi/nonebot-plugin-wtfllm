"""Meme Repository - 图片表情包存储和检索"""

__all__ = ["MemeRepository"]

from typing import List, Optional, Tuple

from qdrant_client import models

from .base import VectorRepository, SearchResult
from ..models.meme import MemePayload


class MemeRepository(VectorRepository[MemePayload]):
    """
    Meme Repository
    """

    def __init__(self) -> None:
        super().__init__(MemePayload, MemePayload.collection_name)

    async def save_meme(self, payload: MemePayload) -> str:
        """保存 Meme

        Args:
            payload: MemePayload 实例

        Returns:
            storage_id
        """
        return await self.upsert(payload)

    async def save_many_memes(self, payloads: List[MemePayload]) -> List[str]:
        """批量保存 Meme

        Args:
            payloads: MemePayload 实例列表

        Returns:
            storage_id 列表
        """
        return await self.upsert_many(payloads)

    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_: Optional[models.Filter] = None,
    ) -> List[SearchResult[MemePayload]]:
        """通过文本描述搜索 Meme

        Args:
            query: 搜索文本
            limit: 返回结果数量
            score_threshold: 相似度阈值
            filter_: 可选的过滤条件

        Returns:
            SearchResult 列表
        """
        return await self.search(query, limit, score_threshold, filter_=filter_)

    async def list_by_uploader(
        self,
        uploader_id: str,
        limit: int = 100,
        offset: Optional[str] = None,
        order_by: str = "created_at",
    ) -> Tuple[List[MemePayload], Optional[str]]:
        """获取指定用户上传的 Meme

        Args:
            uploader_id: 用户ID
            limit: 返回数量
            offset: 分页偏移
            order_by: 排序字段

        Returns:
            (Meme 列表, 下一页 offset)
        """
        filter_ = models.Filter(must=[self.match_keyword("uploader_id", uploader_id)])
        return await self.scroll(
            filter_=filter_,
            limit=limit,
            offset=offset,
            order_by=order_by,
        )

    async def count_by_uploader(self, uploader_id: str) -> int:
        """统计指定用户上传的 Meme 数量"""
        filter_ = models.Filter(must=[self.match_keyword("uploader_id", uploader_id)])
        return await self.count(filter_)

    async def delete_by_uploader(self, uploader_id: str) -> bool:
        """删除指定用户的所有 Meme（仅删除向量数据库记录）"""
        filter_ = models.Filter(must=[self.match_keyword("uploader_id", uploader_id)])
        return await self.delete_by_filter(filter_)

    async def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 100,
        offset: Optional[str] = None,
    ) -> Tuple[List[MemePayload], Optional[str]]:
        """按标签搜索 Meme

        Args:
            tags: 标签列表
            match_all: True 表示必须匹配所有标签，False 表示匹配任意一个
            limit: 返回数量
            offset: 分页偏移

        Returns:
            (Meme 列表, 下一页 offset)
        """
        normalized_tags = [t.strip().lower() for t in tags if t.strip()]

        if match_all:
            conditions: List[models.Condition] = [
                self.match_keyword("tags", tag) for tag in normalized_tags
            ]
            filter_ = models.Filter(must=conditions)
        else:
            filter_ = models.Filter(must=[self.match_any("tags", normalized_tags)])

        return await self.scroll(filter_=filter_, limit=limit, offset=offset)

    async def list_by_tag(self, tag: str, limit: int = 100) -> List[MemePayload]:
        """获取包含指定标签的 Meme"""
        memes, _ = await self.search_by_tags([tag], limit=limit)
        return memes

    async def get_recent(
        self,
        since_timestamp: int | None = None,
        limit: int = 100,
        uploader_id: Optional[str] = None,
    ) -> List[MemePayload]:
        """获取指定时间戳之后的 Meme

        Args:
            since_timestamp: 起始时间戳
            limit: 返回数量
            uploader_id: 可选的上传者过滤

        Returns:
            Meme 列表
        """
        conditions: List[models.Condition] = []
        if since_timestamp is not None:
            conditions.append(self.range_filter("created_at", gte=since_timestamp))
        if uploader_id:
            conditions.append(self.match_keyword("uploader_id", uploader_id))

        filter_ = models.Filter(must=conditions)
        memes, _ = await self.scroll(
            filter_=filter_, limit=limit, order_by="created_at"
        )
        return memes

    async def delete_before_timestamp(self, timestamp: int) -> bool:
        """删除指定时间戳之前的 Meme（仅删除向量数据库记录）"""
        filter_ = models.Filter(must=[self.range_filter("created_at", lt=timestamp)])
        return await self.delete_by_filter(filter_)

    async def get_by_message_id(self, raw_message_id: str) -> Optional[MemePayload]:
        """根据原始消息ID获取 Meme"""
        filter_ = models.Filter(
            must=[self.match_keyword("raw_message_id", raw_message_id)]
        )
        memes, _ = await self.scroll(filter_=filter_, limit=1)
        return memes[0] if memes else None

    async def search_by_text_with_tags(
        self,
        query: str,
        tags: List[str],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult[MemePayload]]:
        """通过文本搜索，同时过滤标签"""
        normalized_tags = [t.strip().lower() for t in tags if t.strip()]
        filter_ = models.Filter(must=[self.match_any("tags", normalized_tags)])
        return await self.search_by_text(query, limit, score_threshold, filter_)

    async def search_by_text_by_uploader(
        self,
        query: str,
        uploader_id: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult[MemePayload]]:
        """在指定用户的 Meme 中搜索"""
        filter_ = models.Filter(must=[self.match_keyword("uploader_id", uploader_id)])
        return await self.search_by_text(query, limit, score_threshold, filter_)

    async def get_meme_by_id(self, storage_id: str) -> Optional[MemePayload]:
        """根据 storage_id 获取 Meme"""
        return await self.get_by_id(storage_id)

    async def exists(self, storage_id: str) -> bool:
        """检查 Meme 是否存在"""
        return (await self.get_by_id(storage_id)) is not None

    async def delete_meme_by_id(self, storage_id: str) -> bool:
        """删除 Meme 及其关联文件

        Args:
            storage_id: Meme 的 storage_id

        Returns:
            是否成功
        """
        payload = await self.get_by_id(storage_id)
        if payload:
            await payload.delete_file()
        return await self.delete_by_id(storage_id)

    async def delete_memes_by_uploader(self, uploader_id: str) -> bool:
        """删除指定用户的所有 Meme 及其文件

        Args:
            uploader_id: 用户ID

        Returns:
            是否成功
        """
        memes, _ = await self.list_by_uploader(uploader_id, limit=10000)
        for meme in memes:
            await meme.delete_file()
        return await self.delete_by_uploader(uploader_id)
