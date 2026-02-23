"""向量数据库 Repository 基类"""

__all__ = ["VectorRepository", "SearchResult"]

from abc import ABC
from typing import Generic, Literal, TypeVar, List, Optional, Tuple, Sequence
from pydantic import BaseModel

from qdrant_client import AsyncQdrantClient, models

from ..engine import get_qdrant_client
from ..models.base import VectorModel
from ...utils import APP_CONFIG

T = TypeVar("T", bound=VectorModel)
R = TypeVar("R")


class SearchResult(BaseModel, Generic[R]):
    """向量搜索结果封装

    Attributes:
        item: 匹配的模型实例
        score: 相似度分数
    """

    item: R
    score: float


class VectorRepository(ABC, Generic[T]):
    """向量数据库 Repository 基类

    提供:
        - CRUD 操作 (create, read, update, delete)
        - 向量搜索 (dense, sparse, hybrid)
        - 基于 Payload 的过滤查询

    Usage:
        class MyRepository(VectorRepository[MyPayload]):
            def __init__(self):
                super().__init__(MyPayload)
    """

    def __init__(self, model_class: type[T], collection_name: str) -> None:
        """初始化 Repository

        Args:
            model_class: 向量数据模型类
            collection_name: Qdrant 集合名称
        """
        self.model_class = model_class
        self.collection_name = collection_name

    @property
    def client(self) -> AsyncQdrantClient:
        """获取 Qdrant 异步客户端"""
        return get_qdrant_client()

    async def get_by_id(self, point_id: str) -> Optional[T]:
        """根据 ID 获取单个向量点

        Args:
            point_id: Qdrant Point ID

        Returns:
            模型实例，不存在则返回 None
        """
        results = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if results:
            return self.model_class.from_payload(results[0].payload or {})
        return None

    async def get_many_by_ids(self, point_ids: List[str]) -> List[T]:
        """批量获取多个向量点

        Args:
            point_ids: Point ID 列表

        Returns:

            模型实例列表（顺序可能与输入不同）
        """
        if not point_ids:
            return []
        results = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return [self.model_class.from_payload(p.payload or {}) for p in results]

    async def upsert(self, item: T, document: Optional[str] = None) -> str:
        """插入或更新单个向量点

        Args:
            item: 向量数据模型实例
            document: 用于生成向量的文本（默认使用 item.get_text_for_embedding()）

        Returns:
            插入的 point_id
        """
        text = document or item.get_text_for_embedding()
        vector_params = self.client.get_fastembed_vector_params()
        sparse_params = self.client.get_fastembed_sparse_vector_params()

        if not vector_params or not sparse_params:
            raise ValueError(
                f"Collection '{self.collection_name}' must have both dense and sparse vectors."
            )

        dense_name = list(vector_params.keys())[0]
        sparse_name = list(sparse_params.keys())[0]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=item.point_id,
                    payload=item.to_payload(),
                    vector={
                        dense_name: models.Document(
                            text=text, model=APP_CONFIG.embedding_model_name
                        ),
                        sparse_name: models.Document(
                            text=text, model=APP_CONFIG.sparse_model_name
                        ),
                    },
                )
            ],
        )
        return item.point_id

    async def upsert_many(
        self,
        items: List[T],
        documents: Optional[List[str]] = None,
    ) -> List[str]:
        """批量插入或更新向量点

        Args:
            items: 模型实例列表
            documents: 对应的文本列表（默认使用各项的 get_text_for_embedding()）

        Returns:
            插入的 point_id 列表
        """
        if not items:
            return []
        if documents and len(items) != len(documents):
            raise ValueError("Length of documents must match length of items.")

        vector_params = self.client.get_fastembed_vector_params()
        sparse_params = self.client.get_fastembed_sparse_vector_params()

        if not vector_params or not sparse_params:
            raise ValueError(
                f"Collection '{self.collection_name}' must have both dense and sparse vectors."
            )

        dense_name = list(vector_params.keys())[0]
        sparse_name = list(sparse_params.keys())[0]

        points = []

        for i, item in enumerate(items):
            text = documents[i] if documents else item.get_text_for_embedding()
            points.append(
                models.PointStruct(
                    id=item.point_id,
                    payload=item.to_payload(),
                    vector={
                        dense_name: models.Document(
                            text=text, model=APP_CONFIG.embedding_model_name
                        ),
                        sparse_name: models.Document(
                            text=text, model=APP_CONFIG.sparse_model_name
                        ),
                    },
                )
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return [item.point_id for item in items]

    async def delete_by_id(self, point_id: str) -> bool:
        """删除单个向量点

        Args:
            point_id: 要删除的 Point ID

        Returns:
            操作是否成功
        """
        result = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[point_id]),
        )
        return result.status == models.UpdateStatus.COMPLETED

    async def delete_many_by_ids(self, point_ids: Sequence[str]) -> bool:
        """批量删除向量点

        Args:
            point_ids: 要删除的 Point ID 列表

        Returns:
            操作是否成功
        """
        if not point_ids:
            return True
        result = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
        )
        return result.status == models.UpdateStatus.COMPLETED

    async def delete_by_filter(self, filter_: models.Filter) -> bool:
        """根据过滤条件删除向量点

        Args:
            filter_: Qdrant Filter 对象

        Returns:
            操作是否成功
        """
        result = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=filter_),
        )
        return result.status == models.UpdateStatus.COMPLETED

    async def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        dense_score_threshold: Optional[float] = None,
        sparse_score_threshold: Optional[float] = None,
        filter_: Optional[models.Filter] = None,
    ) -> List[SearchResult[T]]:
        """混合向量搜索（Dense + Sparse）

        Args:
            query: 查询文本
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值
            dense_score_threshold: 密集向量分数阈值
            sparse_score_threshold: 稀疏向量分数阈值
            filter_: 可选的过滤条件

        Returns:
            SearchResult 列表，包含模型实例和相似度分数
        """
        DEFAULT_SCORE_THRESHOLD = 0.50
        DEFAULT_DENSE_SCORE_THRESHOLD = 0.55
        DEFAULT_SPARSE_SCORE_THRESHOLD = 0.10

        LIMIT_MAGNIFICATION = 3

        score_threshold = (
            score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
        )
        dense_score_threshold = (
            dense_score_threshold
            if dense_score_threshold is not None
            else DEFAULT_DENSE_SCORE_THRESHOLD
        )
        sparse_score_threshold = (
            sparse_score_threshold
            if sparse_score_threshold is not None
            else DEFAULT_SPARSE_SCORE_THRESHOLD
        )

        vector_params = self.client.get_fastembed_vector_params()
        sparse_params = self.client.get_fastembed_sparse_vector_params()

        if not vector_params or not sparse_params:
            raise ValueError("Collection must have both dense and sparse vectors.")
        dense_name = list(vector_params.keys())[0]

        sparse_name = list(sparse_params.keys())[0]

        results = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text=query, model=APP_CONFIG.embedding_model_name
                    ),
                    using=dense_name,
                    limit=limit * LIMIT_MAGNIFICATION,
                    filter=filter_,
                    score_threshold=dense_score_threshold,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=query, model=APP_CONFIG.sparse_model_name
                    ),
                    using=sparse_name,
                    limit=limit * LIMIT_MAGNIFICATION,
                    filter=filter_,
                    score_threshold=sparse_score_threshold,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            SearchResult(
                item=self.model_class.from_payload(r.payload or {}),
                score=r.score,
            )
            for r in results.points
            if r.score >= score_threshold
        ]

    async def search_by_vector(
        self,
        vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_: Optional[models.Filter] = None,
    ) -> List[SearchResult[T]]:
        """使用预计算向量进行搜索

        Args:
            vector: 预计算的向量
            limit: 返回结果数量限制
            score_threshold: 相似度分数阈值
            filter_: 可选的过滤条件

        Returns:
            SearchResult 列表
        """
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_,
            with_payload=True,
        )
        return [
            SearchResult(
                item=self.model_class.from_payload(r.payload or {}),
                score=r.score,
            )
            for r in results.points
        ]

    async def scroll(
        self,
        filter_: Optional[models.Filter] = None,
        limit: int = 100,
        offset: Optional[str] = None,
        order_by: Optional[str] = None,
        order_type: Literal["asc", "desc"] = "asc",
    ) -> Tuple[List[T], Optional[str]]:
        """滚动查询（分页遍历）

        Args:
            filter_: 可选的过滤条件
            limit: 每页数量限制
            offset: 上一页返回的 offset（用于分页）
            order_by: 排序字段名
            order_type: 排序方向，"asc" 或 "desc"

        Returns:
            (结果列表, 下一页 offset)，offset 为 None 表示已到末页
        """
        order_by_param = None
        if order_by:
            if order_type == "asc":
                order_by_param = models.OrderBy(
                    key=order_by, direction=models.Direction.ASC
                )
            elif order_type == "desc":
                order_by_param = models.OrderBy(
                    key=order_by, direction=models.Direction.DESC
                )
            else:
                raise ValueError("order_type must be 'asc' or 'desc'")

        results, next_offset = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            order_by=order_by_param,
        )
        items = [self.model_class.from_payload(r.payload or {}) for r in results]
        next_offset_str = str(next_offset) if next_offset is not None else None
        return items, next_offset_str

    async def count(self, filter_: Optional[models.Filter] = None) -> int:
        """统计符合条件的向量点数量

        Args:
            filter_: 可选的过滤条件

        Returns:
            匹配的点数量
        """
        result = await self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_,
            exact=True,
        )
        return result.count

    @staticmethod
    def match_keyword(field: str, value: str) -> models.FieldCondition:
        """构建关键词精确匹配条件

        Args:
            field: Payload 字段名
            value: 要匹配的值

        Returns:
            FieldCondition 对象
        """
        return models.FieldCondition(
            key=field,
            match=models.MatchValue(value=value),
        )

    @staticmethod
    def match_any(field: str, values: List[str]) -> models.FieldCondition:
        """构建关键词任意匹配条件（IN 查询）

        Args:
            field: Payload 字段名
            values: 可匹配的值列表

        Returns:
            FieldCondition 对象
        """
        return models.FieldCondition(
            key=field,
            match=models.MatchAny(any=values),
        )

    @staticmethod
    def range_filter(
        field: str,
        gte: Optional[int] = None,
        lte: Optional[int] = None,
        gt: Optional[int] = None,
        lt: Optional[int] = None,
    ) -> models.FieldCondition:
        """构建范围过滤条件

        Args:
            field: Payload 字段名
            gte: 大于等于
            lte: 小于等于
            gt: 大于
            lt: 小于

        Returns:
            FieldCondition 对象
        """
        return models.FieldCondition(
            key=field,
            range=models.Range(gte=gte, lte=lte, gt=gt, lt=lt),
        )

    @staticmethod
    def is_null(field: str) -> models.IsNullCondition:
        """构建 NULL 检查条件

        Args:
            field: Payload 字段名

        Returns:
            IsNullCondition 对象
        """
        return models.IsNullCondition(is_null=models.PayloadField(key=field))
