"""向量数据模型基类"""

__all__ = ["VectorModel"]

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Self

from qdrant_client import models
from pydantic import BaseModel

from ..engine import get_qdrant_client
from ...utils import logger


class VectorModel(BaseModel, ABC):
    """向量数据模型基类

    所有存储到 Qdrant 的数据模型都应继承此类。
    提供 Pydantic 验证 + Qdrant Point 转换的标准接口。
    """

    collection_name: ClassVar[str]  # 指定所属的 Qdrant 集合名称
    indexes: ClassVar[
        Dict[str, models.PayloadSchemaType]
    ]  # 指定所属集合的 Payload 索引定义
    point_id_field: ClassVar[str] = "id"  # 指定用作 Point ID 的字段名

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return

        required_class_attrs = ["collection_name", "indexes"]

        for attr_name in required_class_attrs:
            attr_value = getattr(cls, attr_name, None)

            if attr_value is None:
                raise TypeError(
                    f"Class {cls.__name__} must define a class attribute '{attr_name}'."
                )

            if isinstance(attr_value, str) and not attr_value.strip():
                raise TypeError(
                    f"Class {cls.__name__} must provide a non-empty value for '{attr_name}'."
                )

    @property
    @abstractmethod
    def point_id(self) -> str:
        """返回 Qdrant Point 的唯一标识符"""
        ...

    @abstractmethod
    def get_text_for_embedding(self) -> str:
        """返回用于生成向量嵌入的文本内容"""
        ...

    def to_payload(self) -> Dict[str, Any]:
        """将模型转换为 Qdrant Payload"""
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> Self:
        """从 Qdrant Payload 重建模型实例

        Args:
            payload: Qdrant Point 的 payload 字典

        Returns:
            模型实例
        """
        return cls.model_validate(payload)

    @classmethod
    async def init_collection(
        cls,
        collection_name: str,
        payload_indexes: Dict[str, models.PayloadSchemaType] | None,
    ) -> None:
        """初始化 Qdrant 集合

        Args:
            collection_name: 集合名称
            payload_indexes: Payload 索引定义
        """
        logger.info(f"正在检查集合 '{collection_name}' 状态...")
        client = get_qdrant_client()

        try:
            if await client.collection_exists(collection_name):
                logger.info(f"集合 '{collection_name}' 已存在。")
            else:
                vectors_config = client.get_fastembed_vector_params(
                    on_disk=True,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                )

                sparse_config = client.get_fastembed_sparse_vector_params(
                    on_disk=True, modifier=models.Modifier.IDF
                )

                await client.create_collection(
                    collection_name=collection_name,
                    on_disk_payload=True,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_config,
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000
                    ),
                )
                logger.success(f"集合 '{collection_name}' 创建成功。")

            if not payload_indexes:
                return

            collection_info = await client.get_collection(collection_name)
            existing_schema = collection_info.payload_schema or {}

            tasks = []
            for field, schema_type in payload_indexes.items():
                if field not in existing_schema:
                    tasks.append(
                        client.create_payload_index(
                            collection_name=collection_name,
                            field_name=field,
                            field_schema=schema_type,
                        )
                    )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.success("Payload 索引初始化完成。")

        except Exception as e:
            logger.error(f"索引初始化失败: {e}", exc_info=True)
