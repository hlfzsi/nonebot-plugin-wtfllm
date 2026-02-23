"""CoreMemory 的 Qdrant Payload 模型"""

__all__ = ["CoreMemoryPayload"]

from typing import ClassVar, List, Self, TYPE_CHECKING

from pydantic import Field
from qdrant_client import models

from .base import VectorModel

if TYPE_CHECKING:
    from ...memory.items.core_memory import CoreMemory


class CoreMemoryPayload(VectorModel):
    """CoreMemory 的 Qdrant Payload 模型

    Attributes:
        storage_id: 核心记忆唯一ID（作为 Qdrant Point ID）
        content: 记忆内容文本（用于向量嵌入）
        group_id: 所属群组ID
        user_id: 所属用户ID
        agent_id: 所属智能体ID
        created_at: 创建时间戳
        updated_at: 最后更新时间戳
        source: 来源标记
        token_count: 缓存的 token 计数
        related_entities: 相关的实体ID列表
    """

    collection_name: ClassVar[str] = "wtfllm_core_memory"
    indexes: ClassVar[dict[str, models.PayloadSchemaType]] = {
        "agent_id": models.PayloadSchemaType.KEYWORD,
        "group_id": models.PayloadSchemaType.KEYWORD,
        "user_id": models.PayloadSchemaType.KEYWORD,
        "created_at": models.PayloadSchemaType.INTEGER,
        "updated_at": models.PayloadSchemaType.INTEGER,
        "source": models.PayloadSchemaType.KEYWORD,
        "related_entities": models.PayloadSchemaType.KEYWORD,
    }
    point_id_field: ClassVar[str] = "storage_id"

    storage_id: str = Field(..., description="核心记忆唯一ID")
    content: str = Field(..., description="记忆内容文本")
    group_id: str | None = Field(default=None, description="所属群组ID")
    user_id: str | None = Field(default=None, description="所属用户ID")
    agent_id: str = Field(..., description="所属智能体ID")
    created_at: int = Field(..., description="创建时间戳")
    updated_at: int = Field(..., description="最后更新时间戳")
    source: str = Field(default="agent", description="来源标记")
    token_count: int = Field(default=0, description="缓存的 token 计数")
    related_entities: List[str] = Field(
        default_factory=list, description="相关的实体ID列表"
    )

    @property
    def point_id(self) -> str:
        return self.storage_id

    def get_text_for_embedding(self) -> str:
        return self.content

    @classmethod
    def from_core_memory(cls, memory: "CoreMemory") -> Self:
        return cls.model_validate(memory.model_dump())

    def to_core_memory(self) -> "CoreMemory":
        from ...memory.items.core_memory import CoreMemory

        return CoreMemory.model_validate(self.model_dump())
