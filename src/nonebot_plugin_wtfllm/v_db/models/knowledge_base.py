"""知识库的 Qdrant Payload 模型"""

__all__ = ["KnowledgeBasePayload"]

from typing import ClassVar, List, Self, TYPE_CHECKING

from pydantic import Field
from qdrant_client import models

from .base import VectorModel

if TYPE_CHECKING:
    from ...memory.items.knowledge_base import KnowledgeEntry


class KnowledgeBasePayload(VectorModel):
    """知识库的 Qdrant Payload 模型

    全局共享的知识条目，不绑定特定会话，仅按 agent_id 隔离。

    Attributes:
        storage_id: 知识条目唯一ID（作为 Qdrant Point ID）
        content: 知识内容文本（用于向量嵌入）
        title: 知识条目简短标题/关键词
        category: 知识分类
        agent_id: 所属智能体ID
        created_at: 创建时间戳
        updated_at: 最后更新时间戳
        source_session_type: 来源会话类型（仅记录，不影响可见性）
        source_session_id: 来源会话ID（仅记录，不影响可见性）
        tags: 标签列表
        token_count: 缓存的 token 计数
    """

    collection_name: ClassVar[str] = "wtfllm_knowledge_base"
    indexes: ClassVar[dict[str, models.PayloadSchemaType]] = {
        "agent_id": models.PayloadSchemaType.KEYWORD,
        "category": models.PayloadSchemaType.KEYWORD,
        "tags": models.PayloadSchemaType.KEYWORD,
        "created_at": models.PayloadSchemaType.INTEGER,
        "updated_at": models.PayloadSchemaType.INTEGER,
    }
    point_id_field: ClassVar[str] = "storage_id"

    storage_id: str = Field(..., description="知识条目唯一ID")
    content: str = Field(..., description="知识内容文本")
    title: str = Field(..., description="知识条目简短标题/关键词")
    category: str = Field(default="general", description="知识分类")
    agent_id: str = Field(..., description="所属智能体ID")
    created_at: int = Field(..., description="创建时间戳")
    updated_at: int = Field(..., description="最后更新时间戳")
    source_session_type: str = Field(default="agent", description="来源会话类型")
    source_session_id: str | None = Field(default=None, description="来源会话ID")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    token_count: int = Field(default=0, description="缓存的 token 计数")

    @property
    def point_id(self) -> str:
        return self.storage_id

    def get_text_for_embedding(self) -> str:
        return f"{self.title}: {self.content}"

    @classmethod
    def from_knowledge_entry(cls, entry: "KnowledgeEntry") -> Self:
        return cls.model_validate(entry.model_dump())

    def to_knowledge_entry(self) -> "KnowledgeEntry":
        from ...memory.items.knowledge_base import KnowledgeEntry

        return KnowledgeEntry.model_validate(self.model_dump())
