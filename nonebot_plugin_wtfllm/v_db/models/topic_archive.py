"""话题归档 Qdrant Payload 模型"""

__all__ = ["TopicArchivePayload"]

from typing import ClassVar, Dict

from pydantic import Field
from qdrant_client import models

from .base import VectorModel


class TopicArchivePayload(VectorModel):
    collection_name: ClassVar[str] = "wtfllm_topic_archive"
    indexes: ClassVar[Dict[str, models.PayloadSchemaType]] = {
        "agent_id": models.PayloadSchemaType.KEYWORD,
        "group_id": models.PayloadSchemaType.KEYWORD,
        "user_id": models.PayloadSchemaType.KEYWORD,
        "created_at": models.PayloadSchemaType.INTEGER,
    }
    point_id_field: ClassVar[str] = "archive_id"

    archive_id: str = Field(..., description="归档唯一ID")
    agent_id: str = Field(..., description="所属智能体ID")
    group_id: str | None = Field(default=None, description="所属群组ID")
    user_id: str | None = Field(default=None, description="所属用户ID")
    representative_message_ids: list[str] = Field(
        default_factory=list, description="MMR 选出的代表消息ID"
    )
    message_count: int = Field(default=0, description="簇原始消息总数")
    created_at: int = Field(..., description="归档时间戳")

    @property
    def point_id(self) -> str:
        return self.archive_id

    def get_text_for_embedding(self) -> str:
        return ""
