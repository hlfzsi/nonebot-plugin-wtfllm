from typing import TYPE_CHECKING, Self, Union

import orjson
from pydantic import TypeAdapter
from sqlmodel import SQLModel, Field

if TYPE_CHECKING:
    from ...memory import MemoryItemUnion
    from ...memory.items.base_items import PrivateMemoryItem, GroupMemoryItem


class MemoryItemTable(SQLModel, table=True):
    """记忆项存储表"""

    message_id: str = Field(..., primary_key=True, description="记忆对应消息ID")
    related_message_id: str | None = Field(
        default=None, description="记忆相关联的消息ID"
    )

    sender: str = Field(..., description="记忆发送者")
    content: str = Field(..., description="记忆内容(JSON)")
    created_at: int = Field(..., description="记忆时间戳")
    agent_id: str = Field(..., index=True, description="记忆所属智能体ID")
    memory_type: str = Field(..., description="记忆类型: private | group")
    group_id: str | None = Field(
        default=None, index=True, description="群组ID(群组记忆时必填)"
    )
    user_id: str | None = Field(
        default=None, index=True, description="所属用户ID(私有记忆时必填)"
    )

    @classmethod
    def from_MemoryItem(cls, item: "MemoryItemUnion") -> Self:
        """从 MemoryItem 实例创建 MemoryItemTable 实例"""
        from ...memory.items.base_items import PrivateMemoryItem, GroupMemoryItem

        if isinstance(item, PrivateMemoryItem):
            memory_type = "private"
            group_id = None
            user_id = item.user_id
        elif isinstance(item, GroupMemoryItem):
            memory_type = "group"
            group_id = item.group_id
            user_id = None
        else:
            raise TypeError(f"Unsupported MemoryItem type: {type(item)}")

        return cls(
            message_id=item.message_id,
            related_message_id=item.related_message_id,
            sender=item.sender,
            content=item.content.model_dump_json(),
            created_at=item.created_at,
            agent_id=item.agent_id,
            memory_type=memory_type,
            group_id=group_id,
            user_id=user_id,
        )

    def to_MemoryItem(self) -> "MemoryItemUnion":
        """将 MemoryItemTable 转换为 MemoryItem 实例（使用 pydantic 多态反序列化）"""
        from ...memory.items.base_items import MemoryItemUnion

        _adapter: TypeAdapter[Union["PrivateMemoryItem", "GroupMemoryItem"]] = (
            TypeAdapter(MemoryItemUnion)
        )

        data = {
            "message_id": self.message_id,
            "related_message_id": self.related_message_id,
            "sender": self.sender,
            "content": orjson.loads(self.content),
            "created_at": self.created_at,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
        }
        if self.group_id:
            data["group_id"] = self.group_id
        if self.user_id:
            data["user_id"] = self.user_id

        return _adapter.validate_python(data)
