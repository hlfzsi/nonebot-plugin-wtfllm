from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import Field, Tag

from .base import MemoryItem

if TYPE_CHECKING:
    from ..context import LLMContext


class PrivateMemoryItem(MemoryItem):
    """私有记忆项"""

    memory_type: Literal["private"] = Field(default="private", description="记忆类型")

    user_id: str = Field(..., description="属于用户的用户ID")

    def to_llm_context(self, ctx: "LLMContext") -> str:
        sender_alias = ctx.alias_provider.get_alias(self.sender) or self.sender
        ref = ctx.ref_provider.next_memory_ref(self)
        content_str = self.content.to_llm_context(ctx, self.message_id, ref)
        if self.related_message_id:
            return f"[{ref}]  (in reply to [{ctx.ref_provider.get_ref_by_item_id(self.related_message_id) or '未知消息'}]) {sender_alias}: {content_str}"
        return f"[{ref}] {sender_alias}: {content_str}"


class GroupMemoryItem(MemoryItem):
    """群组记忆项"""

    memory_type: Literal["group"] = Field(default="group", description="记忆类型")

    group_id: str = Field(..., description="群组ID")

    def register_entities(self, ctx: "LLMContext") -> None:
        super().register_entities(ctx)
        ctx.alias_provider.register_group(self.group_id)

    def to_llm_context(self, ctx: "LLMContext") -> str:
        sender_alias = ctx.alias_provider.get_alias(self.sender) or self.sender
        ref = ctx.ref_provider.next_memory_ref(self)
        content_str = self.content.to_llm_context(ctx, self.message_id, ref)
        if self.related_message_id:
            return f"[{ref}] (in reply to [{ctx.ref_provider.get_ref_by_item_id(self.related_message_id) or '未知消息'}]) {sender_alias}: {content_str}"
        return f"[{ref}] {sender_alias}: {content_str}"


MemoryItemUnion = Annotated[
    Union[
        Annotated[PrivateMemoryItem, Tag("private")],
        Annotated[GroupMemoryItem, Tag("group")],
    ],
    Field(discriminator="memory_type"),
]
