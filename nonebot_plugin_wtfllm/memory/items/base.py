import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Tuple

from pydantic import BaseModel, Field

from ..content import Message, MentionSegment, ForwardSegment

if TYPE_CHECKING:
    from ..context import LLMContext

SCOPE = Literal["private", "group"]


class MemorySource(ABC):
    """记忆源抽象基类"""

    @property
    def role(self) -> str | None:
        """记忆源角色标识, 用于按用途检索特定的记忆源"""
        return None

    @property
    @abstractmethod
    def source_id(self) -> str:
        """唯一标识符"""
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        """优先级, 数值越大优先级越高, 用于排序"""
        ...

    @property
    @abstractmethod
    def sort_key(self) -> Tuple[float, str]:
        """
        用于排序

        Returns:
            Tuple[第一条件, 第二条件]
        """
        ...

    @abstractmethod
    def register_all_alias(self, ctx: "LLMContext") -> None:
        """注册所有内部记忆/实体的别名"""
        ...

    @abstractmethod
    def to_llm_context(self, ctx: "LLMContext") -> str:
        """生成 LLM 上下文"""
        ...


class MemoryItem(BaseModel, ABC):
    """记忆项抽象基类"""

    message_id: str = Field(..., description="记忆对应消息ID")
    related_message_id: str | None = Field(
        default=None, description="记忆相关联的消息ID"
    )

    sender: str = Field(..., description="记忆发送者")
    content: Message = Field(default_factory=Message.create, description="记忆内容")
    created_at: int = Field(
        default_factory=lambda: int(time.time()), description="记忆时间戳"
    )
    agent_id: str = Field(..., description="记忆所属智能体ID")

    @property
    def is_from_agent(self) -> bool:
        """判断消息是否来自 Agent"""
        return self.sender == self.agent_id

    @abstractmethod
    def to_llm_context(self, ctx: "LLMContext") -> str:
        """使用指定的引用号生成 LLM 格式"""
        ...

    def get_plain_text(self) -> str:
        """获取纯文本内容"""
        return self.content.get_plain_text()

    def register_entities(self, ctx: "LLMContext") -> None:
        """注册实体别名（含嵌套 ForwardSegment 内的实体）"""
        ctx.alias_provider.register_agent(self.agent_id)
        if not self.is_from_agent:
            ctx.alias_provider.register_user(self.sender)

        self._register_message_entities(ctx, self.content)

    def _register_message_entities(self, ctx: "LLMContext", message: Message) -> None:
        """递归注册消息中的实体别名"""
        for seg in message.segments:
            if isinstance(seg, MentionSegment) and seg.user_id:
                ctx.alias_provider.register_user(seg.user_id)
            elif isinstance(seg, ForwardSegment):
                for node in seg.children:
                    ctx.alias_provider.register_user(node.sender)
                    if node.group_id:
                        ctx.alias_provider.register_group(node.group_id)
                    self._register_message_entities(ctx, node.content)

    def __hash__(self) -> int:
        return hash(self.message_id)
