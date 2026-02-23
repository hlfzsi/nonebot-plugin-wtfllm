import re
import time
import uuid
from typing import List, TYPE_CHECKING, Tuple

from pydantic import BaseModel, Field

from .base import MemorySource
from .._types import ID_PATTERN

if TYPE_CHECKING:
    from ..context import LLMContext


class CoreMemory(BaseModel, MemorySource):
    """核心记忆条目

    由 CHAT_AGENT 在对话过程中主动写入的持久化记忆。
    每个会话（群聊/私聊）独立维护。
    """

    storage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="核心记忆唯一ID"
    )
    content: str = Field(..., description="记忆内容文本")
    group_id: str | None = Field(default=None, description="所属群组（私聊为 None）")
    user_id: str | None = Field(default=None, description="所属用户（群聊为 None）")
    agent_id: str = Field(..., description="所属 agent")
    created_at: int = Field(
        default_factory=lambda: int(time.time()), description="创建时间戳"
    )
    updated_at: int = Field(
        default_factory=lambda: int(time.time()), description="最后更新时间戳"
    )
    source: str = Field(default="agent", description="来源标记：agent | compression")
    token_count: int = Field(default=0, description="缓存的 token 计数")
    related_entities: List[str] = Field(
        default_factory=list, description="相关的实体ID列表"
    )

    @property
    def source_id(self) -> str:
        return f"core-memory-{self.storage_id}"

    @property
    def priority(self) -> int:
        return 2

    @property
    def sort_key(self) -> Tuple[int, str]:
        return (self.updated_at, self.storage_id)

    def register_all_alias(self, ctx: "LLMContext") -> None:
        if self.group_id:
            ctx.alias_provider.register_group(self.group_id)
        if self.user_id:
            ctx.alias_provider.register_user(self.user_id)
        for entity_id in self.related_entities:
            ctx.alias_provider.register_user(entity_id)

        registered = set(self.related_entities)
        if self.group_id:
            registered.add(self.group_id)
        if self.user_id:
            registered.add(self.user_id)
        for match in ID_PATTERN.finditer(self.content):
            entity_id = match.group(1)
            if entity_id not in registered:
                ctx.alias_provider.register_user(entity_id)
                registered.add(entity_id)

    def normalize_placeholders(self, ctx: "LLMContext") -> None:
        """将 content 中的占位符别名替换为实体 ID

        应在存储前调用，将 LLM 生成的占位符（如 {{User_1}} 或 {{小明}}）转换为 {{实体ID}}
        """

        def replace_placeholder(match: re.Match) -> str:
            full_match = match.group(0)
            alias = match.group(1)
            entity_id = ctx.alias_provider.resolve_alias(alias)
            if entity_id and entity_id not in self.related_entities:
                self.related_entities.append(entity_id)
            return f"{{{{{entity_id}}}}}" if entity_id else full_match

        self.content = ID_PATTERN.sub(replace_placeholder, self.content)

    def _render_content(self, ctx: "LLMContext") -> str:
        """将 content 中的实体 ID 占位符替换为当前上下文的别名"""

        def replace_entity_id(match: re.Match) -> str:
            entity_id = match.group(1)
            alias = ctx.alias_provider.get_alias(entity_id)
            return f"{{{{{alias}}}}}" if alias else match.group(0)

        return ID_PATTERN.sub(replace_entity_id, self.content)

    def _render_session_tag(self, ctx: "LLMContext") -> str:
        """渲染会话来源标签，如 (Group_1) 或 (User_2 私聊)"""
        if self.group_id:
            alias = ctx.alias_provider.get_alias(self.group_id)
            return f"({alias})" if alias else f"({self.group_id})"
        if self.user_id:
            alias = ctx.alias_provider.get_alias(self.user_id)
            return f"({alias} 私聊)" if alias else f"({self.user_id} 私聊)"
        return ""

    def to_llm_context(self, ctx: "LLMContext") -> str:
        ref = ctx.ref_provider.next_core_memory_ref(self)
        session_tag = self._render_session_tag(ctx)
        content = self._render_content(ctx)
        if session_tag:
            return f"[{ref}] {session_tag} {content}"
        return f"[{ref}] {content}"

    def __hash__(self) -> int:
        return hash(self.source_id)


class CoreMemoryBlock(BaseModel, MemorySource):
    """核心记忆块，用于将一组核心记忆注入到 LLM 上下文中"""

    memories: List[CoreMemory] = Field(default_factory=list)
    prefix: str | None = Field(default=None)
    suffix: str | None = Field(default=None)

    @property
    def source_id(self) -> str:
        return f"core-memory-block-{hash(id(self))}"

    @property
    def priority(self) -> int:
        return 2

    @property
    def sort_key(self) -> Tuple[int, str]:
        if not self.memories:
            return (0, self.source_id)
        latest = max(m.updated_at for m in self.memories)
        return (latest, self.source_id)

    def register_all_alias(self, ctx: "LLMContext") -> None:
        for memory in self.memories:
            memory.register_all_alias(ctx)

    def to_llm_context(self, ctx: "LLMContext") -> str:
        if not self.memories:
            return ""
        self.memories.sort(key=lambda m: m.updated_at)
        lines = []
        if self.prefix:
            lines.append(self.prefix)
        for memory in self.memories:
            lines.append(memory.to_llm_context(ctx))
        if self.suffix:
            lines.append(self.suffix)
        return "\n".join(lines)

    def __hash__(self) -> int:
        return hash(id(self))
