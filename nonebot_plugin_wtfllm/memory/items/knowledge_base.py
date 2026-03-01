import time
import uuid
from typing import List, Self, TYPE_CHECKING, Tuple

from pydantic import BaseModel, Field, PrivateAttr

from .base import MemorySource

if TYPE_CHECKING:
    from ..context import LLMContext


class KnowledgeEntry(BaseModel, MemorySource):
    """知识库条目

    全局共享的知识，不绑定任何特定会话。
    由 agent 在对话中主动写入，所有会话可见。
    """

    storage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="知识条目唯一ID"
    )
    content: str = Field(..., description="知识内容文本")
    title: str = Field(..., description="知识条目简短标题/关键词")
    category: str = Field(default="general", description="知识分类")
    agent_id: str = Field(..., description="所属智能体ID")
    created_at: int = Field(
        default_factory=lambda: int(time.time()), description="创建时间戳"
    )
    updated_at: int = Field(
        default_factory=lambda: int(time.time()), description="最后更新时间戳"
    )
    source_session_type: str = Field(default="agent", description="来源会话类型")
    source_session_id: str | None = Field(default=None, description="来源会话ID")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    token_count: int = Field(default=0, description="缓存的 token 计数")
    _priority: float = PrivateAttr(default=0)

    @classmethod
    def create(
        cls,
        content: str,
        title: str,
        agent_id: str,
        category: str = "general",
        source_session_type: str = "agent",
        source_session_id: str | None = None,
        tags: List[str] | None = None,
        token_count: int = 0,
        priority: float = 0,
    ) -> Self:
        if not (0 <= priority < 1):
            raise ValueError("priority must be between 0 and 1(no inclusive)")

        instance = cls(
            content=content,
            title=title,
            category=category,
            agent_id=agent_id,
            source_session_type=source_session_type,
            source_session_id=source_session_id,
            tags=tags or [],
            token_count=token_count,
        )
        instance._priority = priority
        return instance

    @property
    def source_id(self) -> str:
        return f"knowledge-{self.storage_id}"

    @property
    def priority(self) -> float:
        return 3 + self._priority

    @property
    def sort_key(self) -> Tuple[int, str]:
        return (self.updated_at, self.storage_id)

    def register_all_alias(self, ctx: "LLMContext") -> None:
        pass

    def to_llm_context(self, ctx: "LLMContext") -> str:
        ref = ctx.ref_provider.next_knowledge_ref(self)
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"[{ref}] 【{self.title}】{tag_str} {self.content}"

    def __hash__(self) -> int:
        return hash(self.source_id)


class KnowledgeBlock(BaseModel, MemorySource):
    """知识库块，用于将一组知识条目注入到 LLM 上下文中"""

    entries: List[KnowledgeEntry] = Field(default_factory=list)
    prefix: str | None = Field(default=None)
    suffix: str | None = Field(default=None)
    _priority: float = PrivateAttr(default=0)

    @classmethod
    def create(
        cls,
        entries: List[KnowledgeEntry],
        prefix: str | None = None,
        suffix: str | None = None,
        priority: float = 0,
    ) -> Self:
        if not (0 <= priority < 1):
            raise ValueError("priority must be between 0 and 1(no inclusive)")

        instance = cls(entries=entries, prefix=prefix, suffix=suffix)
        instance._priority = priority
        return instance

    @property
    def source_id(self) -> str:
        return f"knowledge-block-{hash(id(self))}"

    @property
    def priority(self) -> float:
        return 3 + self._priority

    @property
    def sort_key(self) -> Tuple[int, str]:
        if not self.entries:
            return (0, self.source_id)
        latest = max(e.updated_at for e in self.entries)
        return (latest, self.source_id)

    def register_all_alias(self, ctx: "LLMContext") -> None:
        pass

    def to_llm_context(self, ctx: "LLMContext") -> str:
        if not self.entries:
            return ""
        self.entries.sort(key=lambda e: e.updated_at)
        lines = []
        if self.prefix:
            lines.append(self.prefix)
        for entry in self.entries:
            lines.append(entry.to_llm_context(ctx))
        if self.suffix:
            lines.append(self.suffix)
        return "\n".join(lines)

    def __hash__(self) -> int:
        return hash(id(self))
