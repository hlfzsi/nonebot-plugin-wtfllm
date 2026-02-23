import time
import uuid
from typing import List, TYPE_CHECKING, Tuple

from pydantic import BaseModel, Field

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

    @property
    def source_id(self) -> str:
        return f"knowledge-{self.storage_id}"

    @property
    def priority(self) -> int:
        return 1

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

    @property
    def source_id(self) -> str:
        return f"knowledge-block-{hash(id(self))}"

    @property
    def priority(self) -> int:
        return 1

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
