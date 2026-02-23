from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Self,
    Type,
    Iterable,
    TypeVar,
    overload,
)

from .context import LLMContext
from .utils import DirtyStateMarker

if TYPE_CHECKING:
    from .items import MemorySource, MemoryItem
    from .items.core_memory import CoreMemory
    from .items.knowledge_base import KnowledgeEntry
    from .content import BaseSegment

T = TypeVar("T", bound="BaseSegment")


class MemoryContextBuilder:
    """
    协调记忆项和摘要的注册与引用, 最终生成可读文本。
    """

    def __init__(
        self,
        suffix_prompt: str | None = None,
        prefix_prompt: str | None = None,
        ctx: LLMContext | None = None,
        sources: List["MemorySource"] | None = None,
        agent_id: List[str] | str | None = None,
        custom_ref: Dict[str, str] | None = None,
    ):
        """
        Args:
            sources (List["MemorySource"] | None, optional): 记忆源列表.
            agent_id (List[str] | str | None, optional): 智能体ID列表或单个ID.
            custom_ref (Dict[str, str] | None, optional): 自定义实体别名映射. [实体ID, 别名].
        """
        self.suffix_prompt = suffix_prompt
        self.prefix_prompt = prefix_prompt
        self.ctx = ctx or LLMContext.create()
        self._sources: List["MemorySource"] = sources or []
        self.agent_ids = [agent_id] if isinstance(agent_id, str) else agent_id
        self._dirty = True

        if custom_ref:
            self.ctx.alias_provider.update_aliases(custom_ref)

        if self.agent_ids:
            for aid in self.agent_ids:
                self.ctx.alias_provider.register_agent(aid)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def agent_refs(self) -> List[str] | None:
        if not self.agent_ids:
            return None
        _ids = [self.ctx.alias_provider.get_alias(a) for a in self.agent_ids]
        return [i for i in _ids if i is not None]

    @DirtyStateMarker.marks_dirty
    def add(self, source: "MemorySource") -> Self:
        """添加记忆源"""
        self._sources.append(source)
        return self

    def extend(self, sources: Iterable["MemorySource"]) -> Self:
        """添加多个记忆源"""
        self._sources.extend(sources)
        return self

    def index(self, source: "MemorySource") -> int:
        """获取记忆源的索引位置"""
        return self._sources.index(source)

    def remove(self, source: "MemorySource") -> Self:
        """移除记忆源"""
        self._sources.remove(source)
        return self

    @DirtyStateMarker.needs_clean
    def to_prompt(self, sep: str = "\n") -> str:
        """生成最终的记忆文本"""
        lines: List[str] = []
        sorted_sources = sorted(
            self._sources, key=lambda x: (-x.priority, x.sort_key[0], x.sort_key[1])
        )
        lines.extend([source.to_llm_context(self.ctx) for source in sorted_sources])
        lines = [line for line in lines if line.strip()]
        if self.suffix_prompt:
            lines.append(self.suffix_prompt)
        if self.prefix_prompt:
            lines.insert(0, self.prefix_prompt)
        return sep.join(lines)

    @DirtyStateMarker.marks_clean
    def _ensure_clean(self) -> None:
        """注册所有记忆源中的实体别名"""
        for source in self._sources:
            source.register_all_alias(self.ctx)

    def resolve_aliases(self, alias: str) -> str | None:
        return self.ctx.alias_provider.resolve_alias(alias)

    def resolve_memory_ref(self, ref: int) -> Optional["MemoryItem"]:
        """
        Args:
            ref (int): 记忆引用ID.
        """
        return self.ctx.ref_provider.get_item_by_ref(ref)

    def resolve_core_memory_ref(self, ref: str) -> Optional["CoreMemory"]:
        """
        Args:
            ref (str): 核心记忆引用ID (如 'CM:1').
        """
        return self.ctx.ref_provider.get_core_memory_by_ref(ref)

    def resolve_knowledge_ref(self, ref: str) -> Optional["KnowledgeEntry"]:
        """
        Args:
            ref (str): 知识库引用ID (如 'KB:1').
        """
        return self.ctx.ref_provider.get_knowledge_by_ref(ref)

    def resolve_media_ref(self, ref: str, expect_type: Type[T]) -> Optional[T]:
        """
        Args:
            ref (str): 媒体引用ID.
            expect_type (Type[BaseSegment]): 期望的媒体类型.
        """
        return self.ctx.ref_provider.get_media_typed(ref, expect_type)

    def resolve_media_by_memory_ref(
        self, memory_ref: int, expect_type: Type[T]
    ) -> List[T]:
        """通过记忆项引用ID获取媒体片段

        Args:
            memory_ref: 记忆项引用ID (整数，如 1, 2, 3)
            expect_type: 期望的媒体类型

        Returns:
            媒体片段列表
        """
        return self.ctx.ref_provider.get_media_by_memory_ref_typed(
            memory_ref, expect_type
        )

    def get_source_by_role(self, role: str) -> "MemorySource|None":
        """根据角色获取记忆源

        Args:
            role (str): 记忆源角色
        """
        return next((s for s in self._sources if s.role == role), None)

    def copy(
        self, share_context: bool = True, empty: bool = False
    ) -> "MemoryContextBuilder":
        """创建当前构建器的副本

        Args:
            share_context (bool, optional): 是否共享上下文. 默认为 True.
            empty (bool, optional): 是否创建一个空的构建器. 默认为 False.
        """
        return MemoryContextBuilder(
            ctx=self.ctx if share_context else self.ctx.copy(share_providers=False),
            sources=self._sources.copy() if not empty else [],
            agent_id=self.agent_ids
            if share_context
            else self.agent_ids.copy()
            if self.agent_ids
            else None,
            custom_ref=None
            if share_context
            else self.ctx.alias_provider.alias_map.copy(),
        )

    def __add__(self, other: "MemorySource") -> "MemoryContextBuilder":
        new_builder = self.copy(share_context=False)
        new_builder.add(other)
        return new_builder

    def __iadd__(self, other: "MemorySource") -> Self:
        return self.add(other)

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, item: "MemorySource"):
        return item in self._sources

    @overload
    def __getitem__(self, index: int) -> "MemorySource": ...

    @overload
    def __getitem__(self, index: slice) -> "MemoryContextBuilder": ...

    def __getitem__(self, index: int | slice) -> "MemorySource | MemoryContextBuilder":
        if isinstance(index, slice):
            new_builder = self.copy(share_context=True)
            new_builder._sources = self._sources[index]
            new_builder._dirty = True
            return new_builder
        return self._sources[index]

    def __bool__(self) -> bool:
        return len(self._sources) > 0

    def __iter__(self):
        return iter(self._sources)
