import datetime
from typing import List, Self, TYPE_CHECKING

from pydantic import BaseModel, Field, PrivateAttr

from .base import MemorySource
from .base_items import MemoryItemUnion
from ...utils import count_tokens

if TYPE_CHECKING:
    from ..context import LLMContext


class MemoryItemStream(BaseModel, MemorySource):
    suffix: str | None = Field(default=None, description="记忆流后缀提示")
    prefix: str | None = Field(default=None, description="记忆流前缀提示")
    items: List[MemoryItemUnion] = Field(..., description="记忆项列表")

    _max_token: int | None = PrivateAttr(default=None)
    _role: str | None = PrivateAttr(default=None)
    _priority: float = PrivateAttr(default=0)

    @classmethod
    def create(
        cls,
        max_token: int | None = None,
        prefix: str | None = None,
        suffix: str | None = None,
        items: List[MemoryItemUnion] | MemoryItemUnion | None = None,
        role: str | None = None,
        priority: float = 0,
    ) -> Self:
        if not (0 <= priority < 1):
            raise ValueError("priority must be between 0 and 1(no inclusive)")

        if items is None:
            items_list = []
        elif isinstance(items, list):
            items_list = items
        else:
            items_list = [items]

        instance = cls(items=items_list, prefix=prefix, suffix=suffix)
        instance._max_token = max_token
        instance._role = role
        instance._priority = priority
        return instance

    @property
    def role(self) -> str | None:
        return self._role

    @property
    def started_at(self) -> int:
        if not self.items:
            return 0
        return min(item.created_at for item in self.items)

    @property
    def ended_at(self) -> int:
        if not self.items:
            return 0
        return max(item.created_at for item in self.items)

    @property
    def source_id(self) -> str:
        return f"stream-{hash(id(self))}"

    @property
    def priority(self) -> float:
        return 0 + self._priority

    @property
    def sort_key(self) -> tuple[int, str]:
        return (self.ended_at, str(self.source_id))

    def register_all_alias(self, ctx: "LLMContext") -> None:
        for item in self.items:
            item.register_entities(ctx)

    def to_llm_context(self, ctx: "LLMContext") -> str:
        self.items.sort(key=lambda x: x.created_at)
        lines = []
        if self.prefix:
            lines.append(self.prefix)
        last_ts = 0
        last_date_str = ""
        for item in self.items:
            curr_time = datetime.datetime.fromtimestamp(item.created_at)
            curr_date_str = curr_time.strftime("%Y-%m-%d")

            if curr_date_str != last_date_str:
                lines.append(curr_time.strftime("%Y-%m-%d %H:%M"))
                last_date_str = curr_date_str
                last_ts = item.created_at

            elif item.created_at - last_ts > 60 * 10:
                lines.append(curr_time.strftime("%H:%M"))
                last_ts = item.created_at

            lines.append(item.to_llm_context(ctx))

        if self.suffix:
            lines.append(self.suffix)

        if self._max_token is not None:
            result = "\n".join(lines)
            prefix_lines = 1 if self.prefix else 0
            min_lines = prefix_lines + (1 if self.suffix else 0)
            while count_tokens(result) > self._max_token and len(lines) > min_lines:
                lines.pop(prefix_lines)
                result = "\n".join(lines)
            return result

        return "\n".join(lines)

    def append(self, item: MemoryItemUnion) -> Self:
        self.items.append(item)
        return self

    def __add__(
        self, other: List[MemoryItemUnion] | MemoryItemUnion | "MemoryItemStream"
    ) -> "MemoryItemStream":
        """注意, 该方法不保留_role属性"""
        new_stream = MemoryItemStream.create(
            items=self.items.copy(), max_token=self._max_token
        )

        if isinstance(other, MemoryItemStream):
            new_stream.items.extend(other.items)
        elif isinstance(other, list):
            new_stream.items.extend(other)
        else:
            new_stream.items.append(other)

        return new_stream

    def __iadd__(
        self, other: List[MemoryItemUnion] | MemoryItemUnion | "MemoryItemStream"
    ) -> Self:
        if isinstance(other, MemoryItemStream):
            self.items.extend(other.items)
        elif isinstance(other, list):
            self.items.extend(other)
        else:
            self.items.append(other)
        return self

    def __len__(self) -> int:
        return len(self.items)

    def __hash__(self) -> int:
        """不安全的哈希值，仅用于区分不同实例"""
        return hash(id(self))
