from typing import List, TYPE_CHECKING, Tuple

from pydantic import BaseModel, Field

from .base import MemorySource

if TYPE_CHECKING:
    from ..context import LLMContext


class ToolCallSummaryBlock(BaseModel, MemorySource):
    """工具调用历史概要块，仅展示工具名列表"""

    tool_names: List[str] = Field(default_factory=list)
    prefix: str | None = Field(default=None)
    suffix: str | None = Field(default=None)

    @property
    def source_id(self) -> str:
        return f"tool-call-summary-{hash(id(self))}"

    @property
    def priority(self) -> int:
        return 1

    @property
    def sort_key(self) -> Tuple[int, str]:
        return (0, self.source_id)

    def register_all_alias(self, ctx: "LLMContext") -> None:
        pass

    def to_llm_context(self, ctx: "LLMContext") -> str:
        if not self.tool_names:
            return ""

        names = sorted(list(set(self.tool_names)))

        lines = []
        if self.prefix:
            lines.append(self.prefix)
        for name in names:
            lines.append(f"- {name}")
        if self.suffix:
            lines.append(self.suffix)
        return "\n".join(lines)

    def __hash__(self) -> int:
        return hash(id(self))
