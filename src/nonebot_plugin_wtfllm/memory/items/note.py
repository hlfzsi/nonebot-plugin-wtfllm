import time
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Tuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from .base import MemorySource

if TYPE_CHECKING:
    from ..context import LLMContext


class Note(BaseModel, MemorySource):
    """会话级短期备忘。"""

    model_config = ConfigDict(validate_assignment=True)

    storage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Note 唯一ID"
    )
    content: str = Field(..., description="备忘内容文本")
    group_id: str | None = Field(default=None, description="所属群组（私聊为 None）")
    user_id: str | None = Field(default=None, description="所属用户（群聊为 None）")
    agent_id: str = Field(..., description="所属 agent")
    expires_at: int = Field(..., description="过期时间戳")
    created_at: int = Field(
        default_factory=lambda: int(time.time()), description="创建时间戳"
    )
    updated_at: int = Field(
        default_factory=lambda: int(time.time()), description="最后更新时间戳"
    )
    token_count: int = Field(default=0, description="缓存的 token 计数")

    _priority: float = PrivateAttr(default=0)

    @field_validator("expires_at", mode="before")
    @classmethod
    def _normalize_expires_at(cls, value: object) -> int:
        try:
            timestamp = int(value)  # pyright: ignore[reportArgumentType]
        except (TypeError, ValueError) as exc:
            raise ValueError("expires_at must be an integer timestamp") from exc

        abs_timestamp = abs(timestamp)
        if abs_timestamp >= 10**18:
            return timestamp // 10**9
        if abs_timestamp >= 10**15:
            return timestamp // 10**6
        if abs_timestamp >= 10**12:
            return timestamp // 10**3
        return timestamp

    @classmethod
    def create(
        cls,
        content: str,
        agent_id: str,
        expires_at: int,
        group_id: str | None = None,
        user_id: str | None = None,
        priority: float = 0,
    ) -> "Note":
        if not (0 <= priority < 1):
            raise ValueError("priority must be between 0 and 1(no inclusive)")

        instance = cls(
            content=content,
            agent_id=agent_id,
            expires_at=expires_at,
            group_id=group_id,
            user_id=user_id,
        )
        instance._priority = priority
        return instance

    @property
    def source_id(self) -> str:
        return f"note-{self.storage_id}"

    @property
    def priority(self) -> float:
        return 3 + self._priority

    @property
    def sort_key(self) -> Tuple[int, str]:
        return (self.expires_at, self.storage_id)

    @property
    def is_expired(self) -> bool:
        return self.expires_at <= int(time.time())

    def register_all_alias(self, ctx: "LLMContext") -> None:
        if self.group_id:
            ctx.alias_provider.register_group(self.group_id)
        if self.user_id:
            ctx.alias_provider.register_user(self.user_id)

    def _render_session_tag(self, ctx: "LLMContext") -> str:
        if self.group_id:
            alias = ctx.alias_provider.get_alias(self.group_id)
            return f"({alias})" if alias else f"({self.group_id})"
        if self.user_id:
            alias = ctx.alias_provider.get_alias(self.user_id)
            return f"({alias} 私聊)" if alias else f"({self.user_id} 私聊)"
        return ""

    def _render_expire_tag(self) -> str:
        now = int(time.time())
        remaining_seconds = self.expires_at - now
        try:
            expire_text = datetime.fromtimestamp(self.expires_at).strftime(
                "%Y-%m-%d %H:%M"
            )
        except (OSError, OverflowError, ValueError):
            expire_text = str(self.expires_at)
            return f"[expires {expire_text}]"

        if remaining_seconds <= 0:
            return f"[expired at {expire_text}]"

        remaining_minutes = max(1, (remaining_seconds + 59) // 60)
        return f"[expires in {remaining_minutes}m at {expire_text}]"

    def to_llm_context(self, ctx: "LLMContext") -> str:
        ref = ctx.ref_provider.next_note_ref(self)
        session_tag = self._render_session_tag(ctx)
        expire_tag = self._render_expire_tag()
        if session_tag:
            return f"[{ref}] {expire_tag} {session_tag} {self.content}"
        return f"[{ref}] {expire_tag} {self.content}"

    def __hash__(self) -> int:
        return hash(self.source_id)


class NoteBlock(BaseModel, MemorySource):
    """Note 记忆块，用于将一组短期备忘注入到 LLM 上下文中。"""

    notes: list[Note] = Field(default_factory=list)
    prefix: str | None = Field(default=None)
    suffix: str | None = Field(default=None)
    _priority: float = PrivateAttr(default=0)

    @classmethod
    def create(
        cls,
        notes: list[Note],
        prefix: str | None = None,
        suffix: str | None = None,
        priority: float = 0,
    ) -> "NoteBlock":
        if not (0 <= priority < 1):
            raise ValueError("priority must be between 0 and 1(no inclusive)")

        instance = cls(notes=notes, prefix=prefix, suffix=suffix)
        instance._priority = priority
        return instance

    @property
    def source_id(self) -> str:
        return f"note-block-{hash(id(self))}"

    @property
    def priority(self) -> float:
        return 3 + self._priority

    @property
    def sort_key(self) -> Tuple[int, str]:
        if not self.notes:
            return (0, self.source_id)
        earliest = min(note.expires_at for note in self.notes)
        return (earliest, self.source_id)

    def register_all_alias(self, ctx: "LLMContext") -> None:
        for note in self.notes:
            note.register_all_alias(ctx)

    def to_llm_context(self, ctx: "LLMContext") -> str:
        if not self.notes:
            return ""
        self.notes.sort(
            key=lambda note: (note.expires_at, note.updated_at, note.storage_id)
        )
        lines: list[str] = []
        if self.prefix:
            lines.append(self.prefix)
        for note in self.notes:
            lines.append(note.to_llm_context(ctx))
        if self.suffix:
            lines.append(self.suffix)
        return "\n".join(lines)

    def __hash__(self) -> int:
        return hash(id(self))
