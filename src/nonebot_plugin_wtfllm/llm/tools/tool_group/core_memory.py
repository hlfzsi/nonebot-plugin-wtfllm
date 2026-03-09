import time
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from .base import ToolGroupMeta
from ...deps import Context
from ....abilities.core_memory_compressor import schedule_compress
from ....db import note_memory_repo
from ....memory.items.core_memory import CoreMemory
from ....memory.items.note import Note
from ....utils import logger, count_tokens
from ....v_db import core_memory_repo

core_memory_group = ToolGroupMeta(
    name="Memory",
    description="记忆工具组，用于管理核心记忆和会话级短期备忘",
)


class CoreMemoryAppendPayload(BaseModel):
    memory_kind: Literal["core_memory"] = "core_memory"
    content: str = Field(..., description="核心记忆内容，适合长期保留")


class NoteAppendPayload(BaseModel):
    memory_kind: Literal["note"] = "note"
    content: str = Field(..., description="短期备忘内容")
    duration_minutes: int = Field(
        ...,
        ge=1,
        description="持续时间，单位分钟，必须大于 0",
    )

    @model_validator(mode="after")
    def validate_expiry(self) -> "NoteAppendPayload":
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be greater than 0")
        return self


class CoreMemoryUpdatePayload(BaseModel):
    memory_kind: Literal["core_memory"] = "core_memory"
    new_content: str = Field(..., description="更新后的核心记忆内容")


class NoteUpdatePayload(BaseModel):
    memory_kind: Literal["note"] = "note"
    new_content: str | None = Field(default=None, description="更新后的短期备忘内容")
    new_duration_minutes: int | None = Field(
        default=None,
        ge=1,
        description="更新后的持续时间，单位分钟，必须大于 0",
    )

    @model_validator(mode="after")
    def validate_update(self) -> "NoteUpdatePayload":
        if self.new_content is None and self.new_duration_minutes is None:
            raise ValueError("note update requires new_content or new_duration_minutes")
        if self.new_duration_minutes is not None and self.new_duration_minutes <= 0:
            raise ValueError("new_duration_minutes must be greater than 0")
        return self


AppendPayload = Annotated[
    CoreMemoryAppendPayload | NoteAppendPayload,
    Field(discriminator="memory_kind"),
]
UpdatePayload = Annotated[
    CoreMemoryUpdatePayload | NoteUpdatePayload,
    Field(discriminator="memory_kind"),
]


class AppendMemoryRequest(BaseModel):
    operation: Literal["append"] = "append"
    payload: AppendPayload


class UpdateMemoryRequest(BaseModel):
    operation: Literal["update"] = "update"
    target_ref: str = Field(..., description="要更新的记忆引用，如 CM:1 或 NT:1")
    payload: UpdatePayload


class DeleteMemoryRequest(BaseModel):
    operation: Literal["delete"] = "delete"
    memory_kind: Literal["core_memory", "note"]
    target_ref: str = Field(..., description="要删除的记忆引用，如 CM:1 或 NT:1")


MemoryCrudAction = Annotated[
    AppendMemoryRequest | UpdateMemoryRequest | DeleteMemoryRequest,
    Field(discriminator="operation"),
]


class MemoryCrudRequest(BaseModel):
    action: MemoryCrudAction


def _agent_id(ctx: Context) -> str:
    return ctx.deps.ids.agent_id


def _group_id(ctx: Context) -> str | None:
    return ctx.deps.ids.group_id


def _private_user_id(ctx: Context) -> str | None:
    return ctx.deps.ids.user_id if not ctx.deps.ids.group_id else None


def _duration_minutes_to_expires_at(duration_minutes: int) -> int:
    return int(time.time()) + duration_minutes * 60


@core_memory_group.tool(cost=1)
async def memory_crud(ctx: Context, request: MemoryCrudRequest) -> str:
    """统一管理会话记忆。

    使用结构化请求完成 append/update/delete 操作：
    - core_memory: 长期稳定事实
    - note: 会话级短期备忘
    """
    action = request.action

    if isinstance(action, AppendMemoryRequest):
        payload = action.payload
        if isinstance(payload, CoreMemoryAppendPayload):
            token_count = count_tokens(payload.content)
            memory = CoreMemory(
                content=payload.content,
                token_count=token_count,
                group_id=_group_id(ctx),
                user_id=_private_user_id(ctx),
                agent_id=_agent_id(ctx),
            )
            memory.normalize_placeholders(ctx.deps.context.ctx)
            await core_memory_repo.save_core_memory(memory)
            logger.debug(
                f"Appended core memory {memory.storage_id}: {payload.content} "
            )
            schedule_compress(
                agent_id=_agent_id(ctx),
                group_id=_group_id(ctx),
                user_id=_private_user_id(ctx),
            )
            return "核心记忆已记录"

        elif isinstance(payload, NoteAppendPayload):
            token_count = count_tokens(payload.content)
            expires_at = _duration_minutes_to_expires_at(payload.duration_minutes)
            note = Note(
                content=payload.content,
                expires_at=expires_at,
                token_count=token_count,
                group_id=_group_id(ctx),
                user_id=_private_user_id(ctx),
                agent_id=_agent_id(ctx),
            )
            await note_memory_repo.save_note(note)
            logger.debug(
                f"Appended note {note.storage_id}: {payload.content} (tokens: {token_count}, duration_minutes: {payload.duration_minutes}, expires_at: {expires_at})"
            )
            return f"短期备忘已记录，将在 {payload.duration_minutes} 分钟后过期。"
        else:
            raise ValueError("Unsupported payload type")

    if isinstance(action, UpdateMemoryRequest):
        payload = action.payload
        if isinstance(payload, CoreMemoryUpdatePayload):
            core_memory = ctx.deps.context.resolve_core_memory_ref(action.target_ref)
            if core_memory is None:
                return f"错误：未找到引用 '{action.target_ref}' 对应的核心记忆。"

            core_memory.content = payload.new_content
            core_memory.updated_at = int(time.time())
            core_memory.token_count = count_tokens(payload.new_content)
            core_memory.related_entities = []
            core_memory.normalize_placeholders(ctx.deps.context.ctx)
            await core_memory_repo.save_core_memory(core_memory)
            logger.debug(
                f"Updated core memory {core_memory.storage_id}: {payload.new_content} (tokens: {core_memory.token_count})"
            )
            schedule_compress(
                agent_id=_agent_id(ctx),
                group_id=_group_id(ctx),
                user_id=_private_user_id(ctx),
            )
            return "核心记忆已更新"

        elif isinstance(payload, NoteUpdatePayload):
            note = ctx.deps.context.resolve_note_ref(action.target_ref)
            if note is None:
                return f"错误：未找到引用 '{action.target_ref}' 对应的短期备忘。"

            next_token_count = note.token_count
            if payload.new_content is not None:
                next_token_count = count_tokens(payload.new_content)
            if payload.new_content is not None:
                note.content = payload.new_content
                note.token_count = next_token_count
            if payload.new_duration_minutes is not None:
                note.expires_at = _duration_minutes_to_expires_at(
                    payload.new_duration_minutes
                )
            note.updated_at = int(time.time())
            await note_memory_repo.save_note(note)
            logger.debug(
                f"Updated note {note.storage_id}: content_updated={payload.new_content is not None}, duration_minutes={payload.new_duration_minutes}, expires_at={note.expires_at}"
            )
            return "短期备忘已更新"
        else:
            raise ValueError("Unsupported payload type")

    if isinstance(action, DeleteMemoryRequest):
        if action.memory_kind == "core_memory":
            core_memory = ctx.deps.context.resolve_core_memory_ref(action.target_ref)
            if core_memory is None:
                return f"错误：未找到引用 '{action.target_ref}' 对应的核心记忆。"
            await core_memory_repo.delete_by_storage_ids([core_memory.storage_id])
            logger.debug(f"Deleted core memory {core_memory.storage_id}")
            return "核心记忆已删除。"

        elif action.memory_kind == "note":
            note = ctx.deps.context.resolve_note_ref(action.target_ref)
            if note is None:
                return f"错误：未找到引用 '{action.target_ref}' 对应的短期备忘。"
            await note_memory_repo.delete_by_storage_ids([note.storage_id])
            logger.debug(f"Deleted note {note.storage_id}")
            return "短期备忘已删除。"

        else:
            raise ValueError("Unsupported memory_kind")

    raise ValueError("Unsupported action type")
