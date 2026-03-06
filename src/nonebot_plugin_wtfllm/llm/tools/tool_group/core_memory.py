import time

from .base import ToolGroupMeta
from ...deps import Context
from ....utils import logger, count_tokens
from ....memory.items.core_memory import CoreMemory
from ....v_db import core_memory_repo
from ....abilities.core_memory_compressor import schedule_compress

core_memory_group = ToolGroupMeta(
    name="CoreMemory",
    description="核心记忆工具组，用于在对话中主动记录、更新和管理重要记忆",
)


@core_memory_group.tool(cost=1)
async def append_core_memory(ctx: Context, content: str) -> str:
    """记录一条新的核心记忆。当对话中出现值得长期记住的信息时调用。

    记忆应当是精炼的、有价值的信息，而非原始对话的复述。
    例如："{{User_1}} 是一名大三CS学生，对系统编程特别感兴趣"
    而非："User_1 说他是学计算机的"

    Args:
        content: 记忆内容，使用 {{entity_id}} 格式引用实体
    """
    token_count = count_tokens(content)

    memory = CoreMemory(
        content=content,
        group_id=ctx.deps.ids.group_id,
        user_id=ctx.deps.ids.user_id if not ctx.deps.ids.group_id else None,
        agent_id=ctx.deps.ids.agent_id,
        token_count=token_count,
    )

    memory.normalize_placeholders(ctx.deps.context.ctx)

    await core_memory_repo.save_core_memory(memory)

    logger.debug(
        f"Appended core memory {memory.storage_id}: {content} (tokens: {token_count})"
    )

    schedule_compress(
        agent_id=ctx.deps.ids.agent_id,
        group_id=ctx.deps.ids.group_id,
        user_id=ctx.deps.ids.user_id if not ctx.deps.ids.group_id else None,
    )

    return f"核心记忆已记录。(tokens: {token_count})"


@core_memory_group.tool(cost=0)
async def update_core_memory(ctx: Context, memory_ref: str, new_content: str) -> str:
    """更新一条已有的核心记忆。当信息需要修正或补充时调用。

    Args:
        memory_ref: 核心记忆的引用标识（从 prompt 注入的记忆块中获取，如 CM:1）
        new_content: 更新后的内容
    """
    core_memory = ctx.deps.context.resolve_core_memory_ref(memory_ref)
    if core_memory is None:
        return f"错误：未找到引用 '{memory_ref}' 对应的核心记忆。"

    core_memory.content = new_content
    core_memory.updated_at = int(time.time())
    core_memory.token_count = count_tokens(new_content)
    core_memory.related_entities = []

    core_memory.normalize_placeholders(ctx.deps.context.ctx)

    await core_memory_repo.save_core_memory(core_memory)

    logger.debug(
        f"Updated core memory {core_memory.storage_id}: {new_content} "
        f"(tokens: {core_memory.token_count})"
    )

    schedule_compress(
        agent_id=ctx.deps.ids.agent_id,
        group_id=ctx.deps.ids.group_id,
        user_id=ctx.deps.ids.user_id if not ctx.deps.ids.group_id else None,
    )

    return f"核心记忆已更新。(tokens: {core_memory.token_count})"


@core_memory_group.tool(cost=0)
async def delete_core_memory(ctx: Context, memory_ref: str) -> str:
    """删除一条不再相关的核心记忆。

    Args:
        memory_ref: 核心记忆的引用标识（从 prompt 注入的记忆块中获取，如 CM:1）
    """
    core_memory = ctx.deps.context.resolve_core_memory_ref(memory_ref)
    if core_memory is None:
        return f"错误：未找到引用 '{memory_ref}' 对应的核心记忆。"

    await core_memory_repo.delete_by_storage_ids([core_memory.storage_id])

    logger.debug(f"Deleted core memory {core_memory.storage_id}")

    return "核心记忆已删除。"
