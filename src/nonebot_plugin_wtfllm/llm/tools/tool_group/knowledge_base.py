import time
from typing import List

from .base import ToolGroupMeta
from ...deps import Context
from ....utils import logger, count_tokens
from ....memory.items.knowledge_base import KnowledgeEntry
from ....v_db import knowledge_base_repo

knowledge_base_group = ToolGroupMeta(
    name="KnowledgeBase",
    description="知识库工具组，用于记录、更新和管理全局共享的知识、术语和事实",
)


@knowledge_base_group.tool(cost=1)
async def add_knowledge(
    ctx: Context,
    title: str,
    content: str,
    category: str = "general",
    tags: List[str] | None = None,
) -> str:
    """记录一条新的知识到全局知识库。当对话中出现值得所有会话共享的知识、术语或事实时调用。

    知识应当是客观的、可共享的信息，而非个人记忆。
    适合记录的内容：新术语/概念的定义、技术知识、规则和约定、有趣的常识
    不适合记录的内容（应用 core_memory）：关于特定用户的个人信息、仅在某个特定会话有意义的上下文

    Args:
        title: 知识条目的简短标题/关键词（如 "React Hooks", "量子纠缠"）
        content: 知识的详细内容描述
        category: 知识分类，如 "术语", "事实", "规则", "技术", "常识" 等
        tags: 可选的标签列表，用于组织
    """

    token_count = count_tokens(content)
    entry = KnowledgeEntry(
        content=content,
        title=title,
        category=category,
        agent_id=ctx.deps.ids.agent_id,
        source_session_type="group" if ctx.deps.ids.group_id else "private",
        source_session_id=ctx.deps.ids.group_id or ctx.deps.ids.user_id,
        tags=tags or [],
        token_count=token_count,
    )

    await knowledge_base_repo.save_knowledge(entry)

    logger.debug(
        f"Added knowledge entry {entry.storage_id}: [{title}] {content[:50]}... "
        f"(category: {category}, tokens: {token_count})"
    )

    return f"知识已记录：【{title}】(分类: {category}, tokens: {token_count})"


@knowledge_base_group.tool(cost=0)
async def update_knowledge(
    ctx: Context,
    knowledge_ref: str,
    new_content: str,
    new_title: str | None = None,
) -> str:
    """更新一条已有的知识。当知识需要修正或补充时调用。或者发现有知识冲突时，主动整理更新。

    Args:
        knowledge_ref: 知识条目的引用标识（从 prompt 注入的知识块中获取，如 KB:1）
        new_content: 更新后的内容
        new_title: 可选的新标题
    """
    entry = ctx.deps.context.resolve_knowledge_ref(knowledge_ref)
    if entry is None:
        return f"错误：未找到引用 '{knowledge_ref}' 对应的知识条目。"

    entry.content = new_content
    if new_title:
        entry.title = new_title
    entry.updated_at = int(time.time())
    entry.token_count = count_tokens(new_content)

    await knowledge_base_repo.save_knowledge(entry)

    logger.debug(
        f"Updated knowledge entry {entry.storage_id}: {new_content[:50]}... "
        f"(tokens: {entry.token_count})"
    )

    return f"知识已更新。(tokens: {entry.token_count})"


@knowledge_base_group.tool(cost=0)
async def delete_knowledge(ctx: Context, knowledge_ref: str) -> str:
    """删除一条不再正确或相关的知识。

    Args:
        knowledge_ref: 知识条目的引用标识（从 prompt 注入的知识块中获取，如 KB:1）
    """
    entry = ctx.deps.context.resolve_knowledge_ref(knowledge_ref)
    if entry is None:
        return f"错误：未找到引用 '{knowledge_ref}' 对应的知识条目。"

    await knowledge_base_repo.delete_knowledge(entry.storage_id)

    logger.debug(f"Deleted knowledge entry {entry.storage_id}")

    return "知识条目已删除。"
