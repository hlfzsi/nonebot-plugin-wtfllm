"""Segment 内容压缩函数 — 对超长文本/转发消息进行启发式压缩展示"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from ..context import LLMContext
    from .segments import Node


def condense_text(content: str, max_chars: int = 60) -> Tuple[str, bool]:
    """对超长文本取头尾各半压缩。

    Args:
        content: 原始文本
        max_chars: 最大字符数，超过时压缩

    Returns:
        (压缩后文本, 是否被压缩)
    """
    if len(content) <= max_chars:
        return content, False

    half = max_chars // 2
    head = content[:half]
    tail = content[-half:]
    return f"{head}\n[...省略...]\n{tail}", True


def condense_forward(
    children: List["Node"],
    ctx: "LLMContext",
    message_id: str,
    memory_ref: int | None,
    max_chars: int = 60,
) -> str | None:
    """对超长合并转发消息保留首尾子消息，中间省略。

    Args:
        children: 转发消息子节点列表
        ctx: LLM 上下文
        message_id: 所属消息 ID
        memory_ref: 记忆引用号
        max_chars: 子消息文本最大字符数

    Returns:
        压缩后的渲染文本，若不需要压缩返回 None
    """
    KEEP_HEAD = 3
    KEEP_TAIL = 2

    total = len(children)
    if total <= KEEP_HEAD + KEEP_TAIL + 2:
        return None

    lines: List[str] = [f"[合并转发消息, 共{total}条:]"]

    for node in children[:KEEP_HEAD]:
        sender_alias = ctx.alias_provider.get_alias(node.sender) or node.sender
        content_str = node.content.to_llm_context(ctx, message_id, memory_ref)
        text, _ = condense_text(content_str, max_chars)
        lines.append(f"  > {sender_alias}: {text}")

    skipped = total - KEEP_HEAD - KEEP_TAIL
    lines.append(f"  > [...省略中间{skipped}条消息...]")

    for node in children[-KEEP_TAIL:]:
        sender_alias = ctx.alias_provider.get_alias(node.sender) or node.sender
        content_str = node.content.to_llm_context(ctx, message_id, memory_ref)
        text, _ = condense_text(content_str, max_chars)
        lines.append(f"  > {sender_alias}: {text}")

    lines.append("[合并转发结束]")
    return "\n".join(lines)
