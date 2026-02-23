"""history_processor: 在每轮 LLM 请求前检查并注入新到达的消息。"""

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from .deps import AgentDeps
from ..memory import MemoryItemStream
from ..utils import logger


async def inject_new_messages(
    ctx: RunContext[AgentDeps], messages: list[ModelMessage]
) -> list[ModelMessage]:
    """将 Agent 处理期间新到达的消息注入 LLM 上下文。"""
    queue = ctx.deps.message_queue

    if not queue or not isinstance(queue, list):
        return messages

    new_items = queue[:]
    del queue[:]

    if not new_items:
        return messages

    logger.debug(f"Injecting {len(new_items)} new message(s) into LLM context")

    builder = ctx.deps.context.copy(share_context=True, empty=True)
    stream = MemoryItemStream(
        items=new_items,
        prefix="<new_messages>",
        suffix="</new_messages>",
    )
    builder.add(stream)
    new_prompt_text = builder.to_prompt()

    injection_part = UserPromptPart(content=new_prompt_text)

    logger.debug(f"Injected prompt part: {injection_part.content}")
    if messages and isinstance(messages[-1], ModelRequest):
        messages[-1].parts = list(messages[-1].parts) + [injection_part]
    else:
        messages.append(ModelRequest(parts=[injection_part]))

    return messages
