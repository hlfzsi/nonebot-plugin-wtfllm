import asyncio
from typing import List

from pydantic_ai import ToolReturn
from nonebot.adapters import Event, Bot
from nonebot_plugin_waiter import waiter
from nonebot_plugin_alconna import UniMessage, MsgId


from .base import ToolGroupMeta
from .utils import reschedule_deadline
from ...deps import Context
from ....stream_processing import (
    store_message_with_context,
)
from ....memory import MemoryItemStream
from ....utils import ensure_msgid_from_receipt

chat_tool_group = ToolGroupMeta(
    name="Chat",
    description=(
        "用于在对话流程中主动与用户进行交互式沟通的工具组，"
        "支持发送中间思考/引导性消息（如‘让我想想…’）以及主动提问并等待用户回复，"
        "需要完成连续任务时调用, 比如研究分析"
    ),
)


def build_uni_from_metions_and_reply(
    ctx: Context,
    content: str,
    mentions: List[str] | None = None,
    reply_to: str | None = None,
) -> UniMessage:
    msg = UniMessage()
    if mentions:
        for mention in mentions:
            _user_id = ctx.deps.context.resolve_aliases(mention)
            if _user_id:
                msg.at(user_id=_user_id)
    msg.text(content)
    if reply_to:
        _reply_msg = ctx.deps.context.resolve_memory_ref(int(reply_to))
        if _reply_msg:
            msg.reply(id=_reply_msg.message_id)
    return msg


@chat_tool_group.tool(cost=-1)
async def send(
    ctx: Context,
    message: str,
    mentions: List[str] | None = None,
    reply_to: str | None = None,
    added_timeout: float = 60.0,
) -> str:
    """
    在聊天中发送一条【中间性、试探性或思考过程】的消息。

    - 用于在对话流程中插入“我看看...”、“让我想想...”、“哦哦是这样”等过渡性、非最终结论性的回复。
    - 适用于需要先“试探”、“确认”、“引导”或“表达思考过程”的场景。
    - 与最终结束对话有本质区别：本工具发送的消息**不是最终答案或结论**，而是对话流程中的“中间步骤”或“思考反馈”。

    Args:
        message (str): 要发送的文本内容
        mentions (List[str] | None, optional): 需要@的用户ID列表 , 仅在针对性回复时使用
        reply_to (str | None, optional): 回复某条消息的ID, 仅在针对性回复时使用
        added_timeout (float): 发送此消息后，为当前对话增加的超时时间（秒），默认为60秒
    """
    delay = min(4.0, max(0.0, len(message) * 0.1))
    if ctx.deps.nb_runtime is None:
        raise ValueError("NonebotRuntime is required to send messages")
    if ctx.deps.ids.user_id is None:
        raise ValueError("User ID is required to send messages")

    reschedule_deadline(ctx, delay + added_timeout)

    msg = build_uni_from_metions_and_reply(
        ctx=ctx,
        content=message,
        mentions=mentions,
        reply_to=reply_to,
    )

    if delay <= 0:
        delay = 0
    elif delay > 3:
        delay = 3

    async def _send():
        if ctx.deps.nb_runtime is None:
            raise ValueError("NonebotRuntime is required to send messages")
        if ctx.deps.ids.user_id is None:
            raise ValueError("User ID is required to send messages")
        sent_message = await msg.send(target=ctx.deps.nb_runtime.target)
        sent_msg_id = ensure_msgid_from_receipt(
            sent_message, ctx.deps.nb_runtime.session
        )
        await store_message_with_context(
            agent_id=ctx.deps.ids.agent_id,
            uni_msg=msg,
            sender=ctx.deps.ids.agent_id,
            msg_id=sent_msg_id,
            user_id=ctx.deps.ids.user_id,
            group_id=ctx.deps.ids.group_id,
            track_message=True,
            ingest_topic=True,
        )

    async with asyncio.TaskGroup() as tg:
        tg.create_task(_send())
        tg.create_task(asyncio.sleep(delay))

    return f"已发送消息: {msg}"


@chat_tool_group.tool(cost=-1)
async def ask(
    ctx: Context,
    question: str,
    mentions: List[str] | None = None,
    reply_to: str | None = None,
    timeout: float = 30.0,
    added_timeout: float = 60.0,
) -> ToolReturn:
    """
    向用户提出一个问题并等待其回复。

    - 用于在对话流程中主动收集用户信息、确认细节或获取下一步指令。
    - 如果用户在指定时间内未回复，则返回超时信息。

    Args:
        ctx (Context): 依赖注入的上下文对象。
        question (str): 要向用户提出的问题内容。
        mentions (List[str] | None, optional): 需要@的用户ID列表, 仅在针对性提问时使用。
        reply_to (str | None, optional): 回复某条消息的ID, 仅在针对性提问时使用。
        timeout (float): 等待用户回复的超时时间（秒），默认为 30.0 秒。
        added_timeout (float): 发送此消息后，为当前对话增加的超时时间（秒），默认为60秒
    """
    if ctx.deps.nb_runtime is None:
        raise ValueError("NonebotRuntime is required to ask questions")
    if ctx.deps.ids.user_id is None:
        raise ValueError("User ID is required to ask questions")

    reschedule_deadline(ctx, timeout + added_timeout)

    msg = build_uni_from_metions_and_reply(
        ctx=ctx,
        content=question,
        mentions=mentions,
        reply_to=reply_to,
    )

    def wrapper(bot: Bot, event: Event, msg_id: MsgId) -> tuple[UniMessage, MsgId]:
        return UniMessage.of(event.get_message(), bot=bot), msg_id

    wait = waiter(["message"], keep_session=True, block=True)(wrapper)

    sent_message = await msg.send(target=ctx.deps.nb_runtime.target)
    sent_msg_id = ensure_msgid_from_receipt(sent_message, ctx.deps.nb_runtime.session)
    await store_message_with_context(
        agent_id=ctx.deps.ids.agent_id,
        uni_msg=msg,
        sender=ctx.deps.ids.agent_id,
        msg_id=sent_msg_id,
        user_id=ctx.deps.ids.user_id,
        group_id=ctx.deps.ids.group_id,
        track_message=True,
        ingest_topic=True,
    )
    result = await wait.wait(timeout=timeout)

    if result is None:
        return ToolReturn(return_value="用户未回复")
    else:
        uni_msg, reply_msg_id = result
        result_memory_item = await store_message_with_context(
            agent_id=ctx.deps.ids.agent_id,
            uni_msg=uni_msg,
            sender=ctx.deps.ids.user_id,
            msg_id=reply_msg_id,
            user_id=ctx.deps.ids.user_id,
            group_id=ctx.deps.ids.group_id,
            track_message=True,
            ingest_topic=True,
        )
        return_msg = "已收到回复"

    new_builder = ctx.deps.context.copy(share_context=True, empty=True)

    new_builder.add(MemoryItemStream(items=[result_memory_item]))

    prompt = new_builder.to_prompt()

    return ToolReturn(return_value=return_msg, content=prompt)
