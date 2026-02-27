"""消息事件存储 handler"""

from nonebot import on_message
from nonebot.adapters import Bot
from nonebot_plugin_alconna import MsgId, OriginalUniMsg
from nonebot_plugin_uninfo import Uninfo

from .func import set_alias_to_cache, like_command
from ..utils import get_agent_id_from_bot
from ..stream_processing import store_message_with_context

matcher = on_message(block=False, priority=98)  # 避免其他插件指令消息影响


@matcher.handle()
async def handle(
    bot: Bot,
    uni_msg: OriginalUniMsg,
    session: Uninfo,
    msg_id: MsgId,
) -> None:
    agent_id = get_agent_id_from_bot(session)

    set_alias_to_cache(bot=bot, session=session)

    if like_command(uni_msg):
        return

    await store_message_with_context(
        agent_id=agent_id,
        uni_msg=uni_msg,
        sender=session.user.id,
        msg_id=msg_id,
        session=session,
        bot=bot,
        enqueue_to_agent_queue=True,
        ingest_topic=True,
    )
