from typing import TYPE_CHECKING

from nonebot.adapters import Bot
from nonebot_plugin_uninfo import Uninfo
from nonebot_plugin_alconna import MsgId, UniMessage

from ..msg_tracker import msg_tracker
from ..topic import topic_manager
from ..utils import extract_session_info
from .extract import convert_and_store_item

if TYPE_CHECKING:
    from ..memory import MemoryItemUnion


async def store_message_with_context(
    *,
    agent_id: str,
    uni_msg: UniMessage,
    sender: str,
    msg_id: MsgId | str,
    session: "Uninfo | None" = None,
    user_id: str | None = None,
    group_id: str | None = None,
    bot: "Bot | None" = None,
    enqueue_to_agent_queue: bool = False,
    track_message: bool = False,
    ingest_topic: bool = False,
) -> "MemoryItemUnion":
    if session is not None:
        info = extract_session_info(session)
        resolved_user_id = info["user_id"]
        resolved_group_id = info["group_id"]
    else:
        resolved_user_id = user_id
        resolved_group_id = group_id

    if not resolved_user_id and not resolved_group_id:
        raise ValueError("Either session or user_id/group_id must be provided")

    item = await convert_and_store_item(
        agent_id=agent_id,
        user_id=resolved_user_id,
        uni_msg=uni_msg,
        group_id=resolved_group_id,
        sender=sender,
        msg_id=msg_id,
    )

    if enqueue_to_agent_queue:
        if bot is None or session is None:
            raise ValueError(
                "bot and session are required when enqueue_to_agent_queue is enabled"
            )

        from ..services.func import try_enqueue_message

        try_enqueue_message(bot, session, item)

    if track_message and resolved_user_id:
        msg_tracker.track(
            agent_id=agent_id,
            user_id=resolved_user_id,
            group_id=resolved_group_id,
            msg_id=str(msg_id),
        )

    if ingest_topic:
        plain_text = item.get_plain_text()
        if plain_text:
            await topic_manager.ingest(
                agent_id=agent_id,
                group_id=resolved_group_id,
                user_id=resolved_user_id,
                message_id=msg_id,
                plain_text=plain_text,
                related_message_id=item.related_message_id,
            )

    return item
