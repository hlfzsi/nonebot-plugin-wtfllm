import time

from nonebot_plugin_alconna import Alconna, on_alconna, Args, Option, Query
from nonebot_plugin_uninfo import Uninfo

from ..db import memory_item_repo
from ..memory import MediaBaseSegment
from ..utils import APP_CONFIG, get_agent_id_from_bot


async def _perm(session: Uninfo) -> bool:
    if session.user.id in APP_CONFIG.admin_users:
        return True
    else:
        return False


del_alc = Alconna(
    "delete_media",
    Option(
        "-d|--day", Args["expiry_days", int], help_text="清理天数超过指定天数的媒体文件"
    ),
)

matcher = on_alconna(
    del_alc,
    aliases={
        "del",
        "delete",
        "删除",
    },
    use_cmd_start=True,
    block=True,
    rule=_perm,
)


@matcher.handle()
async def handle_delete_media(
    session: Uninfo, opt_days: Query[int] = Query("day.expiry_days")
):
    expiry_days = (
        opt_days.result if opt_days.available else APP_CONFIG.media_lifecycle_days
    )

    agent_id = get_agent_id_from_bot(session)

    count = await _unbound(agent_id=agent_id, expiry_days=expiry_days)

    await matcher.finish(f"已清理{count}条过期媒体记录")


async def _unbound(agent_id: str, expiry_days: int) -> int:
    expiry_timestamp = int(time.time()) - expiry_days * 24 * 3600

    items_to_unbound = await memory_item_repo.get_by_timestamp_before(
        expiry_timestamp, agent_id
    )
    meida_items_to_unbound = []
    for item in items_to_unbound:
        segs = item.content.deep_get(MediaBaseSegment)
        if segs:
            for seg in segs:
                seg.unbound_local(expired=True)
            meida_items_to_unbound.append(item)
    count = await memory_item_repo.save_many(meida_items_to_unbound)
    return count
