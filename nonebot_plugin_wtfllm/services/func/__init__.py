__all__ = [
    "handle_control",
    "get_alias_from_cache",
    "set_alias_to_cache",
    "try_enqueue_message",
    "get_conv_key",
]

from nonebot_plugin_alconna import UniMessage

from .agent_cache import (
    handle_control,
    get_alias_from_cache,
    set_alias_to_cache,
    try_enqueue_message,
    get_conv_key,
)

_COLLECT = (
    "/ban",
    "/eb",
    "/easyban",
    "/summary",
    "/摘要",
    "/delete_media",
    "/del",
    "/delete",
    "删除",
)


def like_command(msg: str | UniMessage) -> bool:
    if isinstance(msg, UniMessage):
        return any(msg.startswith(cmd) for cmd in _COLLECT)
    elif isinstance(msg, str):
        return msg.startswith(_COLLECT)
