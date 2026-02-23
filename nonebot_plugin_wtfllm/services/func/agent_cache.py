from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Set, Generator

from cachetools import LRUCache
from nonebot.adapters import Bot
from nonebot_plugin_uninfo import Uninfo

from .message_queue import (
    get_conversation_key,
    create_queue,
    remove_queue,
    get_queue,
)

if TYPE_CHECKING:
    from ...memory import MemoryItemUnion

UserOrGroupID = str
GroupID = str
Alias = str

_in_handle: Set[str] = set()
_alias_cache: LRUCache[str, Dict[UserOrGroupID, Alias]] = LRUCache(maxsize=100)
_group_alias_cache: LRUCache[GroupID, Alias] = LRUCache(maxsize=100)


def get_handle_key(bot: Bot, session: Uninfo) -> str:
    adapter = bot.adapter.get_name()
    group_part = session.group.id if session.group else "private"
    return f"{adapter}:{bot.self_id}:{session.user.id}:{group_part}"


def get_conv_key(bot: Bot, session: Uninfo) -> str:
    """生成会话级别的队列键（群聊按 group_id，私聊按 user_id）"""
    adapter = bot.adapter.get_name()
    group_id = session.group.id if session.group else None
    return get_conversation_key(adapter, bot.self_id, group_id, session.user.id)


@contextmanager
def handle_control(bot: Bot, session: Uninfo) -> Generator[bool, None, None]:
    key = get_handle_key(bot, session)
    if key in _in_handle:
        yield False
        return

    conv_key = get_conv_key(bot, session)
    _in_handle.add(key)
    create_queue(conv_key)
    try:
        yield True
    finally:
        _in_handle.discard(key)
        remove_queue(conv_key)


def try_enqueue_message(bot: Bot, session: Uninfo, item: "MemoryItemUnion") -> bool:
    """尝试将消息推入正在运行的 Agent 队列。返回 True 表示已推入。"""
    conv_key = get_conv_key(bot, session)
    queue = get_queue(conv_key)
    if queue is not None:
        queue.append(item)
        return True
    return False


def get_alias_cache_main_key(bot: Bot, session: Uninfo) -> str:
    adapter = bot.adapter.get_name()
    if session.group:
        return f"{adapter}:{bot.self_id}:g:{session.group.id}"
    return f"{adapter}:{bot.self_id}:p:{session.user.id}"


def get_alias_from_cache(bot: Bot, session: Uninfo) -> Dict[UserOrGroupID, Alias]:
    main_key = get_alias_cache_main_key(bot, session)

    res = _alias_cache.get(main_key, {}).copy()
    res.update(dict(_group_alias_cache))
    return res


def set_alias_to_cache(bot: Bot, session: Uninfo) -> None:
    main_key = get_alias_cache_main_key(bot, session)
    current_aliases = _alias_cache.get(main_key, {}).copy()

    changed = False

    user_alias = session.user.nick or session.user.name
    if user_alias:
        current_aliases[session.user.id] = user_alias
        changed = True

    if session.group and session.group.name:
        _group_alias_cache[session.group.id] = session.group.name

    if changed:
        _alias_cache[main_key] = current_aliases
