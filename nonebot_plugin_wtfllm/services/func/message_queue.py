"""Agent 执行期间的消息队列注册表。

当 Agent 正在处理时，同一会话中到达的新消息会被推入队列，
由 history_processor 在下一轮 LLM 迭代前注入到上下文中。
"""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ...memory import MemoryItemUnion

_pending_messages: Dict[str, List["MemoryItemUnion"]] = {}


def get_conversation_key(
    adapter: str, bot_id: str, group_id: Optional[str], user_id: str
) -> str:
    """生成会话级别的队列键（群聊用 group_id，私聊用 user_id）"""
    return f"{adapter}:{bot_id}:{group_id or f'p:{user_id}'}"


def create_queue(conv_key: str) -> List["MemoryItemUnion"]:
    """创建消息队列。在 Agent 开始处理时调用。"""
    if conv_key not in _pending_messages:
        _pending_messages[conv_key] = []
    return _pending_messages[conv_key]


def get_queue(conv_key: str) -> Optional[List["MemoryItemUnion"]]:
    """获取队列（Agent 正在运行时返回 list，否则返回 None）。"""
    return _pending_messages.get(conv_key)


def remove_queue(conv_key: str) -> Optional[List["MemoryItemUnion"]]:
    """移除并返回队列。在 Agent 处理完成时调用。"""
    return _pending_messages.pop(conv_key, None)
