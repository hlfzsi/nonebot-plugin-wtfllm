import time
from collections import deque
from typing import TypeVar, Generic, Optional, List, Tuple

from ..utils import APP_CONFIG

T = TypeVar("T")


class TTLDeque(Generic[T]):
    """
    Args:
        maxlen (int | None): 队列最大长度，None表示不限制长度
        ttl (int): 元素过期时间，单位秒
    """

    def __init__(
        self,
        maxlen: Optional[int] = None,
        ttl: int | float = APP_CONFIG.message_track_time_minutes * 60,
    ) -> None:
        self.queue: deque[Tuple[float, T]] = deque(maxlen=maxlen)
        self.ttl: int | float = ttl

    def append(self, item: T) -> None:
        """向队列添加一个元素，并记录当前时间戳"""
        self.queue.append((time.time(), item))
        self._cleanup()

    def set_ttl(self, ttl: int | float) -> None:
        """重新设置TTL时间并执行清理"""
        self.ttl = ttl
        self._cleanup()

    def _cleanup(self) -> None:
        """弹出头部所有过期的元素"""
        now: float = time.time()
        while self.queue and (now - self.queue[0][0] > self.ttl):
            self.queue.popleft()

    def get_all(self) -> List[T]:
        """获取当前所有未过期的元素列表"""
        self._cleanup()
        return [item for _, item in self.queue]

    def __len__(self) -> int:
        """返回当前未过期元素的数量"""
        self._cleanup()
        return len(self.queue)

    def __repr__(self) -> str:
        return f"TTLDeque(len={len(self.queue)}, ttl={self.ttl}, maxlen={self.queue.maxlen})"
