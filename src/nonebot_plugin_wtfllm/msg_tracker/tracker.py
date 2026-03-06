from typing import Dict, List
from cachetools import LRUCache

from .cache_queue import TTLDeque
from ..utils import APP_CONFIG


class MsgTracker:
    def __init__(
        self, maxsize: int = 3600, ttl: int = APP_CONFIG.message_track_time_minutes * 60
    ) -> None:
        self._cache: LRUCache[str, LRUCache[str, TTLDeque[str]]] = LRUCache(
            maxsize=maxsize
        )
        self.ttl = ttl

    @staticmethod
    def _get_main_key(agent_id: str, user_id: str) -> str:
        return f"{agent_id}:{user_id}"

    @staticmethod
    def _gat_second_key(group_id: str | None) -> str:
        return group_id or ""

    def track(
        self, agent_id: str, user_id: str, group_id: str | None, msg_id: str
    ) -> None:
        """
        Args:
            agent_id: 智能体ID
            user_id: 当前服务用户ID
            group_id: 群组ID
            msg_id: 消息ID
        """
        main_key = self._get_main_key(agent_id, user_id)
        second_key = self._gat_second_key(group_id)

        if main_key not in self._cache:
            self._cache[main_key] = LRUCache(maxsize=5)

        if second_key not in self._cache[main_key]:
            self._cache[main_key][second_key] = TTLDeque(ttl=self.ttl)

        self._cache[main_key][second_key].append(msg_id)

    def get(self, user_id: str, agent_id: str) -> Dict[str, List[str]]:
        """
        Returns:
            Dict[str, List[str]]: 键为second_key（群ID或""），值为对应的消息ID列表
        """
        main_key = self._get_main_key(agent_id, user_id)
        result: Dict[str, List[str]] = {}
        if main_key in self._cache:
            for second_key, ttl_deque in self._cache[main_key].items():
                result[second_key] = ttl_deque.get_all()
        return result
