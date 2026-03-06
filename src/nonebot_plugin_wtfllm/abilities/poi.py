import time
import threading
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PoiInfo(BaseModel):
    user_id: str
    group_id: Optional[str]
    agent_id: str
    reason: str
    turns: int = 1
    expires_at: float = Field(
        default_factory=lambda: time.time() + 60 * 30
    )  # 默认5分钟后过期

    def is_expired(self) -> bool:
        """检查该关注点是否因时间而过期"""
        return time.time() > self.expires_at

    def consume_turn(self) -> bool:
        """消耗一个回合"""
        if self.turns > 0:
            self.turns -= 1
            return True
        return False


class AttentionRouter:
    def __init__(self) -> None:
        self._poi: Dict[str, PoiInfo] = {}
        self._lock = threading.Lock()

    @staticmethod
    def get_key(user_id: str, agent_id: str, group_id: Optional[str] = None) -> str:
        """生成唯一标识键"""
        if group_id:
            return f"ATTN:G:{group_id}:A:{agent_id}:U:{user_id}"
        else:
            return f"ATTN:A:{agent_id}:U:{user_id}"

    def mark_poi(self, poi: PoiInfo) -> None:
        """记录或更新一个关注点"""
        key = self.get_key(poi.user_id, poi.agent_id, poi.group_id)
        with self._lock:
            self._poi[key] = poi

    def get_poi(
        self, user_id: str, agent_id: str, group_id: Optional[str] = None
    ) -> Optional[PoiInfo]:
        """获取关注点详情，如果已过期则自动删除"""
        key = self.get_key(user_id, agent_id, group_id)
        with self._lock:
            poi = self._poi.get(key)
            if not poi:
                return None

            if poi.is_expired() or poi.turns <= 0:
                del self._poi[key]
                return None
            return poi

    def is_interested(
        self, user_id: str, agent_id: str, group_id: Optional[str] = None
    ) -> bool:
        """快速判断 Agent 是否对当前上下文感兴趣"""
        return self.get_poi(user_id, agent_id, group_id) is not None

    def consume_poi(
        self, user_id: str, agent_id: str, group_id: Optional[str] = None
    ) -> bool:
        """
        当 Agent 进行了一次回复后调用。
        消耗一个回合数。如果回合数耗尽或过期，则移除。
        """
        key = self.get_key(user_id, agent_id, group_id)
        with self._lock:
            poi = self._poi.get(key)
            if not poi:
                return False

            poi.consume_turn()
            if poi.turns < 0 or poi.is_expired():
                del self._poi[key]
                return False
            return True

    def remove_poi(self, user_id: str, agent_id: str, group_id: Optional[str] = None):
        """手动强制清除关注点"""
        key = self.get_key(user_id, agent_id, group_id)
        with self._lock:
            if key in self._poi:
                del self._poi[key]

    def get_and_consume_poi(
        self, user_id: str, agent_id: str, group_id: Optional[str] = None
    ) -> Optional[PoiInfo]:
        """
        获取关注点详情并消耗一个回合数。
        如果回合数耗尽或过期，则移除并返回 None。
        """
        key = self.get_key(user_id, agent_id, group_id)
        with self._lock:
            poi = self._poi.get(key)
            if not poi:
                return None

            if poi.turns <= 0 or poi.is_expired():
                del self._poi[key]
                return None
            poi.consume_turn()
            return poi

    def list_agent_interests(self, agent_id: str) -> List[PoiInfo]:
        """列出某个 Agent 当前所有活跃的关注点"""
        self._cleanup()
        with self._lock:
            return [poi for poi in self._poi.values() if poi.agent_id == agent_id]

    def _cleanup(self):
        """内部清理方法，移除所有过期的关注点"""
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, v in self._poi.items() if v.expires_at < now or v.turns <= 0
            ]
            for k in expired_keys:
                del self._poi[k]

    def clear_all(self):
        """清空所有数据"""
        with self._lock:
            self._poi.clear()
