import threading
import time
from typing import Sequence

from ._models import TopicInterest


class TopicInterestStore:
    def __init__(self) -> None:
        self._items: dict[str, TopicInterest] = {}
        self._lock = threading.Lock()

    @staticmethod
    def get_key(user_id: str, agent_id: str, group_id: str | None = None) -> str:
        if group_id:
            return f"TOPIC:G:{group_id}:A:{agent_id}"
        return f"TOPIC:A:{agent_id}:U:{user_id}"

    @staticmethod
    def _normalize_topics(topics: Sequence[str] | None) -> tuple[str, ...]:
        if not topics:
            return ()

        normalized: list[str] = []
        for topic in topics:
            stripped = topic.strip()
            if stripped and stripped not in normalized:
                normalized.append(stripped)
        return tuple(normalized)

    def set_topics(
        self,
        *,
        agent_id: str,
        user_id: str,
        group_id: str | None,
        topics: Sequence[str] | None,
        ttl_seconds: float = 60 * 10,
    ) -> None:
        key = self.get_key(user_id=user_id, agent_id=agent_id, group_id=group_id)
        normalized_topics = self._normalize_topics(topics)

        with self._lock:
            if not normalized_topics or ttl_seconds <= 0:
                self._items.pop(key, None)
                return

            self._items[key] = TopicInterest(
                agent_id=agent_id,
                user_id=user_id,
                group_id=group_id,
                topics=normalized_topics,
                expires_at=time.time() + ttl_seconds,
            )

    def get_interest(
        self,
        *,
        agent_id: str,
        user_id: str,
        group_id: str | None,
    ) -> TopicInterest | None:
        key = self.get_key(user_id=user_id, agent_id=agent_id, group_id=group_id)
        with self._lock:
            interest = self._items.get(key)
            if interest is None:
                return None
            if interest.is_expired():
                del self._items[key]
                return None
            return interest

    def clear_topics(
        self,
        *,
        agent_id: str,
        user_id: str,
        group_id: str | None,
    ) -> None:
        key = self.get_key(user_id=user_id, agent_id=agent_id, group_id=group_id)
        with self._lock:
            self._items.pop(key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._items.clear()


topic_interest_store = TopicInterestStore()