import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class TopicInterest:
    agent_id: str
    user_id: str
    group_id: str | None
    topics: tuple[str, ...]
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 60 * 30)

    def is_expired(self) -> bool:
        return time.time() > self.expires_at