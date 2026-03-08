from dataclasses import dataclass


@dataclass(slots=True)
class SessionKey:
    agent_id: str
    user_id: str
    group_id: str | None

    @property
    def available_id(self) -> tuple[str, str]:
        return self.agent_id, self.group_id or self.user_id
