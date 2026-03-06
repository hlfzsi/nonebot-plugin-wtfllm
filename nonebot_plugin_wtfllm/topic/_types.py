"""话题聚类数据模型"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class SessionKey:
    """会话标识，per-session 话题状态的键"""

    agent_id: str
    group_id: Optional[str] = None
    user_id: Optional[str] = None

    @property
    def cache_key(self) -> str:
        if self.group_id:
            return f"{self.agent_id}:g:{self.group_id}"
        return f"{self.agent_id}:u:{self.user_id}"

    def __hash__(self) -> int:
        return hash((self.agent_id, self.group_id, self.user_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionKey):
            return NotImplemented
        return (
            self.agent_id == other.agent_id
            and self.group_id == other.group_id
            and self.user_id == other.user_id
        )


@dataclass(slots=True)
class TopicCluster:
    """检测到的话题簇"""

    label: int
    message_entries: list[tuple[str, float]] = field(default_factory=list)
    """(message_id, created_at) 列表"""
    last_active_at: float = field(default_factory=time.time)
    message_count: int = 0


@dataclass(slots=True)
class TopicSessionState:
    """per-session 话题追踪状态，纯内存，重启丢失"""

    session_key: SessionKey
    clusters: dict[int, TopicCluster] = field(default_factory=dict)
    total_messages_ingested: int = 0
    message_to_label: dict[str, int] = field(default_factory=dict)

    def remove_cluster_index(self, label: int) -> None:
        """清理指定簇在 message_to_label 中的全部条目"""
        cluster = self.clusters.get(label)
        if cluster:
            for msg_id, _ in cluster.message_entries:
                self.message_to_label.pop(msg_id, None)


@dataclass(slots=True)
class ArchivalCandidate:
    """即将被移除的簇的归档快照"""

    session_key: SessionKey
    cluster: TopicCluster
    centroid: NDArray[np.floating]
