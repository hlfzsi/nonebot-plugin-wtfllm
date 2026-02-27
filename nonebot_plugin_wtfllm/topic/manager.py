import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from cachetools import LRUCache

from ._types import SessionKey, TopicCluster, TopicSessionState
from .clustering import TopicClustering
from .vectorizer import TopicVectorizer
from ..utils import logger


class _SessionContext:
    """绑定 per-session 状态与其聚类模型"""

    __slots__ = ("state", "clustering", "lock")

    def __init__(
        self,
        state: TopicSessionState,
        clustering: TopicClustering,
    ) -> None:
        self.state = state
        self.clustering = clustering
        self.lock = threading.Lock()


class TopicManager:
    def __init__(
        self,
        maxsize: int = 500,
        cluster_threshold: float = 0.5,
        max_clusters: int = 30,
        decay_seconds: float = 7200,
        maintenance_interval: int = 100,
        max_messages_per_cluster: int = 200,
    ) -> None:
        self._sessions: LRUCache[str, _SessionContext] = LRUCache(maxsize=maxsize)
        self._lock = threading.Lock()  # 保护 _sessions 字典访问
        self._vectorizer = TopicVectorizer()
        self._cluster_threshold = cluster_threshold
        self._max_clusters = max_clusters
        self._decay_seconds = decay_seconds
        self._maintenance_interval = maintenance_interval
        self._max_messages_per_cluster = max_messages_per_cluster

    def _get_or_create(self, key: SessionKey) -> _SessionContext:
        cache_key = key.cache_key
        if cache_key not in self._sessions:
            ctx = _SessionContext(
                state=TopicSessionState(session_key=key),
                clustering=TopicClustering(
                    threshold=self._cluster_threshold,
                    max_clusters=self._max_clusters,
                    decay_seconds=self._decay_seconds,
                ),
            )
            self._sessions[cache_key] = ctx
        return self._sessions[cache_key]

    def ingest(
        self,
        agent_id: str,
        group_id: Optional[str],
        user_id: Optional[str],
        message_id: str,
        plain_text: str,
    ) -> int:
        """摄入新消息到话题系统，返回分配的簇标签。"""
        if not plain_text or len(plain_text.strip()) <= 2:
            return -1

        key = SessionKey(agent_id=agent_id, group_id=group_id, user_id=user_id)
        with self._lock:
            ctx = self._get_or_create(key)

        with ctx.lock:
            state = ctx.state
            now = time.time()

            feature_vector = self._vectorizer.transform(plain_text)

            label = ctx.clustering.assign(feature_vector, now=now)

            if label not in state.clusters:
                state.clusters[label] = TopicCluster(label=label)

            cluster = state.clusters[label]
            cluster.message_entries.append((message_id, now))
            cluster.last_active_at = now
            cluster.message_count += 1

            if len(cluster.message_entries) > self._max_messages_per_cluster:
                cluster.message_entries = cluster.message_entries[
                    -self._max_messages_per_cluster :
                ]

            state.total_messages_ingested += 1

            if state.total_messages_ingested % self._maintenance_interval == 0:
                self._maintenance(ctx)

            return label

    def query_topic(
        self,
        agent_id: str,
        group_id: Optional[str],
        user_id: Optional[str],
        query: str,
        max_count: int = 20,
        before_timestamp: Optional[float] = None,
    ) -> Tuple[int, List[str]]:
        """对 query 文本分类到已有簇，返回 (label, message_ids)。

        纯只读操作，不修改模型状态。
        """
        key = SessionKey(agent_id=agent_id, group_id=group_id, user_id=user_id)
        cache_key = key.cache_key
        with self._lock:
            if cache_key not in self._sessions:
                return (-1, [])
            ctx = self._sessions[cache_key]

        with ctx.lock:
            feature_vector = self._vectorizer.transform(query)
            label = ctx.clustering.predict_only(feature_vector)
            if label < 0:
                return (-1, [])

            state = ctx.state
            cluster = state.clusters.get(label)
            if not cluster:
                return (-1, [])

            entries = cluster.message_entries
            if before_timestamp is not None:
                entries = [(mid, ts) for mid, ts in entries if ts < before_timestamp]
            message_ids = [mid for mid, _ in entries[-max_count:]]

            return (label, message_ids)

    def get_active_topics_summary(
        self,
        agent_id: str,
        group_id: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """获取会话所有活跃话题摘要（调试/监控用）"""
        logger.warning("get_active_topics_summary 只应在调试时使用")
        key = SessionKey(agent_id=agent_id, group_id=group_id, user_id=user_id)
        cache_key = key.cache_key
        with self._lock:
            if cache_key not in self._sessions:
                return []
            ctx = self._sessions[cache_key]

        with ctx.lock:
            state = ctx.state
            summaries: List[Dict[str, Any]] = []
            for label, cluster in state.clusters.items():
                summaries.append(
                    {
                        "label": label,
                        "message_count": cluster.message_count,
                        "last_active_at": cluster.last_active_at,
                    }
                )
            return summaries

    def _maintenance(self, ctx: _SessionContext) -> None:
        """周期性维护：清理过期话题"""
        if ctx.clustering.needs_pruning(ctx.state):
            ctx.clustering.prune_stale_topics(ctx.state)
