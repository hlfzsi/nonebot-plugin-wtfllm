import asyncio
import time
from typing import Any, Callable, Coroutine

from cachetools import LRUCache

from ._types import ArchivalCandidate, SessionKey, TopicCluster, TopicSessionState
from .clustering.engine import TopicClustering
from .clustering.vectorizer import vectorizer
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
        self.lock = asyncio.Lock()


class TopicManager:
    def __init__(
        self,
        maxsize: int = 500,
        cluster_threshold: float = 0.5,
        max_clusters: int = 30,
        decay_seconds: float = 7200,
        max_messages_per_cluster: int = 200,
        min_archive_messages: int = 10,
        centroid_ema_alpha: float = 0.5,
    ) -> None:
        self._sessions: LRUCache[str, _SessionContext] = LRUCache(maxsize=maxsize)
        self._lock = asyncio.Lock()
        self._vectorizer = vectorizer
        self._cluster_threshold = cluster_threshold
        self._max_clusters = max_clusters
        self._decay_seconds = decay_seconds
        self._max_messages_per_cluster = max_messages_per_cluster
        self._min_archive_messages = min_archive_messages
        self._centroid_ema_alpha = centroid_ema_alpha

        self._archive_queue: asyncio.Queue[ArchivalCandidate] = asyncio.Queue()
        self._archive_worker: asyncio.Task[None] | None = None
        self._prune_worker: asyncio.Task[None] | None = None
        self._archive_handlers: (
            list[Callable[[ArchivalCandidate], Coroutine[Any, Any, None]]] | None
        ) = None

    def start(
        self,
        handler: Callable[[ArchivalCandidate], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """启动后台归档 worker 和定时清理 worker。"""
        if self._archive_handlers is None:
            self._archive_handlers = []
        if handler is not None:
            self._archive_handlers.append(handler)
        if self._archive_worker is None:
            self._archive_worker = asyncio.create_task(self._archive_consumer())
        if self._prune_worker is None:
            self._prune_worker = asyncio.create_task(self._periodic_prune())

    async def stop(self) -> None:
        """停止归档 worker 和定时清理 worker。"""
        for task in (self._archive_worker, self._prune_worker):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._archive_worker = None
        self._prune_worker = None

    async def _archive_consumer(self) -> None:
        """后台消费者：从队列取候选并执行归档。"""
        while True:
            candidate = await self._archive_queue.get()
            try:
                if self._archive_handlers is not None:
                    await asyncio.gather(
                        *[handler(candidate) for handler in self._archive_handlers]
                    )
            except Exception:
                logger.opt(exception=True).warning("cluster archival failed")
            finally:
                self._archive_queue.task_done()

    async def _periodic_prune(self) -> None:
        """定时扫描所有会话，清理过期簇。

        每 decay_seconds / 2 执行一次，确保静默期内过期簇也能被及时清理。
        """
        interval = max(self._decay_seconds / 2, 60)
        while True:
            await asyncio.sleep(interval)
            try:
                async with self._lock:
                    sessions_snapshot = list(self._sessions.items())
                for cache_key, ctx in sessions_snapshot:
                    async with ctx.lock:
                        _pruned, prune_candidates = ctx.clustering.prune_stale_topics(
                            ctx.state,
                            session_key=ctx.state.session_key,
                            min_archive_messages=self._min_archive_messages,
                        )
                        for candidate in prune_candidates:
                            self._archive_queue.put_nowait(candidate)
            except Exception:
                logger.opt(exception=True).warning("periodic prune failed")

    def _get_or_create(self, key: SessionKey) -> _SessionContext:
        cache_key = key.cache_key
        if cache_key not in self._sessions:
            ctx = _SessionContext(
                state=TopicSessionState(session_key=key),
                clustering=TopicClustering(
                    threshold=self._cluster_threshold,
                    max_clusters=self._max_clusters,
                    decay_seconds=self._decay_seconds,
                    ema_alpha=self._centroid_ema_alpha,
                ),
            )
            self._sessions[cache_key] = ctx
        return self._sessions[cache_key]

    async def ingest(
        self,
        agent_id: str,
        group_id: str | None,
        user_id: str | None,
        message_id: str,
        plain_text: str,
        related_message_id: str | None = None,
    ) -> int:
        """摄入新消息到话题系统，返回分配的簇标签。

        若 related_message_id 对应的消息已在某个簇中，则继承该簇。
        """
        if not plain_text or len(plain_text.strip()) <= 2:
            return -1

        key = SessionKey(agent_id=agent_id, group_id=group_id, user_id=user_id)
        async with self._lock:
            ctx = self._get_or_create(key)

        async with ctx.lock:
            state = ctx.state
            now = time.time()

            inherited_label: int | None = None
            if related_message_id is not None:
                candidate_label = state.message_to_label.get(related_message_id)
                if candidate_label is not None and candidate_label in state.clusters:
                    inherited_label = candidate_label

            feature_vector = await asyncio.to_thread(
                self._vectorizer.transform, plain_text
            )

            if inherited_label is not None:
                label = inherited_label
                eviction_candidate = None
                ctx.clustering.update_centroid_external(
                    label, feature_vector, now=now
                )
            else:
                label, eviction_candidate = ctx.clustering.assign(
                    feature_vector,
                    state=state,
                    session_key=key,
                    min_archive_messages=self._min_archive_messages,
                    now=now,
                )

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

            state.message_to_label[message_id] = label
            state.total_messages_ingested += 1

            # 入队淘汰候选
            if eviction_candidate is not None:
                self._archive_queue.put_nowait(eviction_candidate)

            return label

    async def query_topic(
        self,
        agent_id: str,
        group_id: str | None,
        user_id: str | None,
        query: str,
        max_count: int = 20,
        before_timestamp: float | None = None,
    ) -> tuple[int, list[str]]:
        """对 query 文本分类到已有簇，返回 (label, message_ids)。

        纯只读操作，不修改模型状态。
        """
        key = SessionKey(agent_id=agent_id, group_id=group_id, user_id=user_id)
        cache_key = key.cache_key
        async with self._lock:
            if cache_key not in self._sessions:
                return (-1, [])
            ctx = self._sessions[cache_key]

        async with ctx.lock:
            feature_vector = await asyncio.to_thread(self._vectorizer.transform, query)
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

    async def get_active_topics_summary(
        self,
        agent_id: str,
        group_id: str | None,
        user_id: str | None,
    ) -> list[dict[str, Any]]:
        """获取会话所有活跃话题摘要（调试/监控用）"""
        logger.warning("get_active_topics_summary 只应在调试时使用")
        key = SessionKey(agent_id=agent_id, group_id=group_id, user_id=user_id)
        cache_key = key.cache_key
        async with self._lock:
            if cache_key not in self._sessions:
                return []
            ctx = self._sessions[cache_key]

        async with ctx.lock:
            state = ctx.state
            summaries: list[dict[str, Any]] = []
            for label, cluster in state.clusters.items():
                summaries.append(
                    {
                        "label": label,
                        "message_count": cluster.message_count,
                        "last_active_at": cluster.last_active_at,
                    }
                )
            return summaries
