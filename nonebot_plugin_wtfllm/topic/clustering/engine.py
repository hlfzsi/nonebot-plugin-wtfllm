import copy
import time

import numpy as np
from numpy.typing import NDArray

from .._types import ArchivalCandidate, SessionKey, TopicSessionState


class TopicClustering:
    """质心最近邻聚类"""

    _CENTROID_WEIGHT_CAP: int = 20
    _INERTIA_ALPHA: float = 0.15

    def __init__(
        self,
        threshold: float = 0.65,
        max_clusters: int = 30,
        decay_seconds: float = 7200,
    ) -> None:
        self._threshold = threshold
        self._max_clusters = max_clusters
        self.decay_seconds = decay_seconds
        self._decay_half_life: float = decay_seconds / 4.0
        self._centroids: dict[int, NDArray[np.floating]] = {}
        self._counts: dict[int, int] = {}
        self._last_active: dict[int, float] = {}
        self._next_label: int = 0

    def assign(
        self,
        feature_vector: NDArray[np.floating],
        state: TopicSessionState,
        session_key: SessionKey,
        min_archive_messages: int = 10,
        now: float | None = None,
    ) -> tuple[int, ArchivalCandidate | None]:
        """分配单条消息到簇，返回 (簇标签, 可能的淘汰归档候选)。"""
        if now is None:
            now = time.time()
        vec = feature_vector.flatten()

        eviction_candidate: ArchivalCandidate | None = None

        if not self._centroids:
            return self._create_cluster(vec, now), None

        best_label, best_sim = self._find_nearest(vec)

        if best_sim >= self._effective_threshold(best_label):
            self._update_centroid(best_label, vec)
            self._last_active[best_label] = now
            return best_label, None

        # 容量满时淘汰衰减权重最低的簇
        if len(self._centroids) >= self._max_clusters:
            eviction_candidate = self._evict_weakest(
                now, state, session_key, min_archive_messages
            )

        return self._create_cluster(vec, now), eviction_candidate

    def predict_only(self, feature_vector: NDArray[np.floating]) -> int:
        """仅预测簇标签，不更新模型"""
        if not self._centroids:
            return -1
        vec = feature_vector.flatten()
        best_label, best_sim = self._find_nearest(vec)
        return best_label if best_sim >= self._threshold else -1

    @property
    def n_clusters(self) -> int:
        return len(self._centroids)

    def prune_stale_topics(
        self,
        state: TopicSessionState,
        session_key: SessionKey,
        min_archive_messages: int = 10,
    ) -> tuple[list[int], list[ArchivalCandidate]]:
        """从会话状态中移除过期话题簇，清理内部质心。

        返回 (被清理的簇标签列表, 符合归档条件的候选列表)。
        """
        now = time.time()
        pruned: list[int] = []
        candidates: list[ArchivalCandidate] = []
        for label in list(state.clusters.keys()):
            cluster = state.clusters[label]
            if (now - cluster.last_active_at) > self.decay_seconds:
                if cluster.message_count >= min_archive_messages:
                    candidates.append(
                        ArchivalCandidate(
                            session_key=session_key,
                            cluster=copy.deepcopy(cluster),
                            centroid=self._centroids[label].copy(),
                        )
                    )
                pruned.append(label)
                del state.clusters[label]
                self._centroids.pop(label, None)
                self._counts.pop(label, None)
                self._last_active.pop(label, None)
        return pruned, candidates

    def _effective_threshold(self, label: int) -> float:
        """簇越大，准入阈值越高。"""
        count = self._counts.get(label, 0)
        ratio = min(count / self._CENTROID_WEIGHT_CAP, 1.0)
        return self._threshold + self._INERTIA_ALPHA * ratio

    def _decayed_weight(self, label: int, now: float) -> float:
        """DenStream fading function: weight × 2^(-Δt / half_life)"""
        dt = now - self._last_active.get(label, now)
        weight = min(self._counts[label], self._CENTROID_WEIGHT_CAP)
        return weight * (2.0 ** (-dt / self._decay_half_life))

    def _evict_weakest(
        self,
        now: float,
        state: TopicSessionState,
        session_key: SessionKey,
        min_archive_messages: int,
    ) -> ArchivalCandidate | None:
        """淘汰衰减权重最低的簇，为新话题腾出空间。"""
        weakest = min(
            self._centroids, key=lambda label: self._decayed_weight(label, now)
        )
        candidate: ArchivalCandidate | None = None
        cluster = state.clusters.get(weakest)
        if cluster and cluster.message_count >= min_archive_messages:
            candidate = ArchivalCandidate(
                session_key=session_key,
                cluster=copy.deepcopy(cluster),
                centroid=self._centroids[weakest].copy(),
            )
        del self._centroids[weakest]
        del self._counts[weakest]
        del self._last_active[weakest]
        state.clusters.pop(weakest, None)
        return candidate

    def _find_nearest(self, vec: NDArray[np.floating]) -> tuple[int, float]:
        """找到与 vec 余弦相似度最高的质心"""
        best_label = -1
        best_sim = -1.0
        for label, centroid in self._centroids.items():
            sim = float(np.dot(vec, centroid))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label, best_sim

    def _create_cluster(self, vec: NDArray[np.floating], now: float) -> int:
        label = self._next_label
        self._next_label += 1
        self._centroids[label] = vec.copy()
        self._counts[label] = 1
        self._last_active[label] = now
        return label

    def _update_centroid(self, label: int, vec: NDArray[np.floating]) -> None:
        """增量更新质心：加权平均后重新 L2 归一化。

        权重上限为 _CENTROID_WEIGHT_CAP，防止大簇质心僵化。
        """
        effective_count = min(self._counts[label], self._CENTROID_WEIGHT_CAP)
        raw = self._centroids[label] * effective_count + vec
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw /= norm
        self._centroids[label] = raw
        self._counts[label] += 1
