import time

import numpy as np
from numpy.typing import NDArray

from ._types import TopicSessionState


class TopicClustering:
    """质心最近邻聚类（Leader Algorithm 变体 + DenStream 时间衰减）

    每个实例服务一个会话。直接在余弦空间操作：
    - threshold 即余弦相似度阈值
    - 标签单调递增，永不重编号
    - 质心增量更新，无需 rebuild
    - 容量满时淘汰 decayed_weight 最低的簇（而非 force-assign）
    """

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
        self, feature_vector: NDArray[np.floating], now: float | None = None
    ) -> int:
        """分配单条消息到簇，返回簇标签。"""
        if now is None:
            now = time.time()
        vec = feature_vector.flatten()

        if not self._centroids:
            return self._create_cluster(vec, now)

        best_label, best_sim = self._find_nearest(vec)

        if best_sim >= self._effective_threshold(best_label):
            self._update_centroid(best_label, vec)
            self._last_active[best_label] = now
            return best_label

        # 需要新簇 — 容量满时淘汰衰减权重最低的簇
        if len(self._centroids) >= self._max_clusters:
            self._evict_weakest(now)

        return self._create_cluster(vec, now)

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

    def needs_pruning(self, state: TopicSessionState) -> bool:
        """检查是否需要清理过期话题"""
        now = time.time()
        return any(
            (now - c.last_active_at) > self.decay_seconds
            for c in state.clusters.values()
        )

    def prune_stale_topics(self, state: TopicSessionState) -> list[int]:
        """从会话状态中移除过期话题簇，同步清理内部质心。

        返回被清理的簇标签列表。
        """
        now = time.time()
        pruned: list[int] = []
        for label in list(state.clusters.keys()):
            cluster = state.clusters[label]
            if (now - cluster.last_active_at) > self.decay_seconds:
                pruned.append(label)
                del state.clusters[label]
                self._centroids.pop(label, None)
                self._counts.pop(label, None)
                self._last_active.pop(label, None)
        return pruned

    def _effective_threshold(self, label: int) -> float:
        """大簇惯性：簇越大，准入阈值越高，防止黑洞吞噬。"""
        count = self._counts.get(label, 0)
        ratio = min(count / self._CENTROID_WEIGHT_CAP, 1.0)
        return self._threshold + self._INERTIA_ALPHA * ratio

    def _decayed_weight(self, label: int, now: float) -> float:
        """DenStream fading function: weight × 2^(-Δt / half_life)"""
        dt = now - self._last_active.get(label, now)
        weight = min(self._counts[label], self._CENTROID_WEIGHT_CAP)
        return weight * (2.0 ** (-dt / self._decay_half_life))

    def _evict_weakest(self, now: float) -> None:
        """淘汰衰减权重最低的簇，为新话题腾出空间。"""
        weakest = min(
            self._centroids, key=lambda label: self._decayed_weight(label, now)
        )
        del self._centroids[weakest]
        del self._counts[weakest]
        del self._last_active[weakest]

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

    def _create_cluster(
        self, vec: NDArray[np.floating], now: float
    ) -> int:
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
