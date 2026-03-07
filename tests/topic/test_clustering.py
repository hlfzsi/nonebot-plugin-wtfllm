"""质心最近邻聚类引擎单元测试"""

import time

import numpy as np
import pytest

from nonebot_plugin_wtfllm.topic._types import (
    SessionKey,
    TopicCluster,
    TopicSessionState,
)
from nonebot_plugin_wtfllm.topic.clustering import TopicClustering
from nonebot_plugin_wtfllm.vec import TopicVectorizer


@pytest.fixture
def vectorizer():
    return TopicVectorizer()


@pytest.fixture
def clustering():
    return TopicClustering(threshold=0.5, max_clusters=10, decay_seconds=60)


def _make_state(key: SessionKey | None = None) -> TopicSessionState:
    if key is None:
        key = SessionKey(agent_id="a1", group_id="g1")
    return TopicSessionState(session_key=key)


_DEFAULT_KEY = SessionKey(agent_id="a1", group_id="g1")


class TestTopicClustering:

    def test_first_message_initializes(self, clustering: TopicClustering, vectorizer: TopicVectorizer):
        vec = vectorizer.transform("你好世界")
        state = _make_state()
        label, candidate = clustering.assign(vec, state=state, session_key=_DEFAULT_KEY)
        assert isinstance(label, int)
        assert label >= 0
        assert clustering.n_clusters > 0
        assert candidate is None

    def test_predict_only_before_fit_returns_minus_one(self, clustering: TopicClustering, vectorizer: TopicVectorizer):
        vec = vectorizer.transform("测试文本")
        assert clustering.predict_only(vec) == -1

    def test_predict_only_after_fit(self, clustering: TopicClustering, vectorizer: TopicVectorizer):
        vec = vectorizer.transform("初始化文本")
        state = _make_state()
        clustering.assign(vec, state=state, session_key=_DEFAULT_KEY)
        label = clustering.predict_only(vec)
        assert isinstance(label, int)
        assert label >= 0

    def test_similar_messages_same_cluster(self, vectorizer: TopicVectorizer):
        """高度相似的消息应归入同一簇"""
        clustering = TopicClustering(threshold=0.85, max_clusters=10, decay_seconds=60)
        state = _make_state()
        msgs = [
            "今天天气真好啊非常舒服",
            "今天天气不错很舒服",
            "今天的天气真好非常舒适",
            "今天天气真好适合出门",
            "今天天气好得很啊",
        ]
        labels = []
        for msg in msgs:
            vec = vectorizer.transform(msg)
            label, _ = clustering.assign(vec, state=state, session_key=_DEFAULT_KEY)
            labels.append(label)
        # 高度相似消息应至少部分归入同一簇
        assert len(set(labels)) < len(labels)

    def test_dissimilar_messages_different_clusters(self, vectorizer: TopicVectorizer):
        """截然不同的消息应归入不同簇"""
        clustering = TopicClustering(threshold=0.5, max_clusters=10)
        state = _make_state()
        msgs_food = ["今天吃了红烧肉很好吃", "晚餐吃了麻辣火锅", "中午吃了清蒸鲈鱼"]
        msgs_tech = ["Python编程语言很强大", "JavaScript前端开发框架", "Linux系统管理员配置"]

        food_labels = set()
        tech_labels = set()
        for msg in msgs_food:
            label, _ = clustering.assign(vectorizer.transform(msg), state=state, session_key=_DEFAULT_KEY)
            food_labels.add(label)
        for msg in msgs_tech:
            label, _ = clustering.assign(vectorizer.transform(msg), state=state, session_key=_DEFAULT_KEY)
            tech_labels.add(label)

        # 两组消息不应完全重叠
        assert food_labels != tech_labels

    def test_n_clusters_increases(self, clustering: TopicClustering, vectorizer: TopicVectorizer):
        state = _make_state()
        clustering.assign(vectorizer.transform("第一个话题关于美食"), state=state, session_key=_DEFAULT_KEY)
        n1 = clustering.n_clusters
        clustering.assign(vectorizer.transform("完全不同的话题是编程"), state=state, session_key=_DEFAULT_KEY)
        n2 = clustering.n_clusters
        assert n2 >= n1

    def test_labels_are_stable(self, vectorizer: TopicVectorizer):
        """标签应单调递增且不变"""
        clustering = TopicClustering(threshold=0.3, max_clusters=100)
        state = _make_state()
        label1, _ = clustering.assign(vectorizer.transform("美食话题红烧肉"), state=state, session_key=_DEFAULT_KEY)
        label2, _ = clustering.assign(vectorizer.transform("编程话题Python"), state=state, session_key=_DEFAULT_KEY)
        assert label1 == 0
        assert label2 >= 0
        # 再次 assign 相同话题应返回相同 label
        label3, _ = clustering.assign(vectorizer.transform("美食话题糖醋排骨"), state=state, session_key=_DEFAULT_KEY)
        assert label3 == label1 or label3 == label2

    def test_max_clusters_cap(self, vectorizer: TopicVectorizer):
        """达到 max_clusters 后通过淘汰保持不超限"""
        clustering = TopicClustering(threshold=0.99, max_clusters=3)
        state = _make_state()
        for i in range(10):
            clustering.assign(
                vectorizer.transform(f"完全不同的话题编号{i}关于{chr(0x4e00 + i * 100)}"),
                state=state, session_key=_DEFAULT_KEY,
            )
        assert clustering.n_clusters <= 3

    def test_prune_stale_topics(self, clustering: TopicClustering, vectorizer: TopicVectorizer):
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)

        # assign 一条消息创建内部质心
        vec = vectorizer.transform("测试话题")
        label, _ = clustering.assign(vec, state=state, session_key=key)

        # 添加一个过期簇
        stale_cluster = TopicCluster(label=label)
        stale_cluster.last_active_at = time.time() - 120  # 120s ago > decay_seconds=60
        state.clusters[label] = stale_cluster

        # 添加活跃簇
        for i in range(1, 4):
            active_cluster = TopicCluster(label=i + 100)
            active_cluster.last_active_at = time.time()
            state.clusters[i + 100] = active_cluster

        pruned, candidates = clustering.prune_stale_topics(state, session_key=key)
        assert label in pruned
        assert label not in state.clusters
        # 内部质心和 _last_active 也应被清理
        assert label not in clustering._centroids
        assert label not in clustering._last_active


def _make_vec(dim: int, index: int) -> np.ndarray:
    """创建正交单位向量（1×dim），用于测试中保证不同簇不匹配"""
    v = np.zeros((1, dim), dtype=np.float64)
    v[0, index] = 1.0
    return v


class TestEviction:
    """DenStream 时间衰减淘汰机制测试"""

    def test_eviction_at_capacity_creates_new_cluster(self):
        """容量满时应淘汰最弱簇并创建新簇，而非 force-assign"""
        clustering = TopicClustering(threshold=0.99, max_clusters=3, decay_seconds=60)
        state = _make_state()
        now = 1000.0

        for i in range(3):
            clustering.assign(_make_vec(512, i), state=state, session_key=_DEFAULT_KEY, now=now + i)
        assert clustering.n_clusters == 3

        # 第 4 个不同向量 — 应淘汰最弱并创建新簇
        label4, candidate = clustering.assign(_make_vec(512, 3), state=state, session_key=_DEFAULT_KEY, now=now + 100)
        assert clustering.n_clusters == 3  # 仍为 3（淘汰 1 + 创建 1）
        assert label4 == 3  # 新标签（单调递增）

    def test_eviction_prefers_oldest(self):
        """同 count 下应优先淘汰最旧（last_active 最早）的簇"""
        clustering = TopicClustering(threshold=0.99, max_clusters=3, decay_seconds=3600)
        state = _make_state()

        # 3 个簇，各 1 条消息，last_active 不同
        for i, t in enumerate([100.0, 200.0, 300.0]):
            clustering.assign(_make_vec(512, i), state=state, session_key=_DEFAULT_KEY, now=t)

        # 标签 0 最旧 (t=100)，应被淘汰
        clustering.assign(_make_vec(512, 3), state=state, session_key=_DEFAULT_KEY, now=400.0)

        assert 0 not in clustering._centroids
        assert 1 in clustering._centroids
        assert 2 in clustering._centroids

    def test_eviction_prefers_smallest(self):
        """同时间下应优先淘汰 count 最小的簇"""
        clustering = TopicClustering(threshold=0.5, max_clusters=3, decay_seconds=3600)
        state = _make_state()
        now = 1000.0

        # 簇 0: 5 条消息
        for _ in range(5):
            clustering.assign(_make_vec(512, 0), state=state, session_key=_DEFAULT_KEY, now=now)

        # 簇 1: 1 条消息
        clustering.assign(_make_vec(512, 1), state=state, session_key=_DEFAULT_KEY, now=now)

        # 簇 2: 3 条消息
        for _ in range(3):
            clustering.assign(_make_vec(512, 2), state=state, session_key=_DEFAULT_KEY, now=now)

        assert clustering.n_clusters == 3

        # 新消息 — 簇 1 (count=1) 应被淘汰
        clustering.assign(_make_vec(512, 3), state=state, session_key=_DEFAULT_KEY, now=now)

        assert 0 in clustering._centroids  # count=5, 存活
        assert 1 not in clustering._centroids  # count=1, 淘汰
        assert 2 in clustering._centroids  # count=3, 存活

    def test_decayed_weight_formula(self):
        """验证衰减权重公式 weight × 2^(-Δt / half_life)"""
        clustering = TopicClustering(
            threshold=0.99, max_clusters=10, decay_seconds=120
        )
        state = _make_state()
        # half_life = 120 / 4 = 30s

        clustering.assign(_make_vec(512, 0), state=state, session_key=_DEFAULT_KEY, now=0.0)

        # t=0: weight = min(1, 20) * 2^0 = 1.0
        assert abs(clustering._decayed_weight(0, 0.0) - 1.0) < 1e-6

        # t=30 (one half-life): weight = 1 * 0.5 = 0.5
        assert abs(clustering._decayed_weight(0, 30.0) - 0.5) < 1e-6

        # t=60 (two half-lives): weight = 1 * 0.25 = 0.25
        assert abs(clustering._decayed_weight(0, 60.0) - 0.25) < 1e-6


class TestEvictionReturnsCandidate:
    """验证淘汰时返回 ArchivalCandidate"""

    def test_eviction_returns_candidate_when_qualified(self):
        """消息数 >= min_archive_messages 时应返回候选"""
        clustering = TopicClustering(threshold=0.99, max_clusters=2, decay_seconds=3600)
        state = _make_state()
        key = _DEFAULT_KEY

        # 簇 0: 1 条消息 (旧)
        clustering.assign(_make_vec(512, 0), state=state, session_key=key, now=100.0)
        state.clusters[0] = TopicCluster(label=0, message_count=1, last_active_at=100.0)

        # 簇 1: 5 条消息 (新但会因 count 少于簇0 而被保留? 不，需要精确控制)
        # 让簇1也在同一时间but少消息 → 簇1 decayed weight更低
        clustering.assign(_make_vec(512, 1), state=state, session_key=key, now=100.0)
        state.clusters[1] = TopicCluster(label=1, message_count=15, last_active_at=100.0)
        # 手动设置 _counts 来确保簇 0 (count=1) 比簇 1 (count=5) 更弱
        clustering._counts[1] = 5

        # 第 3 个向量 — 淘汰 decayed_weight 最低的簇
        # 簇 0: count=1 * 2^(-(200-100)/900) ≈ 0.926
        # 簇 1: count=5 * 2^(-(200-100)/900) ≈ 4.63
        # 簇 0 更弱，被淘汰。但 state.clusters[0].message_count=1 < 3, 所以无候选
        # 改为让被淘汰的簇有足够消息
        state.clusters[0] = TopicCluster(label=0, message_count=5, last_active_at=100.0)

        label, candidate = clustering.assign(
            _make_vec(512, 2), state=state, session_key=key,
            min_archive_messages=3, now=200.0,
        )

        # 簇 0 有 5 条消息 >= 3，且是最弱的，应被淘汰并产生候选
        assert candidate is not None
        assert candidate.session_key == key
        assert candidate.cluster.message_count == 5

    def test_eviction_returns_none_when_below_threshold(self):
        """消息数 < min_archive_messages 时不应返回候选"""
        clustering = TopicClustering(threshold=0.99, max_clusters=2, decay_seconds=3600)
        state = _make_state()
        key = _DEFAULT_KEY

        # 簇 0: 1 条消息 (旧)
        clustering.assign(_make_vec(512, 0), state=state, session_key=key, now=100.0)
        state.clusters[0] = TopicCluster(label=0, message_count=1, last_active_at=100.0)

        # 簇 1: 更多消息 (新)
        clustering.assign(_make_vec(512, 1), state=state, session_key=key, now=100.0)
        state.clusters[1] = TopicCluster(label=1, message_count=5, last_active_at=100.0)
        clustering._counts[1] = 5

        # 簇 0 (count=1) 最弱被淘汰，但 message_count=1 < 10 → 无候选
        label, candidate = clustering.assign(
            _make_vec(512, 2), state=state, session_key=key,
            min_archive_messages=10, now=200.0,
        )
        assert candidate is None


class TestPruneReturnsCandidate:
    """验证 prune_stale_topics 返回归档候选"""

    def test_prune_returns_candidates_for_qualified_clusters(self):
        clustering = TopicClustering(threshold=0.5, max_clusters=10, decay_seconds=60)
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)

        # 创建质心
        clustering.assign(_make_vec(512, 0), state=state, session_key=key, now=0.0)

        # 添加过期簇 with enough messages
        stale = TopicCluster(label=0, message_count=15)
        stale.last_active_at = time.time() - 120
        stale.message_entries = [(f"msg_{i}", 0.0) for i in range(15)]
        state.clusters[0] = stale

        pruned, candidates = clustering.prune_stale_topics(
            state, session_key=key, min_archive_messages=10
        )
        assert 0 in pruned
        assert len(candidates) == 1
        assert candidates[0].cluster.message_count == 15

    def test_prune_skips_unqualified_clusters(self):
        clustering = TopicClustering(threshold=0.5, max_clusters=10, decay_seconds=60)
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)

        clustering.assign(_make_vec(512, 0), state=state, session_key=key, now=0.0)

        # 过期簇 but too few messages
        stale = TopicCluster(label=0, message_count=2)
        stale.last_active_at = time.time() - 120
        state.clusters[0] = stale

        pruned, candidates = clustering.prune_stale_topics(
            state, session_key=key, min_archive_messages=10
        )
        assert 0 in pruned
        assert len(candidates) == 0


class TestInertia:
    """大簇惯性机制测试：簇越大准入阈值越高"""

    def test_effective_threshold_increases_with_count(self):
        """簇消息越多，effective_threshold 越高"""
        clustering = TopicClustering(threshold=0.65, max_clusters=10)
        clustering._counts[0] = 1
        t1 = clustering._effective_threshold(0)

        clustering._counts[0] = 10
        t10 = clustering._effective_threshold(0)

        clustering._counts[0] = 20
        t20 = clustering._effective_threshold(0)

        assert t1 < t10 < t20
        assert abs(t20 - 0.80) < 1e-6  # 0.65 + 0.15

    def test_effective_threshold_caps_at_weight_cap(self):
        """count 超过 WEIGHT_CAP 后 effective_threshold 不再增长"""
        clustering = TopicClustering(threshold=0.65, max_clusters=10)
        clustering._counts[0] = 20
        t20 = clustering._effective_threshold(0)

        clustering._counts[0] = 100
        t100 = clustering._effective_threshold(0)

        assert abs(t20 - t100) < 1e-6

    def test_inertia_prevents_black_hole(self, vectorizer: TopicVectorizer):
        """大簇应因惯性拒绝边缘匹配，新消息形成独立小簇"""
        clustering = TopicClustering(
            threshold=0.5, max_clusters=100, decay_seconds=3600
        )
        state = _make_state()
        # 先喂 25 条相同文本让簇 0 长大
        for _ in range(25):
            clustering.assign(vectorizer.transform("今天天气真好"), state=state, session_key=_DEFAULT_KEY, now=1000.0)
        assert clustering.n_clusters == 1
        assert clustering._counts[0] == 25

        # 簇 0 的 effective_threshold 应显著高于 base
        eff = clustering._effective_threshold(0)
        assert eff > clustering._threshold + 0.1

    def test_predict_only_uses_base_threshold(self):
        """predict_only 应使用基础阈值，不受惯性影响"""
        clustering = TopicClustering(threshold=0.5, max_clusters=10)
        state = _make_state()
        # 创建一个大簇
        vec = _make_vec(512, 0)
        for _ in range(25):
            clustering.assign(vec, state=state, session_key=_DEFAULT_KEY, now=1000.0)

        # predict_only 用 base threshold，同向量应匹配
        label = clustering.predict_only(vec)
        assert label == 0
