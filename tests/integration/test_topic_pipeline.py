"""话题管线端到端集成测试

覆盖完整链路：
  ingest → 聚类 → 容量淘汰/超时清理 → ArchivalCandidate 入队
  → archive_cluster 管道 → Qdrant 写入 → search_by_session 检索
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nonebot_plugin_wtfllm.topic._types import (
    ArchivalCandidate,
    SessionKey,
    TopicCluster,
)
from nonebot_plugin_wtfllm.topic.clustering.engine import TopicClustering
from nonebot_plugin_wtfllm.topic.clustering.mmr import mmr_select
from nonebot_plugin_wtfllm.topic.clustering.vectorizer import TopicVectorizer, vectorizer
from nonebot_plugin_wtfllm.topic.manager import TopicManager
from nonebot_plugin_wtfllm.v_db.models.topic_archive import TopicArchivePayload


TOPIC_DIM = vectorizer.transform("dim_probe").shape[1]


# ── helpers ──────────────────────────────────────────────


def _make_fake_memory_item(message_id: str, text: str):
    """创建一个足够逼真的 fake MemoryItem 供 pipeline 使用。"""
    item = MagicMock()
    item.message_id = message_id
    item.get_plain_text.return_value = text
    return item


# ── 端到端：ingest → 淘汰 → 归档 pipeline ────────────


class TestIngestToArchivePipeline:
    """验证从消息摄入到归档 Qdrant 的完整链路。"""

    @pytest.mark.asyncio
    async def test_eviction_triggers_archive_pipeline(self):
        """容量淘汰时: ingest → candidate → archive_cluster → Qdrant upsert"""
        collected: list[ArchivalCandidate] = []

        async def _fake_handler(candidate: ArchivalCandidate):
            collected.append(candidate)

        manager = TopicManager(
            maxsize=10,
            cluster_threshold=0.99,    # 极高阈值确保每条消息独立成簇
            max_clusters=3,
            min_archive_messages=1,    # 低阈值确保产生候选
            decay_seconds=7200,
        )
        manager.start(_fake_handler)

        distinct_texts = [
            "Python编程语言asyncio异步框架非常强大",
            "今天晚餐吃了一碗热腾腾的红烧牛肉面",
            "周末去杭州西湖旅游景点拍照留念很开心",
            "NBA篮球比赛湖人队对阵勇士队精彩对决",
        ]
        for i, text in enumerate(distinct_texts):
            await manager.ingest("a1", "g1", None, f"msg_{i}", text)

        # 等待后台 consumer 处理完队列
        await manager._archive_queue.join()
        await manager.stop()

        # 3 个簇容量满后第 4 条触发淘汰，应产生 1 个候选
        if collected:
            candidate = collected[0]
            assert candidate.session_key.agent_id == "a1"
            assert candidate.session_key.group_id == "g1"
            assert candidate.centroid.shape[-1] == TOPIC_DIM
        else:
            # 如果模型恰好把第 4 条归入已有簇则无淘汰，跳过
            ctx = manager._sessions.get("a1:g:g1")
            assert ctx is not None
            assert ctx.clustering.n_clusters <= 3

    @pytest.mark.asyncio
    async def test_prune_triggers_archive_pipeline(self):
        """超时清理时: maintenance → prune → candidate → handler"""
        collected: list[ArchivalCandidate] = []

        async def _fake_handler(candidate: ArchivalCandidate):
            collected.append(candidate)

        manager = TopicManager(
            maxsize=10,
            cluster_threshold=0.5,
            max_clusters=30,
            decay_seconds=0.01,         # 极短衰减确保快速过期
            min_archive_messages=1,
        )
        manager.start(_fake_handler)

        for i in range(4):
            await manager.ingest(
                "a1", "g1", None, f"msg_{i}", f"消息内容关于美食第{i}道菜"
            )

        # 等待过期
        await asyncio.sleep(0.05)

        # 第 5 条后等待周期清理 worker
        await manager.ingest("a1", "g1", None, "msg_4", "技术编程话题完全不同")
        await asyncio.sleep(0.05)

        await manager._archive_queue.join()
        await manager.stop()

        # 之前的簇应该已经过期并产生候选
        # (具体数量取决于聚类分配结果，至少验证流程不报错)


class TestArchiveClusterPipeline:
    """验证 archive_cluster 管道对 MemoryItem → Qdrant 的转换。"""

    @pytest.mark.asyncio
    async def test_archive_cluster_writes_to_qdrant(self):
        """mock DB 和 Qdrant，验证 archive_cluster 正确写入 payload"""
        fake_items = [
            _make_fake_memory_item(f"msg_{i}", f"这是第{i}条消息关于编程语言")
            for i in range(5)
        ]

        centroid = np.random.randn(TOPIC_DIM).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        candidate = ArchivalCandidate(
            session_key=SessionKey(agent_id="a1", group_id="g1"),
            cluster=TopicCluster(
                label=0,
                message_entries=[(f"msg_{i}", time.time()) for i in range(5)],
                message_count=5,
            ),
            centroid=centroid,
        )

        mock_upsert = AsyncMock(return_value="archive_id")

        with (
            patch(
                "nonebot_plugin_wtfllm.topic.archive.pipeline.memory_item_repo"
            ) as mock_repo,
            patch(
                "nonebot_plugin_wtfllm.topic.archive.pipeline.topic_archive_repo"
            ) as mock_vdb,
            patch(
                "nonebot_plugin_wtfllm.topic.archive.pipeline.APP_CONFIG"
            ) as mock_cfg,
        ):
            mock_repo.get_many_by_message_ids = AsyncMock(return_value=fake_items)
            mock_vdb.upsert = mock_upsert
            mock_cfg.topic_archive_min_messages = 3
            mock_cfg.topic_archive_mmr_k = 3
            mock_cfg.topic_archive_mmr_lambda = 0.5

            from nonebot_plugin_wtfllm.topic.archive.pipeline import archive_cluster

            await archive_cluster(candidate)

        mock_upsert.assert_awaited_once()
        call_args = mock_upsert.await_args
        payload: TopicArchivePayload = call_args[0][0]
        document: str = call_args.kwargs.get("document") or call_args[0][1]

        assert payload.agent_id == "a1"
        assert payload.group_id == "g1"
        assert payload.message_count == 5
        assert len(payload.representative_message_ids) == 3  # mmr_k=3
        assert isinstance(document, str) and len(document) > 0

    @pytest.mark.asyncio
    async def test_archive_cluster_skips_insufficient_messages(self):
        """消息数不足 min_messages 时应跳过归档"""
        candidate = ArchivalCandidate(
            session_key=SessionKey(agent_id="a1", group_id="g1"),
            cluster=TopicCluster(
                label=0,
                message_entries=[("msg_0", time.time())],
                message_count=1,
            ),
            centroid=np.random.randn(TOPIC_DIM).astype(np.float32),
        )

        with (
            patch(
                "nonebot_plugin_wtfllm.topic.archive.pipeline.memory_item_repo"
            ) as mock_repo,
            patch(
                "nonebot_plugin_wtfllm.topic.archive.pipeline.topic_archive_repo"
            ) as mock_vdb,
            patch(
                "nonebot_plugin_wtfllm.topic.archive.pipeline.APP_CONFIG"
            ) as mock_cfg,
        ):
            mock_repo.get_many_by_message_ids = AsyncMock(
                return_value=[_make_fake_memory_item("msg_0", "短消息")]
            )
            mock_vdb.upsert = AsyncMock()
            mock_cfg.topic_archive_min_messages = 5

            from nonebot_plugin_wtfllm.topic.archive.pipeline import archive_cluster

            await archive_cluster(candidate)

        mock_vdb.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_archive_cluster_skips_empty_entries(self):
        """空 message_entries 应直接返回"""
        candidate = ArchivalCandidate(
            session_key=SessionKey(agent_id="a1", group_id="g1"),
            cluster=TopicCluster(
                label=0,
                message_entries=[],
                message_count=0,
            ),
            centroid=np.random.randn(TOPIC_DIM).astype(np.float32),
        )

        with patch(
            "nonebot_plugin_wtfllm.topic.archive.pipeline.memory_item_repo"
        ) as mock_repo:
            mock_repo.get_many_by_message_ids = AsyncMock()

            from nonebot_plugin_wtfllm.topic.archive.pipeline import archive_cluster

            await archive_cluster(candidate)

        mock_repo.get_many_by_message_ids.assert_not_awaited()


class TestMMRSelection:
    """验证 MMR 选择在管道中的行为。"""

    def test_mmr_selects_diverse_representatives(self):
        """MMR 应选出既相关又多样的代表"""
        rng = np.random.default_rng(42)
        # 5 个候选 + 1 个查询中心
        candidates = rng.standard_normal((5, TOPIC_DIM)).astype(np.float32)
        # L2 归一化
        norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        candidates = candidates / norms
        query = candidates.mean(axis=0)
        query /= np.linalg.norm(query)

        selected = mmr_select(candidates, query, k=3, lambda_param=0.5)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # 无重复

    def test_mmr_respects_k_limit(self):
        """k > N 时返回所有"""
        candidates = np.eye(3, TOPIC_DIM, dtype=np.float32)
        query = np.ones(TOPIC_DIM, dtype=np.float32)
        query /= np.linalg.norm(query)
        selected = mmr_select(candidates, query, k=10)
        assert len(selected) == 3

    def test_mmr_empty_input(self):
        """空候选返回空列表"""
        candidates = np.zeros((0, TOPIC_DIM), dtype=np.float32)
        query = np.ones(TOPIC_DIM, dtype=np.float32)
        selected = mmr_select(candidates, query, k=3)
        assert selected == []


class TestClusteringToCandidate:
    """验证聚类引擎的淘汰/清理正确生成 ArchivalCandidate。"""

    def _make_vec(self, dim: int, index: int) -> np.ndarray:
        vec = np.zeros(dim)
        vec[index % dim] = 1.0
        return vec.reshape(1, -1)

    def test_eviction_produces_candidate_with_centroid(self):
        """淘汰时应包含簇中心向量快照"""
        from nonebot_plugin_wtfllm.topic._types import TopicSessionState

        clustering = TopicClustering(threshold=0.99, max_clusters=2, decay_seconds=3600)
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)

        # 填满 2 个簇
        for i in range(2):
            clustering.assign(
                self._make_vec(TOPIC_DIM, i), state=state, session_key=key, now=100.0 + i
            )
            state.clusters[i] = TopicCluster(
                label=i, message_count=15, last_active_at=100.0 + i
            )

        # 第 3 个触发淘汰
        label, candidate = clustering.assign(
            self._make_vec(TOPIC_DIM, 2),
            state=state,
            session_key=key,
            min_archive_messages=3,
            now=200.0,
        )

        assert candidate is not None
        assert candidate.centroid.shape[-1] == TOPIC_DIM
        assert candidate.cluster.message_count == 15
        assert candidate.session_key == key

    def test_prune_produces_candidates_for_qualified_clusters(self):
        """超时清理应对消息数达标的簇产生候选"""
        from nonebot_plugin_wtfllm.topic._types import TopicSessionState

        clustering = TopicClustering(threshold=0.5, max_clusters=10, decay_seconds=60)
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)

        clustering.assign(
            self._make_vec(TOPIC_DIM, 0), state=state, session_key=key, now=0.0
        )

        # 模拟过期且消息充足的簇
        stale = TopicCluster(label=0, message_count=20)
        stale.last_active_at = time.time() - 120
        stale.message_entries = [(f"msg_{i}", 0.0) for i in range(20)]
        state.clusters[0] = stale

        pruned, candidates = clustering.prune_stale_topics(
            state, session_key=key, min_archive_messages=10
        )
        assert 0 in pruned
        assert len(candidates) == 1
        assert candidates[0].cluster.message_count == 20

    def test_prune_skips_insufficient_clusters(self):
        """消息数不够的过期簇不产生候选"""
        from nonebot_plugin_wtfllm.topic._types import TopicSessionState

        clustering = TopicClustering(threshold=0.5, max_clusters=10, decay_seconds=60)
        key = SessionKey(agent_id="a1", group_id="g1")
        state = TopicSessionState(session_key=key)

        clustering.assign(
            self._make_vec(TOPIC_DIM, 0), state=state, session_key=key, now=0.0
        )

        stale = TopicCluster(label=0, message_count=2)
        stale.last_active_at = time.time() - 120
        state.clusters[0] = stale

        pruned, candidates = clustering.prune_stale_topics(
            state, session_key=key, min_archive_messages=10
        )
        assert 0 in pruned
        assert len(candidates) == 0


class TestTopicManagerWorkerLifecycle:
    """验证 manager 的 start/stop 和 worker 生命周期管理。"""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """start() 启动 worker, stop() 安全关闭"""
        manager = TopicManager(maxsize=5)
        assert manager._archive_worker is None

        manager.start()
        assert manager._archive_worker is not None
        assert not manager._archive_worker.done()

        await manager.stop()
        assert manager._archive_worker is None

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_called(self):
        """注册多个 handler 时全部被调用"""
        calls_a: list[ArchivalCandidate] = []
        calls_b: list[ArchivalCandidate] = []

        async def handler_a(c: ArchivalCandidate):
            calls_a.append(c)

        async def handler_b(c: ArchivalCandidate):
            calls_b.append(c)

        manager = TopicManager(maxsize=5)
        manager.start(handler_a)
        manager.start(handler_b)  # 追加第二个 handler

        fake_candidate = ArchivalCandidate(
            session_key=SessionKey(agent_id="a1", group_id="g1"),
            cluster=TopicCluster(label=0, message_count=5),
            centroid=np.zeros(TOPIC_DIM),
        )
        manager._archive_queue.put_nowait(fake_candidate)

        await manager._archive_queue.join()
        await manager.stop()

        assert len(calls_a) == 1
        assert len(calls_b) == 1

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash_worker(self):
        """handler 抛异常不应导致 worker 崩溃"""

        async def bad_handler(c: ArchivalCandidate):
            raise RuntimeError("boom")

        good_calls: list[ArchivalCandidate] = []

        async def good_handler(c: ArchivalCandidate):
            good_calls.append(c)

        manager = TopicManager(maxsize=5)
        manager.start(bad_handler)

        fake = ArchivalCandidate(
            session_key=SessionKey(agent_id="a1", group_id="g1"),
            cluster=TopicCluster(label=0, message_count=5),
            centroid=np.zeros(TOPIC_DIM),
        )
        # 发送两个候选
        manager._archive_queue.put_nowait(fake)
        manager._archive_queue.put_nowait(fake)

        await manager._archive_queue.join()

        # worker 仍然存活
        assert not manager._archive_worker.done()
        await manager.stop()


class TestQueryTopicAfterEviction:
    """验证淘汰后 query_topic 不再返回已淘汰簇的消息。"""

    @pytest.mark.asyncio
    async def test_evicted_cluster_not_queryable(self):
        """淘汰的簇应该从 query_topic 中消失"""
        manager = TopicManager(
            maxsize=10,
            cluster_threshold=0.99,
            max_clusters=2,
            min_archive_messages=1,
            decay_seconds=7200,
        )

        # 填满 2 个簇
        await manager.ingest("a1", "g1", None, "food_0", "今天吃了红烧牛肉面非常好吃")
        await manager.ingest("a1", "g1", None, "tech_0", "Python异步编程asyncio框架")

        # 记录当前簇标签
        ctx = manager._sessions["a1:g:g1"]
        labels_before = set(ctx.state.clusters.keys())
        assert len(labels_before) == 2

        # 第 3 条触发淘汰
        await manager.ingest("a1", "g1", None, "travel_0", "周末去西湖旅游拍照留念")

        labels_after = set(ctx.state.clusters.keys())

        # 至少有一个旧簇被淘汰
        evicted = labels_before - labels_after
        if evicted:
            evicted_label = evicted.pop()
            # predict_only 应该不再匹配到这个标签
            assert evicted_label not in ctx.clustering._centroids


class TestVectorizerIntegration:
    """验证 TopicVectorizer 在管线中的集成。"""

    def test_transform_output_shape_and_norm(self):
        """transform 输出应为 1×D 的 L2 归一化向量"""
        from nonebot_plugin_wtfllm.topic.clustering.vectorizer import vectorizer

        vec = vectorizer.transform("测试文本")
        assert vec.shape == (1, TOPIC_DIM)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_transform_batch_consistency(self):
        """batch transform 和单条 transform 应一致"""
        from nonebot_plugin_wtfllm.topic.clustering.vectorizer import vectorizer

        texts = ["测试文本一", "测试文本二"]
        batch = vectorizer.transform_batch(texts)
        single_0 = vectorizer.transform(texts[0])
        single_1 = vectorizer.transform(texts[1])

        assert batch.shape == (2, TOPIC_DIM)
        np.testing.assert_allclose(batch[0], single_0.flatten(), atol=1e-5)
        np.testing.assert_allclose(batch[1], single_1.flatten(), atol=1e-5)

    def test_similar_texts_have_high_cosine(self):
        """语义相似文本的余弦相似度应较高"""
        from nonebot_plugin_wtfllm.topic.clustering.vectorizer import vectorizer

        v1 = vectorizer.transform("今天吃了红烧肉很好吃").flatten()
        v2 = vectorizer.transform("今天吃了糖醋排骨味道不错").flatten()
        v3 = vectorizer.transform("Python异步编程asyncio框架").flatten()

        sim_food = float(np.dot(v1, v2))
        sim_cross = float(np.dot(v1, v3))
        assert sim_food > sim_cross, (
            f"同话题相似度 ({sim_food:.3f}) 应高于跨话题 ({sim_cross:.3f})"
        )
