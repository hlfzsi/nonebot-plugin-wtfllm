"""话题系统内存与效率基准测试

用 pytest 运行: pytest tests/topic/bench_memory.py -s -v
"""

import gc
import sys
import time
import tracemalloc
from collections import defaultdict

import numpy as np
import pytest

from nonebot_plugin_wtfllm.topic._types import SessionKey, TopicSessionState
from nonebot_plugin_wtfllm.topic.manager import TopicManager
from nonebot_plugin_wtfllm.topic.clustering import TopicVectorizer
from nonebot_plugin_wtfllm.topic.clustering import TopicClustering


def sizeof_deep(obj, seen=None):
    """递归估算对象及其引用图的总内存"""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += sizeof_deep(k, seen) + sizeof_deep(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += sizeof_deep(item, seen)
    elif hasattr(obj, '__dict__'):
        size += sizeof_deep(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += sizeof_deep(getattr(obj, slot), seen)
    return size


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / 1024 ** 2:.2f} MB"


_DEFAULT_KEY = SessionKey(agent_id="a1", group_id="g1")


class TestBenchVectorizer:
    def test_memory_and_latency(self):
        tracemalloc.start()
        gc.collect()
        snap1 = tracemalloc.take_snapshot()

        vec = TopicVectorizer()

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        diff = snap2.compare_to(snap1, "lineno")
        vec_mem = sum(d.size_diff for d in diff if d.size_diff > 0)
        print(f"\n  Vectorizer 初始化内存: {format_bytes(vec_mem)}")

        texts = [
            "今天中午吃了一碗热腾腾的牛肉面",
            "Python的异步编程使用asyncio库来实现并发",
            "周末去看了一场非常精彩的电影",
        ]
        for text in texts:
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                vec.transform(text)
                times.append((time.perf_counter() - t0) * 1000)
            avg = sum(times) / len(times)
            p99 = sorted(times)[98]
            print(f"  transform({text[:15]}...): avg={avg:.2f}ms, p99={p99:.2f}ms")

        fv = vec.transform("测试文本")
        print(f"  输出向量 shape: {fv.shape}, dtype: {fv.dtype}, "
              f"大小: {format_bytes(fv.nbytes)}")
        tracemalloc.stop()


class TestBenchCentroidGrowth:
    def test_centroid_growth(self):
        vec = TopicVectorizer()
        clustering = TopicClustering(threshold=0.5, max_clusters=30)
        state = TopicSessionState(session_key=_DEFAULT_KEY)

        msgs = [
            f"话题{topic}的第{i}条消息内容讨论{['美食烹饪', '编程技术', '旅游出行', '体育运动', '音乐影视'][topic % 5]}"
            for topic in range(10)
            for i in range(100)
        ]

        print()
        checkpoints = [10, 50, 100, 200, 500, 1000]
        for idx, msg in enumerate(msgs, 1):
            fv = vec.transform(msg)
            clustering.assign(fv, state=state, session_key=_DEFAULT_KEY)
            if idx in checkpoints:
                n_centroids = clustering.n_clusters
                centroid_mem = sum(c.nbytes for c in clustering._centroids.values())
                print(f"  {idx:5d} 条消息 -> {n_centroids:3d} 个质心, "
                      f"质心内存: {format_bytes(centroid_mem)}")


class TestBenchPerSession:
    @pytest.mark.asyncio
    async def test_single_session_memory(self):
        tracemalloc.start()
        gc.collect()
        snap1 = tracemalloc.take_snapshot()

        manager = TopicManager(
            maxsize=10, cluster_threshold=0.5,
            max_clusters=30,
        )

        topics_text = [
            "今天做了一道非常好吃的红烧肉",
            "Python的asyncio异步编程框架",
            "周末去杭州西湖旅游景点",
            "NBA篮球比赛勇士队获胜",
            "最近看的电影特别好看推荐",
        ]
        for i in range(1000):
            topic = i % 5
            await manager.ingest("a1", "g1", None, f"msg_{i}", topics_text[topic])

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        diff = snap2.compare_to(snap1, "lineno")
        total_new = sum(d.size_diff for d in diff if d.size_diff > 0)

        ctx = manager._sessions["a1:g:g1"]
        state = ctx.state

        centroid_mem = sum(c.nbytes for c in ctx.clustering._centroids.values())
        cluster_entries_mem = sum(sizeof_deep(c.message_entries) for c in state.clusters.values())

        print(f"\n  1000 条消息后, {len(state.clusters)} 个簇:")
        print(f"    tracemalloc 总增量: {format_bytes(total_new)}")
        print(f"    质心内存: {format_bytes(centroid_mem)} "
              f"({ctx.clustering.n_clusters} centroids)")
        print(f"    各簇 message_entries: {format_bytes(cluster_entries_mem)}")
        tracemalloc.stop()


class TestBenchMultiSession:
    @pytest.mark.asyncio
    async def test_100_sessions(self):
        tracemalloc.start()
        gc.collect()
        snap1 = tracemalloc.take_snapshot()

        manager = TopicManager(
            maxsize=500, cluster_threshold=0.5,
            max_clusters=30,
        )

        for g in range(100):
            for i in range(50):
                await manager.ingest(
                    "a1", f"g{g}", None, f"g{g}_msg_{i}",
                    f"群{g}的第{i}条消息讨论美食编程旅游",
                )

        gc.collect()
        snap2 = tracemalloc.take_snapshot()
        diff = snap2.compare_to(snap1, "lineno")
        total_new = sum(d.size_diff for d in diff if d.size_diff > 0)
        print(f"\n  100 个群 x 50 条消息:")
        print(f"    tracemalloc 总增量: {format_bytes(total_new)}")
        print(f"    平均每会话: {format_bytes(total_new // 100)}")
        print(f"    推算 500 会话: {format_bytes(total_new * 5)}")
        tracemalloc.stop()


class TestBenchIngestLatency:
    @pytest.mark.asyncio
    async def test_latency_by_phase(self):
        manager = TopicManager(
            maxsize=10, cluster_threshold=0.5,
            max_clusters=30,
        )

        print()
        latencies = defaultdict(list)
        for i in range(1, 501):
            text = f"消息内容{i}讨论{['美食', '技术', '旅游'][i % 3]}话题"
            t0 = time.perf_counter()
            await manager.ingest("a1", "g1", None, f"msg_{i}", text)
            elapsed = (time.perf_counter() - t0) * 1000

            if i <= 10:
                latencies["1-10"].append(elapsed)
            elif i <= 50:
                latencies["11-50"].append(elapsed)
            elif i <= 100:
                latencies["51-100"].append(elapsed)
            elif i <= 200:
                latencies["101-200"].append(elapsed)
            elif i <= 500:
                latencies["201-500"].append(elapsed)

        for phase, times in latencies.items():
            avg = sum(times) / len(times)
            p50 = sorted(times)[len(times) // 2]
            p99 = sorted(times)[int(len(times) * 0.99)]
            mx = max(times)
            print(f"  消息 {phase:>8s}: avg={avg:.2f}ms, p50={p50:.2f}ms, "
                  f"p99={p99:.2f}ms, max={mx:.2f}ms")


class TestBenchDiverseMessages:
    @pytest.mark.asyncio
    async def test_memory_growth_with_diverse_text(self):
        """用随机多样文本测试内存增长"""
        import random
        random.seed(42)
        chars = [chr(c) for c in range(0x4e00, 0x9fa5)]  # CJK chars

        manager = TopicManager(
            maxsize=10, cluster_threshold=0.5,
            max_clusters=30,
        )

        print()
        for i in range(1, 2001):
            length = random.randint(10, 60)
            text = ''.join(random.choices(chars, k=length))
            await manager.ingest("a1", "g1", None, f"msg_{i}", text)

            if i in [50, 100, 500, 1000, 2000]:
                ctx = manager._sessions["a1:g:g1"]
                state = ctx.state
                n_centroids = ctx.clustering.n_clusters
                centroid_mem = sum(c.nbytes for c in ctx.clustering._centroids.values())
                entries_mem = sum(sizeof_deep(c.message_entries) for c in state.clusters.values())
                print(f"  {i:5d} msgs | clusters={len(state.clusters)} centroids={n_centroids} | "
                      f"entries={format_bytes(entries_mem)} | "
                      f"centroids={format_bytes(centroid_mem)}")


class TestBenchMaintenanceGap:
    @pytest.mark.asyncio
    async def test_prune(self):
        manager = TopicManager(
            maxsize=10, cluster_threshold=0.5,
            max_clusters=30, decay_seconds=7200,
            maintenance_interval=100,
        )

        for i in range(500):
            await manager.ingest("a1", "g1", None, f"msg_{i}",
                           f"消息{i}关于{['食物', '代码', '电影', '音乐', '运动'][i % 5]}")

        ctx = manager._sessions["a1:g:g1"]
        state = ctx.state
        key = state.session_key
        print(f"\n  500 条消息后:")
        print(f"    簇数: {len(state.clusters)}")
        print(f"    质心数: {ctx.clustering.n_clusters}")

        import time as _time
        old = _time.time() - 8000
        for c in state.clusters.values():
            c.last_active_at = old

        pruned, candidates = ctx.clustering.prune_stale_topics(state, session_key=key)
        print(f"    prune 了 {len(pruned)} 个簇, 剩余: {len(state.clusters)}")
        print(f"    归档候选数: {len(candidates)}")
        print(f"    质心数也同步减少为: {ctx.clustering.n_clusters}")
