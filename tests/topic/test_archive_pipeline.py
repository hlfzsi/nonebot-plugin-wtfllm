"""归档流水线 去重 / 子串过滤 单元测试"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nonebot_plugin_wtfllm.topic.archive.pipeline import (
    _deduplicate_texts,
    _remove_substrings,
    _MIN_UNIQUE_REPRESENTATIVES,
)
from nonebot_plugin_wtfllm.topic._types import (
    ArchivalCandidate,
    SessionKey,
    TopicCluster,
)
from nonebot_plugin_wtfllm.topic.clustering.vectorizer import vectorizer

TOPIC_DIM = vectorizer.transform("dim_probe").shape[1]


# ── helpers ──────────────────────────────────────────────


def _make_fake_item(message_id: str, text: str):
    item = MagicMock()
    item.message_id = message_id
    item.get_plain_text.return_value = text
    return item


def _make_candidate(
    n: int,
    texts: list[str] | None = None,
) -> tuple[ArchivalCandidate, list[MagicMock]]:
    """构造 ArchivalCandidate + 对应的 fake MemoryItem 列表。"""
    if texts is None:
        texts = [f"消息{i}" for i in range(n)]
    items = [_make_fake_item(f"msg_{i}", t) for i, t in enumerate(texts)]
    centroid = np.random.randn(TOPIC_DIM).astype(np.float32)
    centroid /= np.linalg.norm(centroid)
    candidate = ArchivalCandidate(
        session_key=SessionKey(agent_id="a1", group_id="g1"),
        cluster=TopicCluster(
            label=0,
            message_entries=[(f"msg_{i}", time.time()) for i in range(n)],
            message_count=n,
        ),
        centroid=centroid,
    )
    return candidate, items


# ── _deduplicate_texts ───────────────────────────────────


class TestDeduplicateTexts:
    def test_no_duplicates(self):
        indices = [0, 1, 2]
        texts = ["aaa", "bbb", "ccc"]
        ri, rt = _deduplicate_texts(indices, texts)
        assert ri == [0, 1, 2]
        assert rt == ["aaa", "bbb", "ccc"]

    def test_exact_duplicates_keep_first(self):
        indices = [0, 1, 2, 3, 4]
        texts = ["hello", "world", "hello", "hello", "world"]
        ri, rt = _deduplicate_texts(indices, texts)
        assert ri == [0, 1]
        assert rt == ["hello", "world"]

    def test_whitespace_only_difference(self):
        """strip() 后相同的文本视为重复"""
        indices = [0, 1, 2]
        texts = ["  你好  ", "你好", " 你好"]
        ri, rt = _deduplicate_texts(indices, texts)
        assert len(ri) == 1
        assert ri[0] == 0  # 保留首次出现

    def test_empty_list(self):
        ri, rt = _deduplicate_texts([], [])
        assert ri == []
        assert rt == []

    def test_all_same(self):
        """全部重复 → 只剩 1 条"""
        indices = [0, 1, 2, 3]
        texts = ["同一句话"] * 4
        ri, rt = _deduplicate_texts(indices, texts)
        assert len(ri) == 1


# ── _remove_substrings ──────────────────────────────────


class TestRemoveSubstrings:
    def test_no_substrings(self):
        indices = [0, 1, 2]
        texts = ["苹果很好吃", "今天天气晴朗", "编程真有趣"]
        ri, rt = _remove_substrings(indices, texts)
        assert len(ri) == 3

    def test_short_is_substring_of_long(self):
        """'你好' 是 '你好吗今天过得怎么样' 的子串 → 被移除"""
        indices = [0, 1, 2]
        texts = ["你好吗今天过得怎么样", "你好", "天气不错啊朋友"]
        ri, rt = _remove_substrings(indices, texts)
        assert len(ri) == 2
        stripped = [t.strip() for t in rt]
        assert "你好" not in stripped
        assert "你好吗今天过得怎么样" in stripped
        assert "天气不错啊朋友" in stripped

    def test_multiple_substrings(self):
        """多条短文本都是某长文本的子串"""
        indices = [0, 1, 2, 3]
        texts = [
            "ABCDEF",        # 长文本
            "ABC",           # A 的子串
            "DEF",           # A 的子串
            "完全不同的文本",
        ]
        ri, rt = _remove_substrings(indices, texts)
        stripped = [t.strip() for t in rt]
        assert "ABCDEF" in stripped
        assert "ABC" not in stripped
        assert "DEF" not in stripped
        assert "完全不同的文本" in stripped
        assert len(ri) == 2

    def test_identical_texts_first_kept(self):
        """完全相同的文本 → 第一个保留，后续作为子串被移除"""
        indices = [0, 1, 2]
        texts = ["一样的话", "一样的话", "一样的话"]
        ri, rt = _remove_substrings(indices, texts)
        # 长度相同时只有第一个被保留（后续 "in" 已保留的会命中）
        assert len(ri) == 1

    def test_empty(self):
        ri, rt = _remove_substrings([], [])
        assert ri == []
        assert rt == []


# ── archive_cluster 集成：去重导致跳过 ───────────────


class TestArchiveClusterDedup:
    @pytest.mark.asyncio
    async def test_all_duplicates_skip_archive(self):
        """10 条完全相同的消息 → 去重后仅 1 条 < MIN_UNIQUE → 跳过归档"""
        texts = ["大家好我是一条重复消息"] * 10
        candidate, items = _make_candidate(10, texts)

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
            mock_repo.get_many_by_message_ids = AsyncMock(return_value=items)
            mock_vdb.upsert = AsyncMock()
            mock_cfg.topic_archive_min_messages = 3
            mock_cfg.topic_archive_mmr_k = 5
            mock_cfg.topic_archive_mmr_lambda = 0.5

            from nonebot_plugin_wtfllm.topic.archive.pipeline import archive_cluster

            await archive_cluster(candidate)

        mock_vdb.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_substring_texts_skip_archive(self):
        """代表消息全部互为子串关系 → 子串过滤后不足 MIN_UNIQUE → 跳过"""
        texts = [
            "你好",
            "你好吗",
            "你好吗今天怎么样",
            "你好吗今天怎么样呢朋友",
            "你好吗今天怎么样呢朋友们大家好",
        ]
        candidate, items = _make_candidate(5, texts)

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
            mock_repo.get_many_by_message_ids = AsyncMock(return_value=items)
            mock_vdb.upsert = AsyncMock()
            mock_cfg.topic_archive_min_messages = 3
            mock_cfg.topic_archive_mmr_k = 5
            mock_cfg.topic_archive_mmr_lambda = 0.5

            from nonebot_plugin_wtfllm.topic.archive.pipeline import archive_cluster

            await archive_cluster(candidate)

        # 子串过滤后只剩 1 条（最长的那条），< 3 → 跳过
        mock_vdb.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_diverse_texts_archive_succeeds(self):
        """足够多样的消息 → 去重+子串过滤后仍 >= MIN_UNIQUE → 正常归档"""
        texts = [
            "Python编程语言asyncio异步框架非常强大",
            "今天晚餐吃了一碗热腾腾的红烧牛肉面",
            "周末去杭州西湖旅游景点拍照留念很开心",
            "NBA篮球比赛湖人队对阵勇士队精彩对决",
            "最近在研究深度学习中的Transformer架构",
        ]
        candidate, items = _make_candidate(5, texts)

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
            mock_repo.get_many_by_message_ids = AsyncMock(return_value=items)
            mock_vdb.upsert = AsyncMock(return_value="archive_id")
            mock_cfg.topic_archive_min_messages = 3
            mock_cfg.topic_archive_mmr_k = 5
            mock_cfg.topic_archive_mmr_lambda = 0.5

            from nonebot_plugin_wtfllm.topic.archive.pipeline import archive_cluster

            await archive_cluster(candidate)

        mock_vdb.upsert.assert_awaited_once()
