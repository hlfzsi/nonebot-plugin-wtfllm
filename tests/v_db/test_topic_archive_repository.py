"""TopicArchiveRepository 单元测试"""

import math
import time
from types import SimpleNamespace

import pytest
from qdrant_client import models

import nonebot_plugin_wtfllm.v_db.repositories.base as repo_base
from nonebot_plugin_wtfllm.v_db.models.topic_archive import TopicArchivePayload
from nonebot_plugin_wtfllm.v_db.repositories.topic_archive import (
    TopicArchiveRepository,
    _DECAY_LAMBDA,
    _OVERFETCH_FACTOR,
)


@pytest.fixture
def qdrant_repo(monkeypatch, mock_qdrant_client):
    monkeypatch.setattr(repo_base, "get_qdrant_client", lambda: mock_qdrant_client)
    return mock_qdrant_client


def _payload(archive_id: str = "a1", created_at: int = 123) -> TopicArchivePayload:
    return TopicArchivePayload(
        archive_id=archive_id,
        agent_id="agent_1",
        group_id="group_1",
        user_id="user_1",
        representative_message_ids=["m1", "m2"],
        message_count=2,
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_upsert_uses_fastembed_document(qdrant_repo):
    """upsert 应通过 fastembed Document 写入 dense + sparse 向量。"""
    repo = TopicArchiveRepository()
    payload = _payload("a-upsert")
    doc_text = "代表消息1\n代表消息2"

    point_id = await repo.upsert(payload, document=doc_text)

    assert point_id == "a-upsert"
    qdrant_repo.upsert.assert_awaited_once()
    call_kwargs = qdrant_repo.upsert.await_args.kwargs
    assert call_kwargs["collection_name"] == TopicArchivePayload.collection_name
    point = call_kwargs["points"][0]
    assert point.id == "a-upsert"
    # 应包含 dense 和 sparse 两个向量键
    assert "dense" in point.vector
    assert "sparse" in point.vector


@pytest.mark.asyncio
async def test_search_by_session_uses_hybrid_search(qdrant_repo):
    """search_by_session 应走基类混合检索，然后经时间衰减重排。"""
    now = int(time.time())
    payload = _payload("a-search", created_at=now)  # 刚创建 → 衰减接近 1.0
    qdrant_repo.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(payload=payload.to_payload(), score=0.77)]
    )

    repo = TopicArchiveRepository()
    results = await repo.search_by_session(
        agent_id="agent_1",
        query="搜索话题",
        group_id="group_1",
        user_id="user_1",
        limit=2,
        score_threshold=0.5,
    )

    assert len(results) == 1
    assert results[0].item.archive_id == "a-search"
    # 刚创建的归档，衰减后 score ≈ 0.77（几乎不变）
    assert results[0].score == pytest.approx(0.77, abs=0.01)

    qdrant_repo.query_points.assert_awaited_once()
    call_kwargs = qdrant_repo.query_points.await_args.kwargs
    assert isinstance(call_kwargs["query"], models.FusionQuery)
    # 内部请求 limit 应为 requested_limit × overfetch_factor
    assert call_kwargs["limit"] == 2 * _OVERFETCH_FACTOR


@pytest.mark.asyncio
async def test_search_by_session_time_decay_reranks(qdrant_repo):
    """验证时间衰减正确地对新旧归档重排序。

    构造两条归档：
      - old: created_at = 70天前, raw_score = 0.90
      - new: created_at = 刚刚,   raw_score = 0.60
    70天衰减后 old 的 adjusted ≈ 0.90 × e^(-0.01×70) ≈ 0.447
    new 的 adjusted ≈ 0.60 × 1.0 = 0.60
    因此 new 应排在 old 前面。
    """
    now = int(time.time())
    old_ts = now - 70 * 86400  # 70天前
    payload_old = _payload("a-old", created_at=old_ts)
    payload_new = _payload("a-new", created_at=now)

    qdrant_repo.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(payload=payload_old.to_payload(), score=0.90),
            SimpleNamespace(payload=payload_new.to_payload(), score=0.60),
        ]
    )

    repo = TopicArchiveRepository()
    results = await repo.search_by_session(
        agent_id="agent_1",
        query="搜索话题",
        group_id="group_1",
        user_id="user_1",
        limit=2,
        score_threshold=0.3,
    )

    assert len(results) == 2
    # new 应排在 old 前面
    assert results[0].item.archive_id == "a-new"
    assert results[1].item.archive_id == "a-old"
    # 验证 adjusted score 数值
    expected_old = 0.90 * math.exp(-_DECAY_LAMBDA * 70)
    assert results[1].score == pytest.approx(expected_old, abs=0.02)
    assert results[0].score == pytest.approx(0.60, abs=0.01)


@pytest.mark.asyncio
async def test_search_by_session_truncates_to_limit(qdrant_repo):
    """超额获取后应截取到 limit 条。"""
    now = int(time.time())
    payloads = [
        (f"a-{i}", now - i * 86400, 0.80 - i * 0.05) for i in range(6)
    ]
    qdrant_repo.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                payload=_payload(aid, created_at=ts).to_payload(), score=score
            )
            for aid, ts, score in payloads
        ]
    )

    repo = TopicArchiveRepository()
    results = await repo.search_by_session(
        agent_id="agent_1",
        query="搜索话题",
        limit=3,
        score_threshold=0.3,
    )

    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_by_session_empty_results(qdrant_repo):
    """无结果时应返回空列表。"""
    qdrant_repo.query_points.return_value = SimpleNamespace(points=[])

    repo = TopicArchiveRepository()
    results = await repo.search_by_session(
        agent_id="agent_1",
        query="搜索话题",
        limit=3,
        score_threshold=0.3,
    )

    assert results == []


@pytest.mark.asyncio
async def test_delete_by_session(qdrant_repo):
    qdrant_repo.delete.return_value = SimpleNamespace(
        status=models.UpdateStatus.COMPLETED
    )

    repo = TopicArchiveRepository()
    ok = await repo.delete_by_session(
        agent_id="agent_1",
        group_id="group_1",
        user_id="user_1",
    )

    assert ok is True
    qdrant_repo.delete.assert_awaited_once()
