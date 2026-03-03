"""TopicArchiveRepository 单元测试"""

from types import SimpleNamespace

import pytest
from qdrant_client import models

import nonebot_plugin_wtfllm.v_db.repositories.base as repo_base
from nonebot_plugin_wtfllm.v_db.models.topic_archive import TopicArchivePayload
from nonebot_plugin_wtfllm.v_db.repositories.topic_archive import TopicArchiveRepository


@pytest.fixture
def qdrant_repo(monkeypatch, mock_qdrant_client):
    monkeypatch.setattr(repo_base, "get_qdrant_client", lambda: mock_qdrant_client)
    return mock_qdrant_client


def _payload(archive_id: str = "a1") -> TopicArchivePayload:
    return TopicArchivePayload(
        archive_id=archive_id,
        agent_id="agent_1",
        group_id="group_1",
        user_id="user_1",
        representative_message_ids=["m1", "m2"],
        message_count=2,
        created_at=123,
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
    """search_by_session 应走基类混合检索（Prefetch + DBSF fusion）。"""
    payload = _payload("a-search")
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
    assert results[0].score == 0.77

    qdrant_repo.query_points.assert_awaited_once()
    call_kwargs = qdrant_repo.query_points.await_args.kwargs
    # 基类混合检索使用 Prefetch + Fusion，最终 query 为 FusionQuery
    assert isinstance(call_kwargs["query"], models.FusionQuery)
    assert call_kwargs["limit"] == 2
    assert call_kwargs["score_threshold"] == 0.5


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
