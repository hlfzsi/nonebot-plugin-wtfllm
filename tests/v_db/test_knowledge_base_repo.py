"""v_db/repositories/knowledge_base.py 单元测试

使用 mock Qdrant 客户端测试 KnowledgeBaseRepository 的所有方法。
"""

from types import SimpleNamespace

import pytest
from qdrant_client import models

from nonebot_plugin_wtfllm.memory.items.knowledge_base import KnowledgeEntry
from nonebot_plugin_wtfllm.v_db.models.knowledge_base import KnowledgeBasePayload
from nonebot_plugin_wtfllm.v_db.repositories.knowledge_base import KnowledgeBaseRepository
import nonebot_plugin_wtfllm.v_db.repositories.base as repo_base


@pytest.fixture
def qdrant(monkeypatch, mock_qdrant_client):
    """Patch qdrant client globally for v_db repos"""
    monkeypatch.setattr(repo_base, "get_qdrant_client", lambda: mock_qdrant_client)
    return mock_qdrant_client


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = dict(
        storage_id="kb-test-1",
        content="React Hooks 是 React 16.8 引入的特性",
        title="React Hooks",
        category="技术",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        tags=["前端", "React"],
        token_count=15,
    )
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def _make_payload(**kwargs) -> KnowledgeBasePayload:
    entry = _make_entry(**kwargs)
    return KnowledgeBasePayload.from_knowledge_entry(entry)


def _point(payload_dict, score=0.9):
    return SimpleNamespace(payload=payload_dict, score=score)


# ===================== save 测试 =====================


class TestKnowledgeBaseRepoSave:
    """KnowledgeBaseRepository save 测试"""

    async def test_save_knowledge(self, qdrant):
        qdrant.upsert.return_value = None
        repo = KnowledgeBaseRepository()
        entry = _make_entry()
        sid = await repo.save_knowledge(entry)
        assert sid == "kb-test-1"
        qdrant.upsert.assert_called_once()

    async def test_save_many_knowledge(self, qdrant):
        qdrant.upsert.return_value = None
        repo = KnowledgeBaseRepository()
        entries = [
            _make_entry(storage_id=f"kb-bulk-{i}")
            for i in range(3)
        ]
        sids = await repo.save_many_knowledge(entries)
        assert len(sids) == 3
        assert sids == ["kb-bulk-0", "kb-bulk-1", "kb-bulk-2"]


# ===================== get 测试 =====================


class TestKnowledgeBaseRepoGet:
    """KnowledgeBaseRepository get 测试"""

    async def test_get_knowledge_by_id(self, qdrant):
        payload = _make_payload()
        qdrant.retrieve.return_value = [_point(payload.model_dump())]
        repo = KnowledgeBaseRepository()
        result = await repo.get_knowledge_by_id("kb-test-1")
        assert result is not None
        assert result.storage_id == "kb-test-1"
        assert isinstance(result, KnowledgeEntry)

    async def test_get_knowledge_by_id_not_found(self, qdrant):
        qdrant.retrieve.return_value = []
        repo = KnowledgeBaseRepository()
        result = await repo.get_knowledge_by_id("nonexistent")
        assert result is None


# ===================== search 测试 =====================


class TestKnowledgeBaseRepoSearch:
    """KnowledgeBaseRepository search 测试"""

    async def test_search_relevant(self, qdrant):
        payload = _make_payload()
        qdrant.query_points.return_value = SimpleNamespace(
            points=[_point(payload.model_dump(), 0.85)]
        )
        repo = KnowledgeBaseRepository()
        results = await repo.search_relevant(
            agent_id="a1",
            query="React Hooks",
            limit=5,
        )
        assert len(results) == 1
        assert results[0].score == 0.85
        assert isinstance(results[0].item, KnowledgeEntry)
        assert results[0].item.title == "React Hooks"

    async def test_search_relevant_with_category(self, qdrant):
        payload = _make_payload(category="技术")
        qdrant.query_points.return_value = SimpleNamespace(
            points=[_point(payload.model_dump(), 0.8)]
        )
        repo = KnowledgeBaseRepository()
        results = await repo.search_relevant(
            agent_id="a1",
            query="programming",
            category="技术",
            limit=5,
        )
        assert len(results) == 1

    async def test_search_relevant_empty(self, qdrant):
        qdrant.query_points.return_value = SimpleNamespace(points=[])
        repo = KnowledgeBaseRepository()
        results = await repo.search_relevant(
            agent_id="a1",
            query="nothing",
        )
        assert results == []

    async def test_find_similar(self, qdrant):
        payload = _make_payload()
        qdrant.query_points.return_value = SimpleNamespace(
            points=[_point(payload.model_dump(), 0.92)]
        )
        repo = KnowledgeBaseRepository()
        results = await repo.find_similar(
            agent_id="a1",
            query="React Hooks 用于管理状态",
            limit=3,
        )
        assert len(results) == 1
        assert results[0].score == 0.92

    async def test_find_similar_empty(self, qdrant):
        qdrant.query_points.return_value = SimpleNamespace(points=[])
        repo = KnowledgeBaseRepository()
        results = await repo.find_similar(
            agent_id="a1",
            query="completely unrelated",
        )
        assert results == []


# ===================== scroll 测试 =====================


class TestKnowledgeBaseRepoScroll:
    """KnowledgeBaseRepository scroll 测试"""

    async def test_search_by_category(self, qdrant):
        payload = _make_payload(category="技术")
        qdrant.scroll.return_value = ([_point(payload.model_dump())], None)
        repo = KnowledgeBaseRepository()
        results = await repo.search_by_category(agent_id="a1", category="技术")
        assert len(results) == 1
        assert isinstance(results[0], KnowledgeEntry)

    async def test_search_by_category_empty(self, qdrant):
        qdrant.scroll.return_value = ([], None)
        repo = KnowledgeBaseRepository()
        results = await repo.search_by_category(agent_id="a1", category="不存在")
        assert results == []

    async def test_search_by_tags(self, qdrant):
        payload = _make_payload(tags=["前端", "React"])
        qdrant.scroll.return_value = ([_point(payload.model_dump())], None)
        repo = KnowledgeBaseRepository()
        results = await repo.search_by_tags(agent_id="a1", tags=["前端"])
        assert len(results) == 1
        assert "前端" in results[0].tags

    async def test_search_by_tags_empty(self, qdrant):
        qdrant.scroll.return_value = ([], None)
        repo = KnowledgeBaseRepository()
        results = await repo.search_by_tags(agent_id="a1", tags=["不存在"])
        assert results == []


# ===================== delete 测试 =====================


class TestKnowledgeBaseRepoDelete:
    """KnowledgeBaseRepository delete 测试"""

    async def test_delete_knowledge(self, qdrant):
        qdrant.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = KnowledgeBaseRepository()
        result = await repo.delete_knowledge("kb-test-1")
        assert result is True

    async def test_delete_by_agent(self, qdrant):
        qdrant.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = KnowledgeBaseRepository()
        result = await repo.delete_by_agent("a1")
        assert result is True


# ===================== count 测试 =====================


class TestKnowledgeBaseRepoCount:
    """KnowledgeBaseRepository count 测试"""

    async def test_count_by_agent(self, qdrant):
        qdrant.count.return_value = SimpleNamespace(count=10)
        repo = KnowledgeBaseRepository()
        result = await repo.count_by_agent("a1")
        assert result == 10

    async def test_count_by_agent_zero(self, qdrant):
        qdrant.count.return_value = SimpleNamespace(count=0)
        repo = KnowledgeBaseRepository()
        result = await repo.count_by_agent("nonexistent")
        assert result == 0
