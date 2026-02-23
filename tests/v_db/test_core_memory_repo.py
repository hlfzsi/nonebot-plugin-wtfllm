"""v_db/repositories/core_memory.py 单元测试

使用 mock Qdrant 客户端测试 CoreMemoryRepository 的所有方法。
"""

from types import SimpleNamespace

import pytest
from qdrant_client import models

from nonebot_plugin_wtfllm.memory.items.core_memory import CoreMemory
from nonebot_plugin_wtfllm.v_db.models.core_memory import CoreMemoryPayload
from nonebot_plugin_wtfllm.v_db.repositories.core_memory import CoreMemoryRepository
import nonebot_plugin_wtfllm.v_db.repositories.base as repo_base


@pytest.fixture
def qdrant(monkeypatch, mock_qdrant_client):
    """Patch qdrant client globally for v_db repos"""
    monkeypatch.setattr(repo_base, "get_qdrant_client", lambda: mock_qdrant_client)
    return mock_qdrant_client


def _make_core_memory(**kwargs) -> CoreMemory:
    defaults = dict(
        storage_id="cm-test-1",
        content="test memory content",
        agent_id="a1",
        created_at=1000,
        updated_at=1000,
        source="agent",
        token_count=10,
        related_entities=["u1"],
    )
    defaults.update(kwargs)
    return CoreMemory(**defaults)


def _make_payload(**kwargs) -> CoreMemoryPayload:
    memory = _make_core_memory(**kwargs)
    return CoreMemoryPayload.from_core_memory(memory)


def _point(payload_dict, score=0.9):
    return SimpleNamespace(payload=payload_dict, score=score)


# ===================== save 测试 =====================


class TestCoreMemoryRepoSave:
    """CoreMemoryRepository save 测试"""

    @pytest.mark.asyncio
    async def test_save_core_memory(self, qdrant):
        qdrant.upsert.return_value = None
        repo = CoreMemoryRepository()
        memory = _make_core_memory()
        sid = await repo.save_core_memory(memory)
        assert sid == "cm-test-1"
        qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_many_core_memories(self, qdrant):
        qdrant.upsert.return_value = None
        repo = CoreMemoryRepository()
        memories = [
            _make_core_memory(storage_id=f"cm-bulk-{i}")
            for i in range(3)
        ]
        sids = await repo.save_many_core_memories(memories)
        assert len(sids) == 3


# ===================== get 测试 =====================


class TestCoreMemoryRepoGet:
    """CoreMemoryRepository get 测试"""

    @pytest.mark.asyncio
    async def test_get_core_memory_by_id(self, qdrant):
        payload = _make_payload()
        qdrant.retrieve.return_value = [_point(payload.model_dump())]
        repo = CoreMemoryRepository()
        result = await repo.get_core_memory_by_id("cm-test-1")
        assert result is not None
        assert result.storage_id == "cm-test-1"
        assert isinstance(result, CoreMemory)

    @pytest.mark.asyncio
    async def test_get_core_memory_by_id_not_found(self, qdrant):
        qdrant.retrieve.return_value = []
        repo = CoreMemoryRepository()
        result = await repo.get_core_memory_by_id("nonexistent")
        assert result is None


# ===================== get_by_session 测试 =====================


class TestCoreMemoryRepoSession:
    """CoreMemoryRepository session 操作测试"""

    @pytest.mark.asyncio
    async def test_get_by_session_group(self, qdrant):
        payload = _make_payload(group_id="g1")
        qdrant.scroll.return_value = ([_point(payload.model_dump())], None)
        repo = CoreMemoryRepository()
        results = await repo.get_by_session(agent_id="a1", group_id="g1")
        assert len(results) == 1
        assert isinstance(results[0], CoreMemory)

    @pytest.mark.asyncio
    async def test_get_by_session_user(self, qdrant):
        payload = _make_payload(user_id="u1")
        qdrant.scroll.return_value = ([_point(payload.model_dump())], None)
        repo = CoreMemoryRepository()
        results = await repo.get_by_session(agent_id="a1", user_id="u1")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_by_session_requires_group_or_user(self, qdrant):
        repo = CoreMemoryRepository()
        with pytest.raises(ValueError, match="Either group_id or user_id"):
            await repo.get_by_session(agent_id="a1")

    @pytest.mark.asyncio
    async def test_get_by_session_empty(self, qdrant):
        qdrant.scroll.return_value = ([], None)
        repo = CoreMemoryRepository()
        results = await repo.get_by_session(agent_id="a1", group_id="g1")
        assert results == []


# ===================== search 测试 =====================


class TestCoreMemoryRepoSearch:
    """CoreMemoryRepository search 测试"""

    @pytest.mark.asyncio
    async def test_search_cross_session(self, qdrant):
        payload = _make_payload()
        qdrant.query_points.return_value = SimpleNamespace(
            points=[_point(payload.model_dump(), 0.85)]
        )
        repo = CoreMemoryRepository()
        results = await repo.search_cross_session(
            agent_id="a1",
            query="test query",
            exclude_group_id="g_current",
            limit=5,
        )
        assert len(results) == 1
        assert results[0].score == 0.85
        assert isinstance(results[0].item, CoreMemory)

    @pytest.mark.asyncio
    async def test_search_cross_session_empty(self, qdrant):
        qdrant.query_points.return_value = SimpleNamespace(points=[])
        repo = CoreMemoryRepository()
        results = await repo.search_cross_session(
            agent_id="a1",
            query="nothing",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_by_entities(self, qdrant):
        payload = _make_payload(related_entities=["u1"])
        qdrant.query_points.return_value = SimpleNamespace(
            points=[_point(payload.model_dump(), 0.7)]
        )
        repo = CoreMemoryRepository()
        results = await repo.search_by_entities(
            agent_id="a1",
            entity_ids=["u1"],
            query="coding",
            limit=5,
        )
        assert len(results) == 1
        assert results[0].item.related_entities == ["u1"]


# ===================== delete 测试 =====================


class TestCoreMemoryRepoDelete:
    """CoreMemoryRepository delete 测试"""

    @pytest.mark.asyncio
    async def test_delete_by_storage_ids(self, qdrant):
        qdrant.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = CoreMemoryRepository()
        result = await repo.delete_by_storage_ids(["cm-1", "cm-2"])
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_by_storage_ids_empty(self, qdrant):
        repo = CoreMemoryRepository()
        result = await repo.delete_by_storage_ids([])
        assert result is True
        qdrant.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_by_session(self, qdrant):
        qdrant.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = CoreMemoryRepository()
        result = await repo.delete_by_session(agent_id="a1", group_id="g1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_by_session_user(self, qdrant):
        qdrant.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = CoreMemoryRepository()
        result = await repo.delete_by_session(agent_id="a1", user_id="u1")
        assert result is True


# ===================== count 测试 =====================


class TestCoreMemoryRepoCount:
    """CoreMemoryRepository count 测试"""

    @pytest.mark.asyncio
    async def test_count_by_session(self, qdrant):
        qdrant.count.return_value = SimpleNamespace(count=5)
        repo = CoreMemoryRepository()
        result = await repo.count_by_session(agent_id="a1", group_id="g1")
        assert result == 5

    @pytest.mark.asyncio
    async def test_count_by_session_user(self, qdrant):
        qdrant.count.return_value = SimpleNamespace(count=3)
        repo = CoreMemoryRepository()
        result = await repo.count_by_session(agent_id="a1", user_id="u1")
        assert result == 3
