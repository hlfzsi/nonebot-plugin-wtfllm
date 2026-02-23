"""v_db/repositories 单元测试"""

from types import SimpleNamespace

import pytest
from qdrant_client import models

from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
from nonebot_plugin_wtfllm.v_db.repositories.meme import MemeRepository
import nonebot_plugin_wtfllm.v_db.repositories.base as repo_base


@pytest.fixture
def qdrant_repo(monkeypatch, mock_qdrant_client):
    monkeypatch.setattr(repo_base, "get_qdrant_client", lambda: mock_qdrant_client)
    return mock_qdrant_client


def _make_meme_payload(storage_id: str = "m1") -> MemePayload:
    return MemePayload(
        storage_id=storage_id,
        file_path=f"{storage_id}.webp",
        raw_message_id="raw_1",
        docs="funny meme",
        tags=["fun"],
        created_at=100,
        uploader_id="user_1",
    )


@pytest.mark.asyncio
async def test_meme_repository_calls(qdrant_repo):
    payload = _make_meme_payload()
    point = SimpleNamespace(payload=payload.model_dump(), score=0.5)

    qdrant_repo.query_points.return_value = SimpleNamespace(points=[point])
    qdrant_repo.scroll.return_value = ([point], None)
    qdrant_repo.count.return_value = SimpleNamespace(count=2)
    qdrant_repo.delete.return_value = SimpleNamespace(status=models.UpdateStatus.COMPLETED)

    repo = MemeRepository()

    results = await repo.search_by_text(query="funny", limit=5)
    assert results[0].item.storage_id == payload.storage_id

    memes, offset = await repo.list_by_uploader("user_1", limit=10)
    assert offset is None
    assert memes[0].uploader_id == "user_1"

    count = await repo.count_by_uploader("user_1")
    assert count == 2

    deleted = await repo.delete_by_uploader("user_1")
    assert deleted is True


# ===================== MemeRepository 扩展测试 =====================


def _meme_payload(sid: str = "m1", **kwargs) -> MemePayload:
    defaults = dict(
        storage_id=sid,
        file_path=f"{sid}.webp",
        raw_message_id="raw_1",
        docs="funny meme",
        tags=["fun"],
        created_at=100,
        uploader_id="user_1",
    )
    defaults.update(kwargs)
    return MemePayload(**defaults)


def _point(payload_dict, score=0.9):
    return SimpleNamespace(payload=payload_dict, score=score)


class TestMemeRepoSave:
    """MemeRepository save 测试"""

    @pytest.mark.asyncio
    async def test_save_meme(self, qdrant_repo):
        qdrant_repo.upsert.return_value = None
        repo = MemeRepository()
        sid = await repo.save_meme(_meme_payload("save_m1"))
        assert sid == "save_m1"

    @pytest.mark.asyncio
    async def test_save_many_memes(self, qdrant_repo):
        qdrant_repo.upsert.return_value = None
        repo = MemeRepository()
        sids = await repo.save_many_memes(
            [_meme_payload(f"bulk_{i}") for i in range(3)]
        )
        assert len(sids) == 3


class TestMemeRepoSearch:
    """MemeRepository search 测试"""

    @pytest.mark.asyncio
    async def test_search_by_text(self, qdrant_repo):
        payload = _meme_payload("st_1")
        qdrant_repo.query_points.return_value = SimpleNamespace(
            points=[_point(payload.to_payload(), 0.6)]
        )
        repo = MemeRepository()
        results = await repo.search_by_text("funny", limit=5)
        assert len(results) == 1
        assert results[0].item.storage_id == "st_1"
        assert results[0].score == 0.6

    @pytest.mark.asyncio
    async def test_search_by_text_empty(self, qdrant_repo):
        qdrant_repo.query_points.return_value = SimpleNamespace(points=[])
        repo = MemeRepository()
        results = await repo.search_by_text("nothing", limit=5)
        assert results == []


class TestMemeRepoList:
    """MemeRepository list 测试"""

    @pytest.mark.asyncio
    async def test_list_by_uploader(self, qdrant_repo):
        payload = _meme_payload("lu_1")
        qdrant_repo.scroll.return_value = ([_point(payload.to_payload())], None)
        repo = MemeRepository()
        memes, offset = await repo.list_by_uploader("user_1", limit=10)
        assert len(memes) == 1
        assert offset is None

    @pytest.mark.asyncio
    async def test_list_by_uploader_paginated(self, qdrant_repo):
        payload = _meme_payload("lu_p")
        qdrant_repo.scroll.return_value = ([_point(payload.to_payload())], "next_offset_7")
        repo = MemeRepository()
        memes, offset = await repo.list_by_uploader("user_1", limit=1)
        assert len(memes) == 1
        assert offset == "next_offset_7"

    @pytest.mark.asyncio
    async def test_search_by_tags_any(self, qdrant_repo):
        payload = _meme_payload("tag_1")
        qdrant_repo.scroll.return_value = ([_point(payload.to_payload())], None)
        repo = MemeRepository()
        memes, _ = await repo.search_by_tags(["fun"], match_all=False)
        assert len(memes) == 1

    @pytest.mark.asyncio
    async def test_search_by_tags_all(self, qdrant_repo):
        payload = _meme_payload("tag_all", tags=["a", "b"])
        qdrant_repo.scroll.return_value = ([_point(payload.to_payload())], None)
        repo = MemeRepository()
        memes, _ = await repo.search_by_tags(["a", "b"], match_all=True)
        assert len(memes) == 1

    @pytest.mark.asyncio
    async def test_get_recent(self, qdrant_repo):
        payload = _meme_payload("recent")
        qdrant_repo.scroll.return_value = ([_point(payload.to_payload())], None)
        repo = MemeRepository()
        memes = await repo.get_recent(since_timestamp=50, limit=10)
        assert len(memes) == 1

    @pytest.mark.asyncio
    async def test_get_by_message_id(self, qdrant_repo):
        payload = _meme_payload("msg_meme")
        qdrant_repo.scroll.return_value = ([_point(payload.to_payload())], None)
        repo = MemeRepository()
        result = await repo.get_by_message_id("raw_1")
        assert result is not None
        assert result.storage_id == "msg_meme"

    @pytest.mark.asyncio
    async def test_get_by_message_id_not_found(self, qdrant_repo):
        qdrant_repo.scroll.return_value = ([], None)
        repo = MemeRepository()
        result = await repo.get_by_message_id("nonexistent")
        assert result is None


class TestMemeRepoDelete:
    """MemeRepository delete 测试"""

    @pytest.mark.asyncio
    async def test_delete_by_uploader(self, qdrant_repo):
        qdrant_repo.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = MemeRepository()
        result = await repo.delete_by_uploader("user_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_before_timestamp(self, qdrant_repo):
        qdrant_repo.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = MemeRepository()
        result = await repo.delete_before_timestamp(500)
        assert result is True

    @pytest.mark.asyncio
    async def test_count_by_uploader(self, qdrant_repo):
        qdrant_repo.count.return_value = SimpleNamespace(count=7)
        repo = MemeRepository()
        result = await repo.count_by_uploader("user_1")
        assert result == 7

    @pytest.mark.asyncio
    async def test_exists_true(self, qdrant_repo):
        payload = _meme_payload("exists_1")
        qdrant_repo.retrieve.return_value = [_point(payload.to_payload())]
        repo = MemeRepository()
        assert await repo.exists("exists_1") is True

    @pytest.mark.asyncio
    async def test_exists_false(self, qdrant_repo):
        qdrant_repo.retrieve.return_value = []
        repo = MemeRepository()
        assert await repo.exists("ghost") is False


# ===================== VectorRepository 基础操作测试 =====================


class TestVectorRepositoryBase:
    """VectorRepository 基类方法测试"""

    @pytest.mark.asyncio
    async def test_get_by_id(self, qdrant_repo):
        payload = _meme_payload("base_get")
        qdrant_repo.retrieve.return_value = [_point(payload.to_payload())]
        repo = MemeRepository()
        result = await repo.get_by_id("base_get")
        assert result is not None
        assert result.storage_id == "base_get"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, qdrant_repo):
        qdrant_repo.retrieve.return_value = []
        repo = MemeRepository()
        result = await repo.get_by_id("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_many_by_ids_empty(self, qdrant_repo):
        repo = MemeRepository()
        result = await repo.get_many_by_ids([])
        assert result == []
        qdrant_repo.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_by_id(self, qdrant_repo):
        qdrant_repo.delete.return_value = SimpleNamespace(
            status=models.UpdateStatus.COMPLETED
        )
        repo = MemeRepository()
        result = await repo.delete_by_id("del_1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_many_by_ids_empty(self, qdrant_repo):
        repo = MemeRepository()
        result = await repo.delete_many_by_ids([])
        assert result is True

    def test_match_keyword(self):
        from nonebot_plugin_wtfllm.v_db.repositories.base import VectorRepository
        cond = VectorRepository.match_keyword("field", "value")
        assert isinstance(cond, models.FieldCondition)

    def test_match_any(self):
        from nonebot_plugin_wtfllm.v_db.repositories.base import VectorRepository
        cond = VectorRepository.match_any("field", ["a", "b"])
        assert isinstance(cond, models.FieldCondition)

    def test_range_filter(self):
        from nonebot_plugin_wtfllm.v_db.repositories.base import VectorRepository
        cond = VectorRepository.range_filter("field", gte=10, lte=100)
        assert isinstance(cond, models.FieldCondition)

    def test_is_null(self):
        from nonebot_plugin_wtfllm.v_db.repositories.base import VectorRepository
        cond = VectorRepository.is_null("field")
        assert isinstance(cond, models.IsNullCondition)


# ===================== MemeRepository =====================

from unittest.mock import MagicMock, AsyncMock


class TestMemeRepositoryMethods:
    @pytest.fixture
    def meme_repo(self):
        from nonebot_plugin_wtfllm.v_db.repositories.meme import MemeRepository
        repo = MemeRepository.__new__(MemeRepository)
        repo.scroll = AsyncMock(return_value=([], None))
        repo.search_by_text = AsyncMock(return_value=[])
        repo.get_by_id = AsyncMock(return_value=None)
        repo.delete_by_id = AsyncMock(return_value=True)
        return repo

    @pytest.mark.asyncio
    async def test_list_by_tag(self, meme_repo):
        mock_meme = MagicMock()
        meme_repo.scroll = AsyncMock(return_value=([mock_meme], None))
        result = await meme_repo.list_by_tag("funny")
        assert result == [mock_meme]

    @pytest.mark.asyncio
    async def test_get_recent_with_uploader(self, meme_repo):
        mock_meme = MagicMock()
        meme_repo.scroll = AsyncMock(return_value=([mock_meme], None))
        result = await meme_repo.get_recent(since_timestamp=1000, uploader_id="u1")
        assert result == [mock_meme]

    @pytest.mark.asyncio
    async def test_search_by_text_with_tags(self, meme_repo):
        result = await meme_repo.search_by_text_with_tags("query", ["tag1"])
        meme_repo.search_by_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_text_by_uploader(self, meme_repo):
        result = await meme_repo.search_by_text_by_uploader("query", "u1")
        meme_repo.search_by_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_meme_by_id(self, meme_repo):
        result = await meme_repo.get_meme_by_id("some_id")
        meme_repo.get_by_id.assert_called_once_with("some_id")

    @pytest.mark.asyncio
    async def test_delete_meme_by_id_with_payload(self, meme_repo):
        mock_payload = AsyncMock()
        meme_repo.get_by_id = AsyncMock(return_value=mock_payload)
        result = await meme_repo.delete_meme_by_id("some_id")
        mock_payload.delete_file.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_meme_by_id_no_payload(self, meme_repo):
        meme_repo.get_by_id = AsyncMock(return_value=None)
        result = await meme_repo.delete_meme_by_id("ghost")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_memes_by_uploader(self, meme_repo):
        mock_meme1 = AsyncMock()
        mock_meme2 = AsyncMock()
        meme_repo.list_by_uploader = AsyncMock(return_value=([mock_meme1, mock_meme2], None))
        meme_repo.delete_by_uploader = AsyncMock(return_value=True)
        result = await meme_repo.delete_memes_by_uploader("u1")
        assert result is True
        mock_meme1.delete_file.assert_called_once()
        mock_meme2.delete_file.assert_called_once()
