"""db/repositories/thought_record.py 单元测试"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from nonebot_plugin_wtfllm.db import engine as db_engine
from nonebot_plugin_wtfllm.db.models.thought_record import ThoughtRecordTable
from nonebot_plugin_wtfllm.db.repositories import base as repo_base
from nonebot_plugin_wtfllm.db.repositories import thought_record as repo_module
from nonebot_plugin_wtfllm.db.repositories.thought_record import ThoughtRecordRepository


@pytest.fixture
async def in_memory_db_thought(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    lock = asyncio.Lock()

    monkeypatch.setattr(db_engine, "ENGINE", engine)
    monkeypatch.setattr(db_engine, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(db_engine, "WRITE_LOCK", lock)
    monkeypatch.setattr(repo_base, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_base, "WRITE_LOCK", lock)
    monkeypatch.setattr(repo_module, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_module, "WRITE_LOCK", lock)

    yield session_maker
    await engine.dispose()


class TestSaveRecord:
    @pytest.mark.asyncio
    async def test_save_record(self, in_memory_db_thought):
        repo = ThoughtRecordRepository()
        record = await repo.save_record(
            thought_of_chain="先确认上下文，再回答用户",
            agent_id="a1",
            group_id="g1",
            run_id="run-1",
        )

        assert record.id is not None
        assert record.run_id == "run-1"
        assert record.thought_of_chain == "先确认上下文，再回答用户"


class TestGetRecent:
    @pytest.mark.asyncio
    async def test_limit_zero(self, in_memory_db_thought):
        repo = ThoughtRecordRepository()
        result = await repo.get_recent("a1", limit=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_by_group_id(self, in_memory_db_thought):
        repo = ThoughtRecordRepository()
        await repo.save_record("g1-thought", agent_id="a1", group_id="g1")
        await repo.save_record("g2-thought", agent_id="a1", group_id="g2")

        result = await repo.get_recent("a1", group_id="g1", limit=10)
        assert len(result) == 1
        assert result[0].group_id == "g1"
        assert result[0].thought_of_chain == "g1-thought"

    @pytest.mark.asyncio
    async def test_filter_by_user_id_no_group(self, in_memory_db_thought):
        repo = ThoughtRecordRepository()
        await repo.save_record("u1-thought", agent_id="a1", user_id="u1")
        await repo.save_record("u2-thought", agent_id="a1", user_id="u2")

        result = await repo.get_recent("a1", user_id="u1", limit=10)
        assert len(result) == 1
        assert result[0].user_id == "u1"
        assert result[0].thought_of_chain == "u1-thought"

    @pytest.mark.asyncio
    async def test_returns_recent_n_records_in_ascending_order(self, in_memory_db_thought):
        repo = ThoughtRecordRepository()
        records = [
            ThoughtRecordTable(
                agent_id="a1",
                group_id="g1",
                thought_of_chain=f"thought-{idx}",
                timestamp=100 + idx,
            )
            for idx in range(3)
        ]
        await repo.save(records[0])
        await repo.save(records[1])
        await repo.save(records[2])

        result = await repo.get_recent("a1", group_id="g1", limit=2)
        assert [record.thought_of_chain for record in result] == ["thought-1", "thought-2"]