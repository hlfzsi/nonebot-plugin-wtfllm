"""db/repositories/tool_call_record.py 单元测试

覆盖:
- save_empty_record
- save_batch (空列表 / 正常批量)
- get_recent (group_id / user_id 过滤 / limit=0)
- save_batch_from_tool_call_info
"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from nonebot_plugin_wtfllm.db import engine as db_engine
from nonebot_plugin_wtfllm.db.repositories import tool_call_record as repo_tcr
from nonebot_plugin_wtfllm.db.repositories.tool_call_record import (
    ToolCallRecordRepository,
)
from nonebot_plugin_wtfllm.db.models.tool_call_record import ToolCallRecordTable


@pytest.fixture
async def in_memory_db_tcr(monkeypatch):
    """为 ToolCallRecord 专用的 in-memory DB fixture"""
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
    monkeypatch.setattr(repo_tcr, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_tcr, "WRITE_LOCK", lock)

    yield session_maker
    await engine.dispose()


# ===========================================================================
# save_empty_record
# ===========================================================================


class TestSaveEmptyRecord:
    @pytest.mark.asyncio
    async def test_creates_placeholder(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        count = await repo.save_empty_record(agent_id="a1", group_id="g1")
        assert count == 1

    @pytest.mark.asyncio
    async def test_placeholder_has_no_tool_name(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        await repo.save_empty_record(agent_id="a1", user_id="u1")
        records = await repo.get_recent("a1", user_id="u1", limit=1)
        assert len(records) == 1
        assert records[0].tool_name == "No Tool was called"


# ===========================================================================
# save_batch
# ===========================================================================


class TestSaveBatch:
    @pytest.mark.asyncio
    async def test_empty_list(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        count = await repo.save_batch([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_batch_save(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        records = [
            ToolCallRecordTable(
                run_id="run_1",
                run_step=i,
                agent_id="a1",
                group_id="g1",
                tool_name=f"tool_{i}",
                kwargs={},
                timestamp=1000 + i,
            )
            for i in range(3)
        ]
        count = await repo.save_batch(records)
        assert count == 3

        fetched = await repo.get_recent("a1", group_id="g1", limit=1)
        assert len(fetched) == 3  # 同一个 run_id


# ===========================================================================
# get_recent
# ===========================================================================


class TestGetRecent:
    @pytest.mark.asyncio
    async def test_limit_zero(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        result = await repo.get_recent("a1", limit=0)
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_by_group_id(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        records = [
            ToolCallRecordTable(
                run_id="r1", run_step=0, agent_id="a1",
                group_id="g1", tool_name="t1", kwargs={}, timestamp=100,
            ),
            ToolCallRecordTable(
                run_id="r2", run_step=0, agent_id="a1",
                group_id="g2", tool_name="t2", kwargs={}, timestamp=200,
            ),
        ]
        await repo.save_batch(records)

        result = await repo.get_recent("a1", group_id="g1", limit=10)
        assert all(r.group_id == "g1" for r in result)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_filter_by_user_id_no_group(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        records = [
            ToolCallRecordTable(
                run_id="r1", run_step=0, agent_id="a1",
                group_id=None, user_id="u1",
                tool_name="t1", kwargs={}, timestamp=100,
            ),
            ToolCallRecordTable(
                run_id="r2", run_step=0, agent_id="a1",
                group_id=None, user_id="u2",
                tool_name="t2", kwargs={}, timestamp=200,
            ),
        ]
        await repo.save_batch(records)

        result = await repo.get_recent("a1", user_id="u1", limit=10)
        assert len(result) == 1
        assert result[0].user_id == "u1"

    @pytest.mark.asyncio
    async def test_multiple_runs_limit(self, in_memory_db_tcr):
        repo = ToolCallRecordRepository()
        # 3 个不同的 run_id
        for i in range(3):
            await repo.save_batch([
                ToolCallRecordTable(
                    run_id=f"run_{i}", run_step=0, agent_id="a1",
                    group_id="g1", tool_name=f"tool_{i}",
                    kwargs={}, timestamp=100 * (i + 1),
                )
            ])

        # limit=2 应只返回最近 2 次 run
        result = await repo.get_recent("a1", group_id="g1", limit=2)
        run_ids = {r.run_id for r in result}
        assert len(run_ids) == 2
        assert "run_2" in run_ids
        assert "run_1" in run_ids


# ===========================================================================
# save_batch_from_tool_call_info
# ===========================================================================


class TestSaveBatchFromToolCallInfo:
    @pytest.mark.asyncio
    async def test_from_tool_call_info(self, in_memory_db_tcr):
        from unittest.mock import MagicMock

        repo = ToolCallRecordRepository()

        info1 = MagicMock()
        info1.run_id = "run_x"
        info1.round_index = 0
        info1.tool_name = "search"
        info1.kwargs = {"q": "test"}
        info1.timestamp = 500

        info2 = MagicMock()
        info2.run_id = "run_x"
        info2.round_index = 1
        info2.tool_name = "reply"
        info2.kwargs = {"text": "hi"}
        info2.timestamp = 501

        count = await repo.save_batch_from_tool_call_info(
            [info1, info2], agent_id="a1", group_id="g1"
        )
        assert count == 2

        records = await repo.get_recent("a1", group_id="g1", limit=1)
        assert len(records) == 2
        names = {r.tool_name for r in records}
        assert names == {"search", "reply"}
