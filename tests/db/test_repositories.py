"""db/repositories 单元测试"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from nonebot_plugin_wtfllm.db import engine as db_engine
from nonebot_plugin_wtfllm.db import lifecycle as db_lifecycle
from nonebot_plugin_wtfllm.db.repositories import base as repo_base
from nonebot_plugin_wtfllm.db.repositories import memory_items as repo_memory
from nonebot_plugin_wtfllm.db.repositories import scheduled_message as repo_scheduled
from nonebot_plugin_wtfllm.db.repositories import user_persona as repo_user_persona
from nonebot_plugin_wtfllm.db.repositories.memory_items import MemoryItemRepository
from nonebot_plugin_wtfllm.db.repositories.scheduled_message import (
    ScheduledMessageRepository,
)
from nonebot_plugin_wtfllm.db.repositories.user_persona import UserPersonaRepository
from nonebot_plugin_wtfllm.db.models.scheduled_message import (
    ScheduledMessage,
    ScheduledMessageStatus,
    ScheduledFunctionType,
)
from nonebot_plugin_wtfllm.memory.content.message import Message
from nonebot_plugin_wtfllm.memory.items.base_items import (
    PrivateMemoryItem,
    GroupMemoryItem,
)


@pytest.fixture
async def in_memory_db(monkeypatch):
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
    monkeypatch.setattr(db_lifecycle, "ENGINE", engine)

    monkeypatch.setattr(repo_base, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_base, "WRITE_LOCK", lock)
    monkeypatch.setattr(repo_memory, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_memory, "WRITE_LOCK", lock)
    monkeypatch.setattr(repo_scheduled, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_scheduled, "WRITE_LOCK", lock)
    monkeypatch.setattr(repo_user_persona, "SESSION_MAKER", session_maker)
    monkeypatch.setattr(repo_user_persona, "WRITE_LOCK", lock)

    yield session_maker
    await engine.dispose()


@pytest.mark.asyncio
async def test_memory_item_repository_roundtrip(in_memory_db):
    repo = MemoryItemRepository()

    private_item = PrivateMemoryItem(
        message_id="msg_private",
        related_message_id=None,
        sender="user_a",
        content=Message.create().text("hello private"),
        created_at=10,
        agent_id="agent_1",
        user_id="user_a",
    )
    group_item = GroupMemoryItem(
        message_id="msg_group",
        related_message_id="msg_private",
        sender="user_b",
        content=Message.create().text("hello group"),
        created_at=20,
        agent_id="agent_1",
        group_id="group_1",
    )

    await repo.save_memory_item(private_item)
    await repo.save_memory_item(group_item)

    fetched_private = await repo.get_by_message_id("msg_private")
    fetched_group = await repo.get_by_message_id("msg_group")

    assert isinstance(fetched_private, PrivateMemoryItem)
    assert fetched_private.get_plain_text() == "hello private"
    assert isinstance(fetched_group, GroupMemoryItem)
    assert fetched_group.related_message_id == "msg_private"

    chain = await repo.get_chain_by_message_ids(["msg_group"])
    assert [item.message_id for item in chain] == ["msg_private", "msg_group"]


@pytest.mark.asyncio
async def test_memory_item_repository_queries(in_memory_db):
    repo = MemoryItemRepository()

    items = [
        PrivateMemoryItem(
            message_id="p1",
            related_message_id=None,
            sender="user_a",
            content=Message.create().text("p1"),
            created_at=1,
            agent_id="agent_1",
            user_id="user_a",
        ),
        PrivateMemoryItem(
            message_id="p2",
            related_message_id=None,
            sender="user_a",
            content=Message.create().text("p2"),
            created_at=2,
            agent_id="agent_1",
            user_id="user_a",
        ),
        GroupMemoryItem(
            message_id="g1",
            related_message_id=None,
            sender="user_b",
            content=Message.create().text("g1"),
            created_at=3,
            agent_id="agent_1",
            group_id="group_1",
        ),
    ]

    await repo.save_many(items)

    by_agent = await repo.list_by_agent("agent_1", limit=10)
    assert [item.message_id for item in by_agent] == ["g1", "p2", "p1"]

    by_user = await repo.get_by_user("user_a", "agent_1", limit=10)
    assert [item.message_id for item in by_user] == ["p2", "p1"]

    private_by_user = await repo.get_in_private_by_user("user_a", "agent_1", limit=10)
    assert [item.message_id for item in private_by_user] == ["p2", "p1"]

    by_group = await repo.get_by_group("group_1", "agent_1", limit=10)
    assert [item.message_id for item in by_group] == ["g1"]


@pytest.mark.asyncio
async def test_user_persona_repository_updates(in_memory_db):
    repo = UserPersonaRepository()

    profile = await repo.get_or_create("user_1")
    assert profile.user_id == "user_1"
    assert profile.version == 0

    updated = await repo.update_persona(
        user_id="user_1",
        interaction_style="friendly",
        note="likes tests",
        structured_preferences={"lang": "python"},
        impression="positive",
    )
    assert updated.version == 1
    assert updated.interaction_style == "friendly"

    text = await repo.get_persona_text(user_id="U1", real_user_id="user_1")
    assert text is not None
    assert "交互风格" in text


@pytest.mark.asyncio
async def test_scheduled_message_repository_flow(in_memory_db):
    repo = ScheduledMessageRepository()

    msg_pending = ScheduledMessage(
        job_id="job_1",
        target_data={},
        user_id="user_1",
        group_id=None,
        agent_id="agent_1",
        messages=[{"type": "text", "content": "hello"}],
        trigger_time=100,
        status=ScheduledMessageStatus.PENDING,
        created_at=10,
        func_type=ScheduledFunctionType.STATIC_MESSAGE,
    )
    msg_completed = ScheduledMessage(
        job_id="job_2",
        target_data={},
        user_id="user_2",
        group_id="group_1",
        agent_id="agent_1",
        messages=[{"type": "text", "content": "done"}],
        trigger_time=50,
        status=ScheduledMessageStatus.COMPLETED,
        created_at=5,
        executed_at=60,
        func_type=ScheduledFunctionType.STATIC_MESSAGE,
    )

    await repo.save(msg_pending)
    await repo.save(msg_completed)

    pending = await repo.list_pending(agent_id="agent_1")
    assert [msg.job_id for msg in pending] == ["job_1"]

    missed = await repo.list_missed(cutoff=120)
    assert [msg.job_id for msg in missed] == ["job_1"]

    updated = await repo.mark_failed("job_1", error_message="boom")
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error_message == "boom"

    cleaned = await repo.cleanup_completed(before=100)
    assert cleaned == 1


@pytest.mark.asyncio
async def test_init_and_shutdown_db(in_memory_db):
    await db_lifecycle.init_db()
    await db_lifecycle.shutdown_db()


# ===================== MemoryItemRepository 扩展测试 =====================


class TestMemoryItemDelete:
    """MemoryItemRepository 删除操作测试"""

    @pytest.mark.asyncio
    async def test_delete_existing(self, in_memory_db):
        repo = MemoryItemRepository()
        item = PrivateMemoryItem(
            message_id="del_1",
            sender="user_a",
            content=Message.create().text("to delete"),
            created_at=100,
            agent_id="agent_1",
            user_id="user_a",
        )
        await repo.save_memory_item(item)
        assert await repo.get_by_message_id("del_1") is not None

        deleted = await repo.delete_by_message_id("del_1")
        assert deleted is True
        assert await repo.get_by_message_id("del_1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, in_memory_db):
        repo = MemoryItemRepository()
        deleted = await repo.delete_by_message_id("nonexistent_id")
        assert deleted is False


class TestMemoryItemBatchGet:
    """MemoryItemRepository 批量获取测试"""

    @pytest.mark.asyncio
    async def test_get_many_by_message_ids(self, in_memory_db):
        repo = MemoryItemRepository()
        items = [
            GroupMemoryItem(
                message_id=f"batch_{i}",
                sender="user_b",
                content=Message.create().text(f"batch {i}"),
                created_at=i * 10,
                agent_id="agent_1",
                group_id="group_1",
            )
            for i in range(5)
        ]
        await repo.save_many(items)

        result = await repo.get_many_by_message_ids(["batch_0", "batch_2", "batch_4"])
        ids = {item.message_id for item in result}
        assert ids == {"batch_0", "batch_2", "batch_4"}

    @pytest.mark.asyncio
    async def test_get_many_empty_list(self, in_memory_db):
        repo = MemoryItemRepository()
        result = await repo.get_many_by_message_ids([])
        assert result == []


class TestMemoryItemTimeFilters:
    """MemoryItemRepository 时间过滤查询测试"""

    @pytest.fixture
    async def seeded_repo(self, in_memory_db):
        repo = MemoryItemRepository()
        items = [
            GroupMemoryItem(
                message_id=f"tf_{i}",
                sender="user_c",
                content=Message.create().text(f"time {i}"),
                created_at=i * 100,
                agent_id="agent_1",
                group_id="group_tf",
            )
            for i in range(1, 6)  # created_at: 100, 200, 300, 400, 500
        ]
        await repo.save_many(items)
        return repo

    @pytest.mark.asyncio
    async def test_get_by_group_after(self, seeded_repo):
        repo = seeded_repo
        result = await repo.get_by_group_after("group_tf", "agent_1", timestamp=300)
        ids = {item.message_id for item in result}
        assert ids == {"tf_3", "tf_4", "tf_5"}

    @pytest.mark.asyncio
    async def test_get_by_group_before(self, seeded_repo):
        repo = seeded_repo
        result = await repo.get_by_group_before("group_tf", "agent_1", timestamp=200)
        ids = {item.message_id for item in result}
        assert ids == {"tf_1", "tf_2"}

    @pytest.mark.asyncio
    async def test_get_by_group_after_with_limit(self, seeded_repo):
        repo = seeded_repo
        result = await repo.get_by_group_after(
            "group_tf", "agent_1", timestamp=100, limit=2
        )
        assert len(result) == 2


class TestMemoryItemPrivateTimeFilters:
    """MemoryItemRepository 私聊时间过滤查询测试"""

    @pytest.fixture
    async def seeded_private_repo(self, in_memory_db):
        repo = MemoryItemRepository()
        items = [
            PrivateMemoryItem(
                message_id=f"pf_{i}",
                sender="user_a",
                content=Message.create().text(f"private {i}"),
                created_at=i * 100,
                agent_id="agent_1",
                user_id="user_a",
            )
            for i in range(1, 6)  # created_at: 100, 200, 300, 400, 500
        ]
        await repo.save_many(items)
        return repo

    @pytest.mark.asyncio
    async def test_get_in_private_by_user_before(self, seeded_private_repo):
        repo = seeded_private_repo
        result = await repo.get_in_private_by_user_before(
            "user_a", "agent_1", timestamp=300
        )
        ids = {item.message_id for item in result}
        assert ids == {"pf_1", "pf_2"}

    @pytest.mark.asyncio
    async def test_get_in_private_by_user_before_with_limit(self, seeded_private_repo):
        repo = seeded_private_repo
        result = await repo.get_in_private_by_user_before(
            "user_a", "agent_1", timestamp=500, limit=2
        )
        assert len(result) == 2
        ids = {item.message_id for item in result}
        assert ids == {"pf_4", "pf_3"}

    @pytest.mark.asyncio
    async def test_get_in_private_by_user_before_empty(self, seeded_private_repo):
        repo = seeded_private_repo
        result = await repo.get_in_private_by_user_before(
            "user_a", "agent_1", timestamp=50
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_get_in_private_by_user_before_excludes_boundary(
        self, seeded_private_repo
    ):
        """timestamp=200 时不应包含 created_at=200 的消息（严格小于）"""
        repo = seeded_private_repo
        result = await repo.get_in_private_by_user_before(
            "user_a", "agent_1", timestamp=200
        )
        ids = {item.message_id for item in result}
        assert ids == {"pf_1"}


class TestMemoryItemSaveMany:
    """MemoryItemRepository save_many 测试"""

    @pytest.mark.asyncio
    async def test_save_many_empty(self, in_memory_db):
        repo = MemoryItemRepository()
        count = await repo.save_many([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_many_mixed_types(self, in_memory_db):
        repo = MemoryItemRepository()
        items = [
            PrivateMemoryItem(
                message_id="mix_p1",
                sender="user_a",
                content=Message.create().text("private"),
                created_at=10,
                agent_id="agent_1",
                user_id="user_a",
            ),
            GroupMemoryItem(
                message_id="mix_g1",
                sender="user_b",
                content=Message.create().text("group"),
                created_at=20,
                agent_id="agent_1",
                group_id="group_1",
            ),
        ]
        count = await repo.save_many(items)
        assert count == 2

        p = await repo.get_by_message_id("mix_p1")
        g = await repo.get_by_message_id("mix_g1")
        assert isinstance(p, PrivateMemoryItem)
        assert isinstance(g, GroupMemoryItem)


class TestMemoryItemEmptyQueries:
    """空结果集查询测试"""

    @pytest.mark.asyncio
    async def test_empty_agent(self, in_memory_db):
        repo = MemoryItemRepository()
        assert await repo.list_by_agent("nonexistent") == []

    @pytest.mark.asyncio
    async def test_empty_group(self, in_memory_db):
        repo = MemoryItemRepository()
        assert await repo.get_by_group("no_group", "no_agent") == []

    @pytest.mark.asyncio
    async def test_get_nonexistent_id(self, in_memory_db):
        repo = MemoryItemRepository()
        assert await repo.get_by_message_id("does_not_exist") is None


# ===================== UserPersonaRepository 扩展测试 =====================


class TestUserPersonaGetOrCreate:
    """UserPersonaRepository get_or_create 幂等性测试"""

    @pytest.mark.asyncio
    async def test_get_or_create_idempotent(self, in_memory_db):
        repo = UserPersonaRepository()
        p1 = await repo.get_or_create("user_idem")
        p2 = await repo.get_or_create("user_idem")
        assert p1.user_id == p2.user_id
        assert p1.version == p2.version

    @pytest.mark.asyncio
    async def test_get_or_create_new_user(self, in_memory_db):
        repo = UserPersonaRepository()
        profile = await repo.get_or_create("brand_new_user")
        assert profile.user_id == "brand_new_user"
        assert profile.version == 0


class TestUserPersonaPartialUpdate:
    """UserPersonaRepository 部分更新测试"""

    @pytest.mark.asyncio
    async def test_update_only_note(self, in_memory_db):
        repo = UserPersonaRepository()
        await repo.get_or_create("user_partial")
        updated = await repo.update_persona(user_id="user_partial", note="test note")
        assert updated.note == "test note"
        assert updated.interaction_style is None
        assert updated.version == 1

    @pytest.mark.asyncio
    async def test_sequential_updates(self, in_memory_db):
        repo = UserPersonaRepository()
        await repo.get_or_create("user_seq")
        await repo.update_persona(user_id="user_seq", impression="first")
        updated = await repo.update_persona(user_id="user_seq", other="second")
        assert updated.impression == "first"
        assert updated.other == "second"
        assert updated.version == 2

    @pytest.mark.asyncio
    async def test_update_creates_if_not_exist(self, in_memory_db):
        repo = UserPersonaRepository()
        updated = await repo.update_persona(
            user_id="auto_created", interaction_style="quiet"
        )
        assert updated.user_id == "auto_created"
        assert updated.interaction_style == "quiet"
        assert updated.version == 1


class TestUserPersonaText:
    """UserPersonaRepository get_persona_text 测试"""

    @pytest.mark.asyncio
    async def test_persona_text_not_found(self, in_memory_db):
        repo = UserPersonaRepository()
        text = await repo.get_persona_text(user_id="ref", real_user_id="ghost")
        assert text is None

    @pytest.mark.asyncio
    async def test_persona_text_found(self, in_memory_db):
        repo = UserPersonaRepository()
        await repo.update_persona(user_id="user_text", impression="很好")
        text = await repo.get_persona_text(user_id="显示名", real_user_id="user_text")
        assert text is not None
        assert "整体印象" in text

    @pytest.mark.asyncio
    async def test_persona_text_requires_params(self, in_memory_db):
        repo = UserPersonaRepository()
        with pytest.raises(ValueError):
            await repo.get_persona_text(user_id=None, real_user_id=None)


# ===================== ScheduledMessageRepository 扩展测试 =====================


def _make_scheduled_msg(
    job_id: str,
    status: ScheduledMessageStatus = ScheduledMessageStatus.PENDING,
    trigger_time: int = 1000,
    agent_id: str = "agent_1",
    user_id: str = "user_1",
    group_id: str | None = None,
    executed_at: int | None = None,
) -> ScheduledMessage:
    return ScheduledMessage(
        job_id=job_id,
        target_data={"platform": "test"},
        user_id=user_id,
        group_id=group_id,
        agent_id=agent_id,
        messages=[{"type": "text", "content": f"msg_{job_id}"}],
        trigger_time=trigger_time,
        status=status,
        created_at=trigger_time - 100,
        executed_at=executed_at,
        func_type=ScheduledFunctionType.STATIC_MESSAGE,
    )


class TestScheduledMessageMarkOperations:
    """ScheduledMessageRepository mark 操作测试"""

    @pytest.mark.asyncio
    async def test_mark_completed(self, in_memory_db):
        repo = ScheduledMessageRepository()
        msg = _make_scheduled_msg("j_complete")
        await repo.save(msg)

        result = await repo.mark_completed("j_complete")
        assert result is not None
        assert result.status == "completed"
        assert result.executed_at is not None

    @pytest.mark.asyncio
    async def test_mark_missed(self, in_memory_db):
        repo = ScheduledMessageRepository()
        msg = _make_scheduled_msg("j_missed")
        await repo.save(msg)

        result = await repo.mark_missed("j_missed")
        assert result is not None
        assert result.status == "missed"

    @pytest.mark.asyncio
    async def test_mark_canceled(self, in_memory_db):
        repo = ScheduledMessageRepository()
        msg = _make_scheduled_msg("j_cancel")
        await repo.save(msg)

        result = await repo.mark_canceled("j_cancel")
        assert result is not None
        assert result.status == "canceled"

    @pytest.mark.asyncio
    async def test_mark_nonexistent_returns_none(self, in_memory_db):
        repo = ScheduledMessageRepository()
        result = await repo.mark_completed("nonexistent_job")
        assert result is None


class TestScheduledMessageBatchMiss:
    """ScheduledMessageRepository batch_mark_missed 测试"""

    @pytest.mark.asyncio
    async def test_batch_mark_missed(self, in_memory_db):
        repo = ScheduledMessageRepository()
        for jid, tt in [("bm_1", 100), ("bm_2", 200), ("bm_3", 500)]:
            await repo.save(_make_scheduled_msg(jid, trigger_time=tt))

        count = await repo.batch_mark_missed(cutoff=300)
        assert count == 2

        m1 = await repo.get_by_job_id("bm_1")
        m3 = await repo.get_by_job_id("bm_3")
        assert m1 is not None and m1.status == "missed"
        assert m3 is not None and m3.status == "pending"

    @pytest.mark.asyncio
    async def test_batch_mark_missed_none_expired(self, in_memory_db):
        repo = ScheduledMessageRepository()
        await repo.save(_make_scheduled_msg("bm_none", trigger_time=99999))
        count = await repo.batch_mark_missed(cutoff=100)
        assert count == 0


class TestScheduledMessageQueryFilters:
    """ScheduledMessageRepository 查询过滤测试"""

    @pytest.fixture
    async def seeded_repo(self, in_memory_db):
        repo = ScheduledMessageRepository()
        msgs = [
            _make_scheduled_msg("q_1", user_id="u1", group_id="g1", trigger_time=100),
            _make_scheduled_msg("q_2", user_id="u1", group_id="g1", trigger_time=200),
            _make_scheduled_msg("q_3", user_id="u2", group_id="g2", trigger_time=300),
            _make_scheduled_msg(
                "q_4",
                user_id="u1",
                group_id="g1",
                status=ScheduledMessageStatus.COMPLETED,
                trigger_time=50,
                executed_at=60,
            ),
        ]
        for m in msgs:
            await repo.save(m)
        return repo

    @pytest.mark.asyncio
    async def test_list_by_group(self, seeded_repo):
        result = await seeded_repo.list_by_group("g1", "agent_1")
        job_ids = {m.job_id for m in result}
        assert "q_1" in job_ids
        assert "q_2" in job_ids
        assert "q_4" in job_ids
        assert "q_3" not in job_ids

    @pytest.mark.asyncio
    async def test_list_by_group_with_status(self, seeded_repo):
        result = await seeded_repo.list_by_group("g1", "agent_1", status="pending")
        for m in result:
            assert m.status == "pending"

    @pytest.mark.asyncio
    async def test_list_by_user(self, seeded_repo):
        result = await seeded_repo.list_by_user("u1", "agent_1")
        for m in result:
            assert m.user_id == "u1"

    @pytest.mark.asyncio
    async def test_list_by_status(self, seeded_repo):
        result = await seeded_repo.list_by_status("completed")
        assert len(result) == 1
        assert result[0].job_id == "q_4"

    @pytest.mark.asyncio
    async def test_list_pending_by_agent(self, seeded_repo):
        result = await seeded_repo.list_pending(agent_id="agent_1")
        for m in result:
            assert m.status == "pending"


class TestScheduledMessageDelete:
    """ScheduledMessageRepository 删除操作测试"""

    @pytest.mark.asyncio
    async def test_delete_by_job_id(self, in_memory_db):
        repo = ScheduledMessageRepository()
        await repo.save(_make_scheduled_msg("to_del"))
        assert await repo.get_by_job_id("to_del") is not None

        deleted = await repo.delete_by_job_id("to_del")
        assert deleted is True
        assert await repo.get_by_job_id("to_del") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, in_memory_db):
        repo = ScheduledMessageRepository()
        deleted = await repo.delete_by_job_id("ghost_job")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_cleanup_completed(self, in_memory_db):
        repo = ScheduledMessageRepository()
        await repo.save(
            _make_scheduled_msg(
                "cl_1",
                status=ScheduledMessageStatus.COMPLETED,
                executed_at=50,
                trigger_time=40,
            )
        )
        await repo.save(
            _make_scheduled_msg(
                "cl_2",
                status=ScheduledMessageStatus.CANCELED,
                executed_at=80,
                trigger_time=70,
            )
        )
        await repo.save(_make_scheduled_msg("cl_3", trigger_time=200))

        cleaned = await repo.cleanup_completed(before=100)
        assert cleaned == 2

        assert await repo.get_by_job_id("cl_3") is not None


# ===================== BaseRepository 基础测试 =====================


class TestBaseRepositoryOperations:
    """BaseRepository get_by_id / save 测试（通过 UserPersona）"""

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, in_memory_db):
        repo = UserPersonaRepository()
        result = await repo.get_by_id("no_such_user")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, in_memory_db):
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        repo = UserPersonaRepository()
        persona = UserPersona(user_id="base_test", impression="test", version=0)
        saved = await repo.save(persona)
        assert saved.user_id == "base_test"

        fetched = await repo.get_by_id("base_test")
        assert fetched is not None
        assert fetched.impression == "test"

    @pytest.mark.asyncio
    async def test_save_upsert(self, in_memory_db):
        """测试 save 的 upsert 行为"""
        from nonebot_plugin_wtfllm.db.models.user_persona import UserPersona

        repo = UserPersonaRepository()
        p1 = UserPersona(user_id="upsert_test", impression="v1", version=1)
        await repo.save(p1)

        p2 = UserPersona(user_id="upsert_test", impression="v2", version=2)
        await repo.save(p2)

        fetched = await repo.get_by_id("upsert_test")
        assert fetched is not None
        assert fetched.impression == "v2"
        assert fetched.version == 2


# ===================== MemoryItemRepository 扩展: delete_many / save_many 错误路径 =====================


class TestMemoryItemDeleteMany:
    """MemoryItemRepository.delete_many_by_message_ids 测试"""

    @pytest.mark.asyncio
    async def test_delete_many_empty(self, in_memory_db):
        repo = MemoryItemRepository()
        count = await repo.delete_many_by_message_ids([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_many_existing(self, in_memory_db):
        repo = MemoryItemRepository()
        items = [
            PrivateMemoryItem(
                message_id=f"dm_{i}",
                sender="user_a",
                content=Message.create().text(f"del {i}"),
                created_at=i * 10,
                agent_id="agent_1",
                user_id="user_a",
            )
            for i in range(5)
        ]
        await repo.save_many(items)

        count = await repo.delete_many_by_message_ids(["dm_0", "dm_2", "dm_4"])
        assert count == 3

        remaining = await repo.list_by_agent("agent_1")
        ids = {item.message_id for item in remaining}
        assert ids == {"dm_1", "dm_3"}

    @pytest.mark.asyncio
    async def test_delete_many_nonexistent(self, in_memory_db):
        repo = MemoryItemRepository()
        count = await repo.delete_many_by_message_ids(["nonexistent_1", "nonexistent_2"])
        assert count == 0


class TestMemoryItemSaveManyDuplicate:
    """MemoryItemRepository.save_many 重复插入测试"""

    @pytest.mark.asyncio
    async def test_save_many_duplicate_fallback(self, in_memory_db):
        """重复 message_id 时回退到逐条 merge"""
        repo = MemoryItemRepository()
        item = PrivateMemoryItem(
            message_id="dup_1",
            sender="user_a",
            content=Message.create().text("original"),
            created_at=10,
            agent_id="agent_1",
            user_id="user_a",
        )
        await repo.save_many([item])

        # 再次保存同 message_id 的项（触发 IntegrityError 回退）
        item2 = PrivateMemoryItem(
            message_id="dup_1",
            sender="user_a",
            content=Message.create().text("updated"),
            created_at=20,
            agent_id="agent_1",
            user_id="user_a",
        )
        count = await repo.save_many([item2])
        assert count >= 1

        fetched = await repo.get_by_message_id("dup_1")
        assert fetched is not None


class TestMemoryItemTimestampBefore:
    """MemoryItemRepository.get_by_timestamp_before 测试"""

    @pytest.mark.asyncio
    async def test_get_by_timestamp_before(self, in_memory_db):
        repo = MemoryItemRepository()
        items = [
            GroupMemoryItem(
                message_id=f"ts_{i}",
                sender="user_b",
                content=Message.create().text(f"ts {i}"),
                created_at=i * 100,
                agent_id="agent_1",
                group_id="group_1",
            )
            for i in range(1, 6)  # 100, 200, 300, 400, 500
        ]
        await repo.save_many(items)

        result = await repo.get_by_timestamp_before(300, "agent_1")
        ids = {item.message_id for item in result}
        assert "ts_1" in ids
        assert "ts_2" in ids
        assert "ts_3" in ids
        assert "ts_4" not in ids
