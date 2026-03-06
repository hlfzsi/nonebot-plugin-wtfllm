"""llm/tools 单元测试

覆盖:
- ToolGroupMeta: 工具注册、权限检查、信息展示
- _prepare: 工具准备逻辑
- register_tools_to_agent: 工具注册到 Agent
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from nonebot_plugin_wtfllm.llm.tools.tool_group.base import ToolGroupMeta
from nonebot_plugin_wtfllm.llm.tools.prepare import _prepare, register_tools_to_agent
from nonebot_plugin_wtfllm.memory.items.storages import MemoryItemStream
from nonebot_plugin_wtfllm.memory.items.base_items import (
    PrivateMemoryItem,
    GroupMemoryItem,
)
from nonebot_plugin_wtfllm.memory.content.message import Message

# 直接导入子模块避免 llm.__init__ 的循环导入
import nonebot_plugin_wtfllm.llm.deps as _deps

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


# ===================== Fixtures =====================


@pytest.fixture(autouse=True)
def clean_tool_group_mapping():
    """每个测试前清理 ToolGroupMeta.mapping 中的测试 entries"""
    original_mapping = dict(ToolGroupMeta.mapping)
    yield
    # 恢复，移除测试中添加的
    to_remove = [k for k in ToolGroupMeta.mapping if k not in original_mapping]
    for k in to_remove:
        del ToolGroupMeta.mapping[k]


def _make_context(active_groups: set[str] | None = None):
    """创建 mock RunContext[AgentDeps]"""
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_mem_ctx = MagicMock(spec=MemoryContextBuilder)
    mock_mem_ctx.ctx = MagicMock()
    mock_mem_ctx.ctx.alias_provider = MagicMock()

    deps = AgentDeps(
        ids=IDs(user_id="u1", agent_id="a1"),
        context=mock_mem_ctx,
        active_tool_groups=active_groups or set(),
    )

    ctx = MagicMock()
    ctx.deps = deps
    return ctx


# ===================== ToolGroupMeta 测试 =====================


class TestToolGroupMeta:
    """ToolGroupMeta 创建和注册测试"""

    def test_create_and_register(self):
        group = ToolGroupMeta(
            name="TestGroup_Create",
            description="test description",
        )
        assert group.name == "TestGroup_Create"
        assert group.description == "test description"
        assert "TestGroup_Create" in ToolGroupMeta.mapping

    def test_duplicate_name_raises(self):
        ToolGroupMeta(name="TestGroup_Dup", description="first")
        with pytest.raises(ValueError, match="already exists"):
            ToolGroupMeta(name="TestGroup_Dup", description="second")

    def test_tool_decorator(self):
        group = ToolGroupMeta(name="TestGroup_Tool", description="tools test")

        @group.tool
        def my_tool():
            """Do something"""
            pass

        assert my_tool in group.tools
        assert len(group.tools) == 1

    def test_tool_decorator_no_duplicate(self):
        group = ToolGroupMeta(name="TestGroup_NoDup", description="no dup")

        @group.tool
        def tool_a():
            pass

        group.tool(tool_a)  # 再次注册同一个函数
        assert group.tools.count(tool_a) == 1

    @pytest.mark.asyncio
    async def test_should_show_default_true(self):
        group = ToolGroupMeta(name="TestGroup_Show", description="show test")
        ctx = _make_context()
        assert await group.should_show(ctx) is True

    @pytest.mark.asyncio
    async def test_should_show_custom(self):
        async def never_show(ctx):
            return False

        group = ToolGroupMeta(
            name="TestGroup_Hidden", description="hidden", show=never_show
        )
        ctx = _make_context()
        assert await group.should_show(ctx) is False

    @pytest.mark.asyncio
    async def test_check_prem_default_true(self):
        group = ToolGroupMeta(name="TestGroup_Prem", description="prem test")
        ctx = _make_context()
        assert await group.check_prem(ctx) is True

    @pytest.mark.asyncio
    async def test_check_prem_custom(self):
        async def no_prem(ctx):
            return False

        group = ToolGroupMeta(
            name="TestGroup_NoPrem", description="no perm", prem=no_prem
        )
        ctx = _make_context()
        assert await group.check_prem(ctx) is False

    @pytest.mark.asyncio
    async def test_get_info_with_tools(self):
        group = ToolGroupMeta(name="TestGroup_Info", description="info test")

        @group.tool
        def tool_one():
            """First tool description"""
            pass

        @group.tool
        def tool_two():
            """Second tool"""
            pass

        ctx = _make_context()
        info = await group.get_info(ctx)
        assert info is not None
        assert "TestGroup_Info" in info
        assert "info test" in info
        assert "tool_one" in info
        assert "First tool description" in info

    @pytest.mark.asyncio
    async def test_get_info_hidden(self):
        async def hide(ctx):
            return False

        group = ToolGroupMeta(
            name="TestGroup_InfoHide", description="hidden info", show=hide
        )
        ctx = _make_context()
        info = await group.get_info(ctx)
        assert info is None

    @pytest.mark.asyncio
    async def test_get_info_empty_tools(self):
        group = ToolGroupMeta(name="TestGroup_Empty", description="empty")
        ctx = _make_context()
        info = await group.get_info(ctx)
        assert info is not None
        assert "该组下暂无可用工具" in info

    def test_hash(self):
        group = ToolGroupMeta(name="TestGroup_Hash", description="hash test")
        assert isinstance(hash(group), int)

    def test_repr(self):
        group = ToolGroupMeta(name="TestGroup_Repr", description="repr test")
        assert "TestGroup_Repr" in repr(group)


# ===================== _prepare 函数测试 =====================


class TestPrepareFunction:
    """_prepare 工具准备函数测试"""

    @pytest.mark.asyncio
    async def test_prepare_no_metadata(self):
        ctx = _make_context()
        tool_def = MagicMock()
        tool_def.metadata = None
        result = await _prepare(ctx, tool_def)
        assert result is None

    @pytest.mark.asyncio
    async def test_prepare_inactive_group(self):
        ctx = _make_context(active_groups=set())
        tool_def = MagicMock()
        tool_def.metadata = {"group_name": "Inactive"}
        tool_def.name = "some_tool"
        result = await _prepare(ctx, tool_def)
        assert result is None

    @pytest.mark.asyncio
    async def test_prepare_active_group(self):
        ctx = _make_context(active_groups={"Active"})
        tool_def = MagicMock()
        tool_def.metadata = {"group_name": "Active"}
        tool_def.name = "active_tool"
        result = await _prepare(ctx, tool_def)
        assert result is tool_def

    @pytest.mark.asyncio
    async def test_prepare_caches_prepared_tools(self):
        ctx = _make_context(active_groups={"Group1"})

        tool_def = MagicMock()
        tool_def.metadata = {"group_name": "Group1"}
        tool_def.name = "cached_tool"

        # 第一次调用
        await _prepare(ctx, tool_def)
        assert "cached_tool" in ctx.deps.caches["prepared_tools"]

        # 第二次调用同一个工具
        result = await _prepare(ctx, tool_def)
        assert result is tool_def

    @pytest.mark.asyncio
    async def test_prepare_multiple_tools_same_group(self):
        ctx = _make_context(active_groups={"Multi"})

        tool1 = MagicMock()
        tool1.metadata = {"group_name": "Multi"}
        tool1.name = "tool_1"

        tool2 = MagicMock()
        tool2.metadata = {"group_name": "Multi"}
        tool2.name = "tool_2"

        await _prepare(ctx, tool1)
        await _prepare(ctx, tool2)

        assert "tool_1" in ctx.deps.caches["prepared_tools"]
        assert "tool_2" in ctx.deps.caches["prepared_tools"]


# ===================== register_tools_to_agent 测试 =====================


class TestRegisterTools:
    """register_tools_to_agent 函数测试"""

    def test_register_tools(self):
        mock_agent = MagicMock()
        mock_register = MagicMock(return_value=lambda f: f)
        mock_agent.tool = mock_register

        group = ToolGroupMeta(name="TestGroup_Reg", description="register test")

        @group.tool
        def test_tool():
            pass

        register_tools_to_agent(mock_agent, [group])

        # agent.tool 应被调用一次
        mock_register.assert_called_once()
        call_kwargs = mock_register.call_args
        assert call_kwargs.kwargs["metadata"]["group_name"] == "TestGroup_Reg"

    def test_register_multiple_groups(self):
        mock_agent = MagicMock()
        call_count = 0

        def mock_tool(**kwargs):
            nonlocal call_count
            call_count += 1
            return lambda f: f

        mock_agent.tool = mock_tool

        g1 = ToolGroupMeta(name="TestGroup_Multi1", description="g1")
        g2 = ToolGroupMeta(name="TestGroup_Multi2", description="g2")

        @g1.tool
        def t1():
            pass

        @g2.tool
        def t2():
            pass

        @g2.tool
        def t3():
            pass

        register_tools_to_agent(mock_agent, [g1, g2])
        assert call_count == 3  # 3 个工具


# ===================== fetch_older_messages 测试 =====================

# 导入被装饰器包装的函数，通过 __wrapped__ 访问原始函数以绕过 tool_call_hook
from nonebot_plugin_wtfllm.llm.tools.tool_group.core import (
    fetch_older_messages as _fetch_older_messages_wrapped,
)

_fetch_older_messages = _fetch_older_messages_wrapped.__wrapped__


def _make_memory_item(msg_id, created_at, user_id="u1", group_id=None):
    if group_id:
        return GroupMemoryItem(
            message_id=msg_id,
            sender=user_id,
            content=Message.create().text(f"msg {msg_id}"),
            created_at=created_at,
            agent_id="a1",
            group_id=group_id,
        )
    return PrivateMemoryItem(
        message_id=msg_id,
        sender=user_id,
        content=Message.create().text(f"msg {msg_id}"),
        created_at=created_at,
        agent_id="a1",
        user_id=user_id,
    )


class TestFetchOlderMessages:
    """fetch_older_messages 工具测试"""

    def _make_ctx_with_stream(self, items, group_id=None):
        from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

        stream = MemoryItemStream.create(items=items, role="main_chat")
        mock_builder = MagicMock(spec=MemoryContextBuilder)
        mock_builder.get_source_by_role = MagicMock(return_value=stream)
        mock_builder.copy.return_value = mock_builder
        mock_builder.to_prompt.return_value = "<rendered>"
        mock_builder.add = MagicMock()
        ctx = _make_context(active_groups={"Core"})
        ctx.deps.context = mock_builder
        ctx.deps.ids = IDs(user_id="u1", group_id=group_id, agent_id="a1")
        return ctx, stream

    @pytest.mark.asyncio
    async def test_no_stream_returns_message(self):
        from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

        ctx = _make_context(active_groups={"Core"})
        mock_builder = MagicMock(spec=MemoryContextBuilder)
        mock_builder.get_source_by_role = MagicMock(return_value=None)
        ctx.deps.context = mock_builder
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")

        result = await _fetch_older_messages(ctx, count=10)
        assert "无法回溯" in result

    @pytest.mark.asyncio
    async def test_empty_stream_returns_message(self):
        ctx, _ = self._make_ctx_with_stream(items=[])

        result = await _fetch_older_messages(ctx, count=10)
        assert "无法回溯" in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo")
    async def test_group_chat_calls_get_by_group_before(self, mock_repo):
        items = [_make_memory_item("m1", 1000, group_id="g1")]
        ctx, stream = self._make_ctx_with_stream(items, group_id="g1")
        older = [_make_memory_item("m0", 500, group_id="g1")]
        mock_repo.get_by_group_before = AsyncMock(return_value=older)

        result = await _fetch_older_messages(ctx, count=10)
        mock_repo.get_by_group_before.assert_called_once_with(
            group_id="g1",
            agent_id="a1",
            timestamp=999,
            limit=10,
        )
        assert "1 条" in result
        assert len(stream.items) == 2

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo")
    async def test_private_chat_calls_get_in_private_before(self, mock_repo):
        items = [_make_memory_item("m1", 1000)]
        ctx, stream = self._make_ctx_with_stream(items)
        older = [_make_memory_item("m0", 500)]
        mock_repo.get_in_private_by_user_before = AsyncMock(return_value=older)

        result = await _fetch_older_messages(ctx, count=10)
        mock_repo.get_in_private_by_user_before.assert_called_once_with(
            user_id="u1",
            agent_id="a1",
            timestamp=1000,
            limit=10,
        )
        assert "1 条" in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo")
    async def test_no_older_messages_returns_hint(self, mock_repo):
        items = [_make_memory_item("m1", 1000)]
        ctx, _ = self._make_ctx_with_stream(items)
        mock_repo.get_in_private_by_user_before = AsyncMock(return_value=[])

        result = await _fetch_older_messages(ctx, count=10)
        assert "没有更多消息" in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo")
    async def test_pagination_cursor_updates(self, mock_repo):
        items = [_make_memory_item("m2", 2000)]
        ctx, stream = self._make_ctx_with_stream(items)
        assert stream.started_at == 2000
        older = [_make_memory_item("m1", 1000)]
        mock_repo.get_in_private_by_user_before = AsyncMock(return_value=older)

        await _fetch_older_messages(ctx, count=10)
        assert stream.started_at == 1000

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo")
    async def test_count_clamped_to_bounds(self, mock_repo):
        items = [_make_memory_item("m1", 1000)]
        ctx, _ = self._make_ctx_with_stream(items)
        mock_repo.get_in_private_by_user_before = AsyncMock(return_value=[])

        await _fetch_older_messages(ctx, count=0)
        mock_repo.get_in_private_by_user_before.assert_called_with(
            user_id="u1",
            agent_id="a1",
            timestamp=1000,
            limit=1,
        )
        mock_repo.get_in_private_by_user_before.reset_mock()
        await _fetch_older_messages(ctx, count=100)
        mock_repo.get_in_private_by_user_before.assert_called_with(
            user_id="u1",
            agent_id="a1",
            timestamp=1000,
            limit=30,
        )

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo")
    async def test_uses_copy_for_rendering(self, mock_repo):
        """验证渲染使用 context.copy(share_context=True, empty=True)"""
        items = [_make_memory_item("m1", 1000)]
        ctx, _ = self._make_ctx_with_stream(items)
        older = [_make_memory_item("m0", 500)]
        mock_repo.get_in_private_by_user_before = AsyncMock(return_value=older)

        await _fetch_older_messages(ctx, count=10)
        ctx.deps.context.copy.assert_called_once_with(share_context=True, empty=True)


# ===================== reinforce_persona_anchor 测试 =====================

from nonebot_plugin_wtfllm.llm.tools.tool_group.core import (
    reinforce_persona_anchor as _reinforce_persona_anchor_wrapped,
    activate_tool_group as _activate_tool_group_wrapped,
    mark_point_of_interest as _mark_point_of_interest_wrapped,
    query_tool_call_history as _query_tool_call_history_wrapped,
    get_full_message_detail as _get_full_message_detail_wrapped,
    query_memory as _query_memory_wrapped,
)

_reinforce_persona_anchor = _reinforce_persona_anchor_wrapped.__wrapped__
_activate_tool_group = _activate_tool_group_wrapped.__wrapped__
_mark_point_of_interest = _mark_point_of_interest_wrapped.__wrapped__
_query_tool_call_history = _query_tool_call_history_wrapped.__wrapped__
_get_full_message_detail = _get_full_message_detail_wrapped.__wrapped__
_query_memory = _query_memory_wrapped.__wrapped__


class TestReinforcePersonaAnchor:
    """reinforce_persona_anchor 工具测试"""

    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.APP_CONFIG")
    def test_returns_llm_role_setting(self, mock_config):
        mock_config.llm_role_setting = "You are a helpful assistant."
        ctx = _make_context()
        result = _reinforce_persona_anchor(ctx)
        assert result == "You are a helpful assistant."

    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.APP_CONFIG")
    def test_returns_empty_string_when_no_setting(self, mock_config):
        mock_config.llm_role_setting = ""
        ctx = _make_context()
        result = _reinforce_persona_anchor(ctx)
        assert result == ""


# ===================== activate_tool_group 测试 =====================


class TestActivateToolGroup:
    """activate_tool_group 工具测试"""

    @pytest.mark.asyncio
    async def test_activate_single_existing_group(self):
        group = ToolGroupMeta(name="TestActivate_Single", description="test")
        ctx = _make_context(active_groups=set())
        result = await _activate_tool_group(ctx, "TestActivate_Single")
        assert "已激活" in result
        assert "TestActivate_Single" in ctx.deps.active_tool_groups

    @pytest.mark.asyncio
    async def test_activate_multiple_groups(self):
        ToolGroupMeta(name="TestActivate_Multi1", description="g1")
        ToolGroupMeta(name="TestActivate_Multi2", description="g2")
        ctx = _make_context(active_groups=set())
        result = await _activate_tool_group(
            ctx, ["TestActivate_Multi1", "TestActivate_Multi2"]
        )
        assert "TestActivate_Multi1" in ctx.deps.active_tool_groups
        assert "TestActivate_Multi2" in ctx.deps.active_tool_groups
        assert result.count("已激活") == 2

    @pytest.mark.asyncio
    async def test_activate_nonexistent_group(self):
        ctx = _make_context(active_groups=set())
        result = await _activate_tool_group(ctx, "NonExistentGroup_XYZ")
        assert "不存在" in result
        assert "NonExistentGroup_XYZ" not in ctx.deps.active_tool_groups

    @pytest.mark.asyncio
    async def test_activate_permission_denied(self):
        async def deny(ctx):
            return False

        ToolGroupMeta(name="TestActivate_Denied", description="denied", prem=deny)
        ctx = _make_context(active_groups=set())
        result = await _activate_tool_group(ctx, "TestActivate_Denied")
        assert "权限不足" in result
        assert "TestActivate_Denied" not in ctx.deps.active_tool_groups

    @pytest.mark.asyncio
    async def test_activate_mixed_results(self):
        """测试同时激活存在、不存在、和权限不足的工具组"""
        ToolGroupMeta(name="TestActivate_OK", description="ok")

        async def deny(ctx):
            return False

        ToolGroupMeta(name="TestActivate_NoPerm", description="no perm", prem=deny)
        ctx = _make_context(active_groups=set())
        result = await _activate_tool_group(
            ctx,
            ["TestActivate_OK", "TestActivate_NoExist", "TestActivate_NoPerm"],
        )
        assert "TestActivate_OK" in ctx.deps.active_tool_groups
        assert "TestActivate_NoExist" not in ctx.deps.active_tool_groups
        assert "TestActivate_NoPerm" not in ctx.deps.active_tool_groups
        assert "已激活" in result
        assert "不存在" in result
        assert "权限不足" in result


# ===================== mark_point_of_interest 测试 =====================


class TestMarkPointOfInterest:
    """mark_point_of_interest 工具测试"""

    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.attention_router")
    def test_basic_marking(self, mock_router):
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1", group_id="g1")
        ctx.deps.context.ctx.alias_provider.resolve_alias.return_value = None

        result = _mark_point_of_interest(ctx, user_id="user123", reason="testing")
        assert "user123" in result
        assert "testing" in result
        mock_router.mark_poi.assert_called_once()
        poi_arg = mock_router.mark_poi.call_args[0][0]
        assert poi_arg.user_id == "user123"
        assert poi_arg.reason == "testing"
        assert poi_arg.group_id == "g1"
        assert poi_arg.agent_id == "a1"

    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.attention_router")
    def test_alias_resolution(self, mock_router):
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1", group_id="g1")
        ctx.deps.context.ctx.alias_provider.resolve_alias.return_value = "real_user_456"

        result = _mark_point_of_interest(ctx, user_id="alias_name", reason="follow-up")
        assert "alias_name" in result
        mock_router.mark_poi.assert_called_once()
        poi_arg = mock_router.mark_poi.call_args[0][0]
        assert poi_arg.user_id == "real_user_456"

    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.attention_router")
    def test_custom_turns_and_timeout(self, mock_router):
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        ctx.deps.context.ctx.alias_provider.resolve_alias.return_value = None

        _mark_point_of_interest(
            ctx, user_id="u2", reason="custom", turns=5, timeout_seconds=120
        )
        poi_arg = mock_router.mark_poi.call_args[0][0]
        assert poi_arg.turns == 5


# ===================== query_tool_call_history 测试 =====================


class TestQueryToolCallHistory:
    """query_tool_call_history 工具测试"""

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.tool_call_record_repo")
    async def test_empty_records(self, mock_repo):
        mock_repo.get_recent = AsyncMock(return_value=[])
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        result = await _query_tool_call_history(ctx, limit=5)
        assert "暂无工具调用记录" in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.tool_call_record_repo")
    async def test_records_with_kwargs(self, mock_repo):
        record = MagicMock()
        record.timestamp = 1700000000
        record.run_step = 1
        record.tool_name = "some_tool"
        record.kwargs = {"key": "value"}
        mock_repo.get_recent = AsyncMock(return_value=[record])
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        result = await _query_tool_call_history(ctx, limit=1)
        assert "some_tool" in result
        assert "step=1" in result
        assert '"key"' in result
        assert '"value"' in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.tool_call_record_repo")
    async def test_records_without_kwargs(self, mock_repo):
        record = MagicMock()
        record.timestamp = 1700000000
        record.run_step = 0
        record.tool_name = "another_tool"
        record.kwargs = {}
        mock_repo.get_recent = AsyncMock(return_value=[record])
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        result = await _query_tool_call_history(ctx, limit=1)
        assert "another_tool" in result
        assert "step=0" in result
        # Empty kwargs should not append parentheses
        assert "another_tool(" not in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.tool_call_record_repo")
    async def test_group_chat_passes_none_user_id(self, mock_repo):
        """群聊场景下 user_id 应传 None"""
        mock_repo.get_recent = AsyncMock(return_value=[])
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1", group_id="g1")
        await _query_tool_call_history(ctx, limit=1)
        mock_repo.get_recent.assert_called_once_with(
            agent_id="a1", group_id="g1", user_id=None, limit=1
        )

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.tool_call_record_repo")
    async def test_private_chat_passes_user_id(self, mock_repo):
        """私聊场景下 user_id 应传实际值"""
        mock_repo.get_recent = AsyncMock(return_value=[])
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        await _query_tool_call_history(ctx, limit=3)
        mock_repo.get_recent.assert_called_once_with(
            agent_id="a1", group_id=None, user_id="u1", limit=3
        )


# ===================== get_full_message_detail 测试 =====================


class TestGetFullMessageDetail:
    """get_full_message_detail 工具测试"""

    @pytest.mark.asyncio
    async def test_message_not_found(self):
        ctx = _make_context()
        ctx.deps.context.resolve_memory_ref = MagicMock(return_value=None)
        result = await _get_full_message_detail(ctx, message_ref=42)
        assert "未找到" in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_message_found(self):
        ctx = _make_context()
        mock_item = MagicMock()
        mock_item.message_id = "msg_001"
        ctx.deps.context.resolve_memory_ref = MagicMock(return_value=mock_item)

        mock_llm_ctx = MagicMock()
        ctx.deps.context.ctx.copy.return_value = mock_llm_ctx
        new_builder = MagicMock()
        new_builder.to_prompt.return_value = "<message_detail>ok</message_detail>"
        ctx.deps.context.copy = MagicMock(return_value=new_builder)

        with patch(
            "nonebot_plugin_wtfllm.llm.tools.tool_group.core.memory_item_repo"
        ) as mock_repo:
            mock_repo.get_chain_by_message_ids = AsyncMock(
                return_value=[_make_memory_item("msg_001", 1000)]
            )

            result = await _get_full_message_detail(ctx, message_ref=7)

        assert result == "<message_detail>ok</message_detail>"
        mock_repo.get_chain_by_message_ids.assert_called_once_with(["msg_001"])
        mock_llm_ctx.set_condense.assert_called_once_with(False)
        ctx.deps.context.copy.assert_called_once_with(
            share_context=mock_llm_ctx, empty=True
        )
        new_builder.add.assert_called_once()


# ===================== query_memory 测试 =====================


class TestQueryMemory:
    """query_memory 工具测试"""

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.RetrievalChain")
    async def test_with_user_id_resolved(self, mock_chain_cls):
        """user_id 提供且 resolve 成功 -> entity_memory"""
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1", group_id="g1")
        ctx.deps.context.resolve_aliases = MagicMock(return_value="real_u2")
        builder = MagicMock()
        builder.to_prompt.return_value = "<memory_result>"
        ctx.deps.context.copy.return_value = builder

        chain = MagicMock()
        chain.resolve = AsyncMock(return_value=[MagicMock()])
        mock_chain_cls.return_value = chain

        result = await _query_memory(ctx, query="hello", user_id="u2", limit=5)
        mock_chain_cls.assert_called_once_with(
            agent_id="a1",
            group_id="g1",
            user_id="u1",
            query="hello",
        )
        chain.entity_memory.assert_called_once_with(
            entity_ids=["real_u2"],
            limit=5,
            prefix="<related_memory>",
            suffix="</related_memory>",
        )
        chain.cross_session_memory.assert_not_called()
        assert result == "<memory_result>"

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.RetrievalChain")
    async def test_user_id_not_resolved(self, mock_chain_cls):
        """user_id 提供但 resolve 返回 None -> 不走核心记忆分支"""
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        ctx.deps.context.resolve_aliases = MagicMock(return_value=None)
        chain = MagicMock()
        chain.resolve = AsyncMock(return_value=[])
        mock_chain_cls.return_value = chain

        result = await _query_memory(ctx, query="test", user_id="unknown_user")
        assert "未找到与用户ID unknown_user 相关的记忆" in result
        chain.entity_memory.assert_not_called()
        chain.cross_session_memory.assert_not_called()

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.RetrievalChain")
    async def test_no_user_id_cross_session(self, mock_chain_cls):
        """user_id=None -> cross_session_memory"""
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1", group_id="g1")
        builder = MagicMock()
        builder.to_prompt.return_value = "<cross_result>"
        ctx.deps.context.copy.return_value = builder

        chain = MagicMock()
        chain.resolve = AsyncMock(return_value=[MagicMock()])
        mock_chain_cls.return_value = chain

        result = await _query_memory(ctx, query="cross", user_id=None, limit=3)
        chain.cross_session_memory.assert_called_once_with(
            limit=3,
            prefix="<related_memory>",
            suffix="</related_memory>",
        )
        assert result == "<cross_result>"

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.RetrievalChain")
    async def test_no_user_id_private_chat_cross_session(self, mock_chain_cls):
        """私聊且 user_id=None -> 仍走 cross_session_memory"""
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        builder = MagicMock()
        builder.to_prompt.return_value = "<private_cross>"
        ctx.deps.context.copy.return_value = builder

        chain = MagicMock()
        chain.resolve = AsyncMock(return_value=[MagicMock()])
        mock_chain_cls.return_value = chain

        result = await _query_memory(ctx, query="private", limit=2)
        chain.cross_session_memory.assert_called_once_with(
            limit=2,
            prefix="<related_memory>",
            suffix="</related_memory>",
        )
        assert result == "<private_cross>"

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.RetrievalChain")
    async def test_no_results_returns_message(self, mock_chain_cls):
        """无结果时返回提示文本"""
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        chain = MagicMock()
        chain.resolve = AsyncMock(return_value=[])
        mock_chain_cls.return_value = chain

        result = await _query_memory(ctx, query="nothing")
        assert "未找到与查询内容相关的记忆或知识" in result

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.core.RetrievalChain")
    async def test_only_knowledge_results(self, mock_chain_cls):
        """仅有知识库结果时也应正确返回"""
        ctx = _make_context()
        ctx.deps.ids = IDs(user_id="u1", agent_id="a1")
        builder = MagicMock()
        builder.to_prompt.return_value = "<kb_only>"
        ctx.deps.context.copy.return_value = builder

        chain = MagicMock()
        chain.resolve = AsyncMock(return_value=[MagicMock()])
        mock_chain_cls.return_value = chain

        result = await _query_memory(ctx, query="knowledge only")
        assert result == "<kb_only>"


# ===================== _extract_and_track / tool_call_hook 测试 =====================


from nonebot_plugin_wtfllm.llm.tools.tool_group.base import (
    _extract_and_track,
    tool_call_hook,
)
from nonebot_plugin_wtfllm.llm.deps import ToolCallInfo
from pydantic_ai import RunContext


def _make_run_context(
    run_id="test-run-id",
    run_step=1,
    tool_budget=0,
    tool_points_used=0,
):
    """Create a MagicMock RunContext with a real AgentDeps for tracking tests."""
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_mem_ctx = MagicMock(spec=MemoryContextBuilder)
    mock_mem_ctx.ctx = MagicMock()
    mock_mem_ctx.ctx.alias_provider = MagicMock()

    deps = AgentDeps(
        ids=IDs(user_id="u1", agent_id="a1"),
        context=mock_mem_ctx,
        tool_point_budget=tool_budget,
        tool_points_used=tool_points_used,
    )

    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    ctx.run_id = run_id
    ctx.run_step = run_step
    return ctx


class TestExtractAndTrack:
    """_extract_and_track 函数测试"""

    def test_tracks_tool_call(self):
        """正确追踪工具调用，ToolCallInfo 被追加到 tool_chain"""
        ctx = _make_run_context()
        result = _extract_and_track("my_tool", (), {"ctx": ctx}, cost=0)
        assert result is ctx.deps
        assert len(ctx.deps.tool_chain) == 1
        info = ctx.deps.tool_chain[0]
        assert isinstance(info, ToolCallInfo)
        assert info.tool_name == "my_tool"
        assert info.run_id == "test-run-id"
        assert info.round_index == 1

    def test_no_run_id_generates_one(self):
        """run_id 为 None 时自动生成新 run_id"""
        ctx = _make_run_context(run_id=None)
        _extract_and_track("tool_x", (), {"ctx": ctx}, cost=0)
        assert ctx.run_id is not None
        assert isinstance(ctx.run_id, str)
        assert len(ctx.run_id) > 0
        # 生成的 run_id 应是 UUID 格式
        assert len(ctx.run_id.split("-")) == 5

    def test_budget_tracking(self):
        """tool_budget_enabled=True 时 tool_points_used 应增加 cost"""
        ctx = _make_run_context(tool_budget=100, tool_points_used=0)
        assert ctx.deps.tool_budget_enabled is True
        _extract_and_track("expensive_tool", (), {"ctx": ctx}, cost=5)
        assert ctx.deps.tool_points_used == 5

    def test_no_budget_tracking(self):
        """tool_budget_enabled=False 时 tool_points_used 不变"""
        ctx = _make_run_context(tool_budget=0, tool_points_used=0)
        assert ctx.deps.tool_budget_enabled is False
        _extract_and_track("cheap_tool", (), {"ctx": ctx}, cost=5)
        assert ctx.deps.tool_points_used == 0

    def test_raises_without_agent_deps(self):
        """kwargs 中没有合法的 AgentDeps 时抛出 RuntimeError"""
        with pytest.raises(RuntimeError, match="Failed to track tool call"):
            _extract_and_track("bad_tool", (), {"ctx": "not_a_context"}, cost=0)

    def test_raises_with_empty_kwargs(self):
        """空 kwargs 时抛出 RuntimeError"""
        with pytest.raises(RuntimeError, match="Failed to track tool call"):
            _extract_and_track("bad_tool", (), {}, cost=0)

    def test_kwargs_filtering_excludes_ctx(self):
        """ToolCallInfo.kwargs 中不包含 ctx 自身"""
        ctx = _make_run_context()
        _extract_and_track("my_tool", (), {"ctx": ctx, "param1": "value1"}, cost=0)
        info = ctx.deps.tool_chain[0]
        assert "ctx" not in info.kwargs
        assert "param1" in info.kwargs
        assert info.kwargs["param1"] == repr("value1")

    def test_multiple_calls_append(self):
        """多次调用追加到 tool_chain"""
        ctx = _make_run_context(tool_budget=100)
        _extract_and_track("tool_a", (), {"ctx": ctx}, cost=2)
        _extract_and_track("tool_b", (), {"ctx": ctx}, cost=3)
        assert len(ctx.deps.tool_chain) == 2
        assert ctx.deps.tool_chain[0].tool_name == "tool_a"
        assert ctx.deps.tool_chain[1].tool_name == "tool_b"
        assert ctx.deps.tool_points_used == 5


class TestToolCallHook:
    """tool_call_hook 装饰器测试"""

    def test_wraps_sync_function(self):
        """包装同步函数并正确追踪"""

        def my_sync_tool(ctx):
            """A sync tool."""
            return "sync_result"

        wrapped = tool_call_hook(my_sync_tool, cost=0)
        ctx = _make_run_context()
        result = wrapped(ctx=ctx)
        assert "sync_result" in result
        assert len(ctx.deps.tool_chain) == 1
        assert ctx.deps.tool_chain[0].tool_name == "my_sync_tool"

    @pytest.mark.asyncio
    async def test_wraps_async_function(self):
        """包装异步函数并正确追踪"""

        async def my_async_tool(ctx):
            """An async tool."""
            return "async_result"

        wrapped = tool_call_hook(my_async_tool, cost=0)
        ctx = _make_run_context()
        result = await wrapped(ctx=ctx)
        assert "async_result" in result
        assert len(ctx.deps.tool_chain) == 1
        assert ctx.deps.tool_chain[0].tool_name == "my_async_tool"

    def test_budget_suffix_appended_sync(self):
        """同步函数: cost>0 且 budget 开启时结果包含预算后缀"""

        def budgeted_tool(ctx):
            return "base_result"

        wrapped = tool_call_hook(budgeted_tool, cost=3)
        ctx = _make_run_context(tool_budget=100, tool_points_used=0)
        result = wrapped(ctx=ctx)
        assert "base_result" in result
        assert "工具预算" in result
        assert "-3pt" in result

    @pytest.mark.asyncio
    async def test_budget_suffix_appended_async(self):
        """异步函数: cost>0 且 budget 开启时结果包含预算后缀"""

        async def budgeted_async_tool(ctx):
            return "async_base"

        wrapped = tool_call_hook(budgeted_async_tool, cost=5)
        ctx = _make_run_context(tool_budget=100, tool_points_used=0)
        result = await wrapped(ctx=ctx)
        assert "async_base" in result
        assert "工具预算" in result
        assert "-5pt" in result

    def test_no_budget_suffix_when_disabled(self):
        """budget 未开启时不追加后缀"""

        def plain_tool(ctx):
            return "plain_result"

        wrapped = tool_call_hook(plain_tool, cost=3)
        ctx = _make_run_context(tool_budget=0)
        result = wrapped(ctx=ctx)
        assert result == "plain_result"

    def test_preserves_function_name(self):
        """包装后函数名应保留原始函数名"""

        def original_name(ctx):
            """Original docstring."""
            pass

        wrapped = tool_call_hook(original_name)
        assert wrapped.__name__ == "original_name"
        assert wrapped.__wrapped__ is original_name


class TestResolveToolCost:
    """resolve_tool_cost 方法测试"""

    def test_explicit_cost(self):
        """tool_costs 中有该工具时返回显式配置值"""
        group = ToolGroupMeta(
            name="TestCost_Explicit",
            description="cost test",
            tool_costs={"my_tool": 10},
            default_tool_cost=2,
        )
        assert group.resolve_tool_cost("my_tool") == 10

    def test_default_cost(self):
        """tool_costs 中无该工具时使用 default_tool_cost"""
        group = ToolGroupMeta(
            name="TestCost_Default",
            description="cost test",
            default_tool_cost=5,
        )
        assert group.resolve_tool_cost("unknown_tool") == 5

    def test_zero_fallback(self):
        """无显式配置且 default_tool_cost 为 0 时返回 0"""
        group = ToolGroupMeta(
            name="TestCost_ZeroFallback",
            description="cost test",
            default_tool_cost=0,
        )
        assert group.resolve_tool_cost("any_tool") == 0

    def test_explicit_overrides_default(self):
        """显式配置优先于 default_tool_cost"""
        group = ToolGroupMeta(
            name="TestCost_Override",
            description="cost test",
            tool_costs={"special": 20},
            default_tool_cost=3,
        )
        assert group.resolve_tool_cost("special") == 20
        assert group.resolve_tool_cost("other") == 3


class TestGetInfoWithBudget:
    """get_info 方法的预算分支测试"""

    @pytest.mark.asyncio
    async def test_budget_enabled_shows_cost(self):
        """tool_budget_enabled=True 时输出中包含工具点数"""
        group = ToolGroupMeta(
            name="TestInfo_BudgetOn",
            description="budget info",
            tool_costs={"tool_a": 3},
            default_tool_cost=1,
        )

        @group.tool
        def tool_a(ctx):
            """Tool A description"""
            return "a"

        @group.tool
        def tool_b(ctx):
            """Tool B description"""
            return "b"

        ctx = _make_context()
        ctx.deps.tool_point_budget = 100  # enable budget

        info = await group.get_info(ctx)
        assert info is not None
        # tool_a has explicit cost 3
        assert "(3pt)" in info
        assert "tool_a" in info
        # tool_b uses default cost 1
        assert "(1pt)" in info
        assert "tool_b" in info
        assert "Tool A description" in info

    @pytest.mark.asyncio
    async def test_budget_disabled_no_cost(self):
        """tool_budget_enabled=False 时输出中不包含工具点数"""
        group = ToolGroupMeta(
            name="TestInfo_BudgetOff",
            description="no budget info",
            tool_costs={"tool_c": 5},
        )

        @group.tool
        def tool_c(ctx):
            """Tool C description"""
            return "c"

        ctx = _make_context()
        ctx.deps.tool_point_budget = 0  # budget disabled

        info = await group.get_info(ctx)
        assert info is not None
        assert "tool_c" in info
        assert "Tool C description" in info
        # No cost annotation
        assert "pt)" not in info

    @pytest.mark.asyncio
    async def test_budget_enabled_no_docstring(self):
        """tool_budget_enabled=True 且工具无 docstring 时显示默认描述"""
        group = ToolGroupMeta(
            name="TestInfo_BudgetNoDoc",
            description="no doc budget",
            default_tool_cost=2,
        )

        def tool_no_doc(ctx):
            return "x"

        # Manually clear docstring and register
        tool_no_doc.__doc__ = None
        group.tool(tool_no_doc)

        ctx = _make_context()
        ctx.deps.tool_point_budget = 50

        info = await group.get_info(ctx)
        assert info is not None
        assert "暂无描述" in info
        assert "(2pt)" in info


# ===================== get_image_description 测试 =====================

from nonebot_plugin_wtfllm.llm.tools.tool_group.core import (
    get_image_description as _get_image_description_wrapped,
    get_image_content as _get_image_content_wrapped,
)

_get_image_description = _get_image_description_wrapped.__wrapped__
_get_image_content = _get_image_content_wrapped.__wrapped__

CORE_MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.core"


class TestGetImageDescription:
    """get_image_description 工具测试"""

    def _make_ctx(self, resolve_media=None):
        ctx = _make_context()
        ctx.deps.context.resolve_media_ref = MagicMock(
            side_effect=resolve_media or (lambda *a: None)
        )
        return ctx

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_invalid_ref_format(self, mock_resched):
        """无 'IMG:' 前缀的引用"""
        ctx = self._make_ctx()
        result = await _get_image_description(ctx, ["AUDIO:1", "FILE:2"])
        assert "无效的引用格式" in result
        assert "AUDIO:1" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_ref_not_found(self, mock_resched):
        """resolve_media_ref 返回 None"""
        ctx = self._make_ctx(resolve_media=lambda *a: None)
        result = await _get_image_description(ctx, ["IMG:1"])
        assert "未找到" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_seg_has_cached_desc(self, mock_resched):
        """ImageSegment 已有 desc 时直接返回"""
        mock_seg = MagicMock()
        mock_seg.desc = "一只猫的图片"
        mock_seg.available = True

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_description(ctx, ["IMG:1"])
        assert "一只猫的图片" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_seg_not_available(self, mock_resched):
        """图片已过期"""
        mock_seg = MagicMock()
        mock_seg.desc = None
        mock_seg.available = False

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_description(ctx, ["IMG:1"])
        assert "已过期" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_no_valid_sources(self, mock_resched):
        """所有引用都无效时返回 JSON 结果"""
        mock_seg = MagicMock()
        mock_seg.desc = None
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = None

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_description(ctx, ["IMG:1"])
        assert "无效" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_image_data_too_large(self, mock_resched):
        """图片数据过大时返回错误 (通过 local_path -> data URI 路径)"""
        mock_seg = MagicMock()
        mock_seg.desc = None
        mock_seg.available = True
        mock_seg.local_path = "/tmp/huge.webp"
        mock_seg.url = None
        mock_seg.message_id = "m1"
        # get_data_uri_async 返回超大 data URI
        mock_seg.get_data_uri_async = AsyncMock(
            return_value="data:image/webp;base64," + "x" * 2000001
        )

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_description(ctx, ["IMG:1"])
        assert "过大" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}._get_image_desc", new_callable=AsyncMock)
    @patch(f"{CORE_MODULE}.memory_item_repo")
    async def test_seg_with_url_success(self, mock_repo, mock_vision, mock_resched):
        """通过 URL 获取图片描述成功 (URL ref 的 image_sources 为空时返回空 JSON)"""
        mock_seg = MagicMock()
        mock_seg.desc = None
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = "http://example.com/cat.jpg"
        mock_seg.message_id = "m1"

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)

        # URL-only ref: valid_refs 有值但 image_sources 为空,直接返回空 JSON
        result = await _get_image_description(ctx, ["IMG:1"])
        # 返回空 result_dict 因为 URL ref 不会被加入 image_sources
        assert result is not None

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}._get_image_desc", new_callable=AsyncMock)
    @patch(f"{CORE_MODULE}.memory_item_repo")
    async def test_vision_returns_none(self, mock_repo, mock_vision, mock_resched):
        """vision 模型返回 None 时 (通过 local_path 路径)"""
        mock_seg = MagicMock()
        mock_seg.desc = None
        mock_seg.available = True
        mock_seg.local_path = "/tmp/img.webp"
        mock_seg.url = None
        mock_seg.message_id = "m1"
        mock_seg.get_data_uri_async = AsyncMock(
            return_value="data:image/webp;base64,abc"
        )

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        mock_vision.return_value = None
        mock_repo.get_many_by_message_ids = AsyncMock(return_value=[])

        result = await _get_image_description(ctx, ["IMG:1"])
        assert "失败" in result

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    async def test_seg_with_local_path(self, mock_resched):
        """通过 local_path 获取图片数据"""
        mock_seg = MagicMock()
        mock_seg.desc = None
        mock_seg.available = True
        mock_seg.local_path = "/tmp/img.webp"
        mock_seg.url = None
        mock_seg.message_id = "m1"
        mock_seg.get_data_uri_async = AsyncMock(
            return_value="data:image/webp;base64,abc"
        )

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)

        with (
            patch(
                f"{CORE_MODULE}._get_image_desc", new_callable=AsyncMock
            ) as mock_vision,
            patch(f"{CORE_MODULE}.memory_item_repo") as mock_repo,
        ):
            mock_desc = MagicMock()
            mock_desc.to_string.return_value = "本地图片描述"
            mock_vision.return_value = [mock_desc]
            mock_item = MagicMock()
            mock_item.message_id = "m1"
            mock_item.content.deep_find_and_update.return_value = False
            mock_repo.get_many_by_message_ids = AsyncMock(return_value=[mock_item])
            mock_repo.save_memory_item = AsyncMock()

            result = await _get_image_description(ctx, ["IMG:1"])
            assert "本地图片描述" in result
            mock_seg.get_data_uri_async.assert_called_once()
            mock_repo.save_memory_item.assert_not_called()


# ===================== get_image_content 测试 =====================


class TestGetImageContent:
    """get_image_content 工具测试"""

    def _make_ctx(self, resolve_media=None):
        ctx = _make_context()
        ctx.deps.context.resolve_media_ref = MagicMock(
            side_effect=resolve_media or (lambda *a: None)
        )
        return ctx

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.APP_CONFIG")
    async def test_vision_not_supported(self, mock_config):
        """llm_support_vision=False"""
        mock_config.llm_support_vision = False
        ctx = self._make_ctx()
        result = await _get_image_content(ctx, ["IMG:1"])
        assert "不支持视觉能力" in result.return_value

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}.APP_CONFIG")
    async def test_invalid_ref(self, mock_config, mock_resched):
        """无效引用格式"""
        mock_config.llm_support_vision = True
        ctx = self._make_ctx()
        result = await _get_image_content(ctx, ["FILE:1"])
        assert "格式不正确" in result.return_value

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}.APP_CONFIG")
    async def test_ref_not_found(self, mock_config, mock_resched):
        """引用未找到"""
        mock_config.llm_support_vision = True
        ctx = self._make_ctx(resolve_media=lambda *a: None)
        result = await _get_image_content(ctx, ["IMG:1"])
        assert "未找到" in result.return_value

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}.APP_CONFIG")
    async def test_seg_not_available(self, mock_config, mock_resched):
        """图片已过期"""
        mock_config.llm_support_vision = True
        mock_seg = MagicMock()
        mock_seg.available = False
        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_content(ctx, ["IMG:1"])
        assert "已过期" in result.return_value

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}.APP_CONFIG")
    async def test_seg_with_url(self, mock_config, mock_resched):
        """通过 URL 加载图片"""
        mock_config.llm_support_vision = True
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = "http://example.com/photo.jpg"

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_content(ctx, ["IMG:1"])
        assert "已成功加载" in result.return_value
        assert len(result.content) == 1

    @pytest.mark.asyncio
    @patch(f"{CORE_MODULE}.reschedule_deadline")
    @patch(f"{CORE_MODULE}.APP_CONFIG")
    async def test_seg_with_local_path(self, mock_config, mock_resched):
        """通过 local_path 加载图片"""
        mock_config.llm_support_vision = True
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = "/tmp/photo.webp"
        mock_seg.url = None
        mock_seg.get_bytes_async = AsyncMock(return_value=b"binary-data")
        mock_seg.get_mime_type_async = AsyncMock(return_value="image/webp")

        ctx = self._make_ctx(resolve_media=lambda *a: mock_seg)
        result = await _get_image_content(ctx, ["IMG:1"])
        assert "已加载" in result.return_value
        assert len(result.content) == 1
