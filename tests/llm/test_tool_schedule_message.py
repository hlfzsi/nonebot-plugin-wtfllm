"""llm/tools/tool_group/schedule_message.py 单元测试"""

from datetime import datetime

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.schedule_message import (
    RelativeTime,
    AbsoluteTime,
    schedule_message as _schedule_wrapped,
    schedule_agent_task as _schedule_agent_wrapped,
    cancel_scheduled_task as _cancel_wrapped,
    list_scheduled_tasks as _list_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

_schedule = _schedule_wrapped.__wrapped__
_schedule_agent = _schedule_agent_wrapped.__wrapped__
_cancel = _cancel_wrapped.__wrapped__
_list = _list_wrapped.__wrapped__

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_fluent_unimsg():
    """创建模拟 UniMessage 的 fluent API mock，dump() 返回真实列表。"""
    mock_msg = MagicMock()
    mock_msg.text.return_value = mock_msg
    mock_msg.at.return_value = mock_msg
    mock_msg.image.return_value = mock_msg
    mock_msg.dump.return_value = [{"type": "text", "content": "mock"}]
    return mock_msg


def _make_ctx(user_id="u1", group_id=None, with_runtime=True):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id=user_id, group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"ScheduleMessage"},
    )
    if with_runtime:
        mock_runtime = MagicMock()
        mock_runtime.target.dump.return_value = {"platform": "test", "id": "t1"}
        deps.nb_runtime = mock_runtime
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.schedule_message"


# ===================== 时间模型测试 =====================


class TestRelativeTime:
    def test_trigger_timestamp_minutes(self):
        rt = RelativeTime(minutes=30)
        now_ts = int(datetime.now().timestamp())
        assert abs(rt.trigger_timestamp - (now_ts + 30 * 60)) <= 2

    def test_trigger_timestamp_hours(self):
        rt = RelativeTime(hours=2)
        now_ts = int(datetime.now().timestamp())
        assert abs(rt.trigger_timestamp - (now_ts + 2 * 3600)) <= 2

    def test_trigger_timestamp_days(self):
        rt = RelativeTime(days=1)
        now_ts = int(datetime.now().timestamp())
        assert abs(rt.trigger_timestamp - (now_ts + 86400)) <= 2

    def test_trigger_timestamp_combined(self):
        rt = RelativeTime(minutes=10, hours=1, days=1)
        now_ts = int(datetime.now().timestamp())
        expected = now_ts + 86400 + 3600 + 600
        assert abs(rt.trigger_timestamp - expected) <= 2


class TestAbsoluteTime:
    def test_trigger_timestamp(self):
        at = AbsoluteTime(date="2025-06-15", time="14:30")
        expected = int(datetime.strptime("2025-06-15 14:30", "%Y-%m-%d %H:%M").timestamp())
        assert at.trigger_timestamp == expected


# ===================== schedule_message 测试 =====================


class TestScheduleMessage:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.UniMessage")
    async def test_schedule_basic(self, MockUniMsg, mock_schedule_job):
        MockUniMsg.return_value = _make_fluent_unimsg()
        ctx = _make_ctx()
        config = RelativeTime(minutes=10)
        result = await _schedule(ctx, message="提醒", schedule_config=config)
        assert "已设置" in result
        mock_schedule_job.assert_called_once()
        # Verify the task_name is correct
        call_kwargs = mock_schedule_job.call_args
        assert call_kwargs.kwargs.get("task_name") == "send_static_message" or call_kwargs[1].get("task_name") == "send_static_message"

    @pytest.mark.asyncio
    async def test_schedule_no_runtime_raises(self):
        ctx = _make_ctx(with_runtime=False)
        ctx.deps.nb_runtime = None
        config = RelativeTime(minutes=10)
        with pytest.raises(ValueError, match="nb_runtime"):
            await _schedule(ctx, message="msg", schedule_config=config)

    @pytest.mark.asyncio
    async def test_schedule_no_user_id_raises(self):
        ctx = _make_ctx(user_id=None)
        config = RelativeTime(minutes=10)
        with pytest.raises(ValueError, match="User ID"):
            await _schedule(ctx, message="msg", schedule_config=config)


# ===================== cancel_scheduled_task 测试 =====================


class TestCancelScheduledTask:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.cancel_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_cancel_success(self, mock_repo, mock_cancel):
        mock_job = MagicMock()
        mock_job.user_id = "u1"
        mock_job.status = "pending"
        mock_repo.get_by_job_id = AsyncMock(return_value=mock_job)
        ctx = _make_ctx()
        result = await _cancel(ctx, job_id="job_1")
        assert "已取消" in result
        mock_cancel.assert_called_once_with("job_1")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_cancel_not_found(self, mock_repo):
        mock_repo.get_by_job_id = AsyncMock(return_value=None)
        ctx = _make_ctx()
        result = await _cancel(ctx, job_id="ghost")
        assert "未找到" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_cancel_wrong_user(self, mock_repo):
        mock_job = MagicMock()
        mock_job.user_id = "other_user"
        mock_job.status = "pending"
        mock_repo.get_by_job_id = AsyncMock(return_value=mock_job)
        ctx = _make_ctx()
        result = await _cancel(ctx, job_id="job_x")
        assert "没有权限" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_cancel_not_pending(self, mock_repo):
        mock_job = MagicMock()
        mock_job.user_id = "u1"
        mock_job.status = "completed"
        mock_repo.get_by_job_id = AsyncMock(return_value=mock_job)
        ctx = _make_ctx()
        result = await _cancel(ctx, job_id="job_done")
        assert "无法取消" in result



class TestListScheduledTasks:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_list_user_empty(self, mock_repo):
        mock_repo.list_by_user = AsyncMock(return_value=[])
        ctx = _make_ctx()
        result = await _list(ctx, type="user")
        assert "没有" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}._job_to_text", return_value="Job: job_1, 状态: pending")
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_list_user_with_results(self, mock_repo, mock_to_text):
        mock_job = MagicMock()
        mock_repo.list_by_user = AsyncMock(return_value=[mock_job])
        ctx = _make_ctx()
        result = await _list(ctx, type="user")
        assert "job_1" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.scheduled_job_repo")
    async def test_list_group(self, mock_repo):
        mock_repo.list_by_group = AsyncMock(return_value=[])
        ctx = _make_ctx(group_id="g1")
        result = await _list(ctx, type="group")
        assert "没有" in result

    @pytest.mark.asyncio
    async def test_list_group_no_group_id_raises(self):
        ctx = _make_ctx(group_id=None)
        with pytest.raises(ValueError, match="Group ID"):
            await _list(ctx, type="group")


# ===================== schedule_agent_task 测试 =====================


class TestScheduleAgentTask:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    async def test_schedule_agent_basic(self, mock_schedule_job):
        ctx = _make_ctx()
        ctx.deps.nb_runtime.session = MagicMock()
        ctx.deps.nb_runtime.session.dump.return_value = {"platform": "test", "id": "s1"}
        config = RelativeTime(minutes=30)
        result = await _schedule_agent(ctx, schedule_config=config, instruction="检查天气")
        assert "已设置" in result
        mock_schedule_job.assert_called_once()
        call_kwargs = mock_schedule_job.call_args
        assert call_kwargs.kwargs.get("task_name") == "invoke_agent" or call_kwargs[1].get("task_name") == "invoke_agent"

    @pytest.mark.asyncio
    async def test_schedule_agent_no_runtime_raises(self):
        ctx = _make_ctx(with_runtime=False)
        ctx.deps.nb_runtime = None
        config = RelativeTime(minutes=10)
        with pytest.raises(ValueError, match="nb_runtime"):
            await _schedule_agent(ctx, schedule_config=config, instruction="test")

    @pytest.mark.asyncio
    async def test_schedule_agent_no_user_id_raises(self):
        ctx = _make_ctx(user_id=None)
        config = RelativeTime(minutes=10)
        with pytest.raises(ValueError, match="User ID"):
            await _schedule_agent(ctx, schedule_config=config, instruction="test")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    async def test_schedule_agent_absolute_time(self, mock_schedule_job):
        ctx = _make_ctx()
        ctx.deps.nb_runtime.session = MagicMock()
        ctx.deps.nb_runtime.session.dump.return_value = {"platform": "test", "id": "s1"}
        config = AbsoluteTime(date="2027-01-01", time="09:00")
        result = await _schedule_agent(ctx, schedule_config=config, instruction="新年快乐提醒")
        assert "已设置" in result
        call_kwargs = mock_schedule_job.call_args.kwargs
        assert "定时任务" in call_kwargs["description"]

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    async def test_schedule_agent_passes_ids(self, mock_schedule_job):
        """验证 InvokeAgentParams 包含正确的 user_id / group_id / agent_id"""
        ctx = _make_ctx(user_id="u_custom", group_id="g_custom")
        ctx.deps.nb_runtime.session = MagicMock()
        ctx.deps.nb_runtime.session.dump.return_value = {}
        config = RelativeTime(minutes=1)
        await _schedule_agent(ctx, schedule_config=config, instruction="ping")
        call_kwargs = mock_schedule_job.call_args.kwargs
        assert call_kwargs["user_id"] == "u_custom"
        assert call_kwargs["group_id"] == "g_custom"
        assert call_kwargs["agent_id"] == "a1"


# ===================== schedule_message meme 路径测试 =====================


class TestScheduleMessageMeme:
    """schedule_message 的 meme 处理路径"""

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.UniMessage")
    async def test_with_mentions(self, MockUniMsg, mock_schedule_job):
        """mentions 列表非空时调用 resolve_aliases + at()"""
        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(side_effect=lambda x: f"resolved_{x}")

        mock_msg = _make_fluent_unimsg()
        MockUniMsg.return_value = mock_msg

        config = RelativeTime(minutes=5)
        result = await _schedule(ctx, message="hi", schedule_config=config, mentions=["alice"])
        assert "已设置" in result
        ctx.deps.context.resolve_aliases.assert_called_once_with("alice")
        mock_msg.at.assert_called_once_with("resolved_alice")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.UniMessage")
    async def test_with_meme_from_context_local(self, MockUniMsg, mock_schedule_job):
        """meme 从 context 获取且有 local_path"""
        from nonebot_plugin_wtfllm.memory import ImageSegment

        ctx = _make_ctx()
        mock_seg = MagicMock(spec=ImageSegment)
        mock_seg.available = True
        mock_seg.local_path = "/tmp/meme.webp"
        mock_seg.url = None
        mock_seg.get_bytes_async = AsyncMock(return_value=b"meme-bytes")
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)

        mock_msg = _make_fluent_unimsg()
        MockUniMsg.return_value = mock_msg

        config = RelativeTime(minutes=5)
        result = await _schedule(ctx, message="look", schedule_config=config, meme="IMG:1")
        assert "已设置" in result
        mock_msg.image.assert_called_once_with(raw=b"meme-bytes")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.get_http_client")
    @patch(f"{MODULE}.UniMessage")
    async def test_with_meme_from_context_url(self, MockUniMsg, mock_http, mock_schedule_job):
        """meme 从 context 获取且只有 url"""
        from nonebot_plugin_wtfllm.memory import ImageSegment

        ctx = _make_ctx()
        mock_seg = MagicMock(spec=ImageSegment)
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = "http://example.com/meme.jpg"
        mock_seg.get_bytes_async = AsyncMock(return_value=b"url-bytes")
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)

        mock_msg = _make_fluent_unimsg()
        MockUniMsg.return_value = mock_msg

        config = RelativeTime(minutes=5)
        result = await _schedule(ctx, message="look", schedule_config=config, meme="IMG:2")
        assert "已设置" in result
        mock_msg.image.assert_called_once_with(raw=b"url-bytes")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.meme_repo")
    @patch(f"{MODULE}.UniMessage")
    async def test_with_meme_from_repo(self, MockUniMsg, mock_meme_repo, mock_schedule_job):
        """resolve_media_ref 失败后从 meme_repo 获取"""
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("no ref"))

        mock_payload = AsyncMock()
        mock_payload.storage_id = "meme-uuid"
        mock_payload.get_bytes_async = AsyncMock(return_value=b"repo-bytes")
        mock_meme_repo.get_meme_by_id = AsyncMock(return_value=mock_payload)

        mock_msg = _make_fluent_unimsg()
        MockUniMsg.return_value = mock_msg

        config = RelativeTime(minutes=5)
        result = await _schedule(ctx, message="look", schedule_config=config, meme="some-uuid")
        assert "已设置" in result
        mock_msg.image.assert_called_once_with(raw=b"repo-bytes", name="meme-uuid.webp")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.meme_repo")
    @patch(f"{MODULE}.UniMessage")
    async def test_with_meme_not_found(self, MockUniMsg, mock_meme_repo, mock_schedule_job):
        """meme_repo 返回 None"""
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("no ref"))
        mock_meme_repo.get_meme_by_id = AsyncMock(return_value=None)

        mock_msg = _make_fluent_unimsg()
        MockUniMsg.return_value = mock_msg

        config = RelativeTime(minutes=5)
        result = await _schedule(ctx, message="look", schedule_config=config, meme="bad")
        assert "已设置" in result
        mock_msg.text.assert_any_call("\n哎呀图丢了")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.schedule_job", new_callable=AsyncMock)
    @patch(f"{MODULE}.meme_repo")
    @patch(f"{MODULE}.UniMessage")
    async def test_with_meme_repo_error(self, MockUniMsg, mock_meme_repo, mock_schedule_job):
        """meme_repo 抛异常"""
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("no ref"))
        mock_meme_repo.get_meme_by_id = AsyncMock(side_effect=OSError("disk"))

        mock_msg = _make_fluent_unimsg()
        MockUniMsg.return_value = mock_msg

        config = RelativeTime(minutes=5)
        result = await _schedule(ctx, message="look", schedule_config=config, meme="err")
        assert "已设置" in result
        mock_msg.text.assert_any_call("\n哎呀图丢了")
