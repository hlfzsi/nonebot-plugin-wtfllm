import time
from datetime import datetime, timedelta
from typing import List, Literal, Union

from pydantic import BaseModel, Field
from nonebot_plugin_alconna import UniMessage

from .base import ToolGroupMeta
from .utils import reschedule_deadline
from ...deps import Context
from ....scheduler import schedule_job, cancel_job
from ....scheduler.triggers import DateTriggerConfig
from ....scheduler.tasks.invoke_agent import InvokeAgentParams
from ....scheduler.tasks.send_static_message import SendStaticMessageParams
from ....db import scheduled_job_repo
from ....db.models.scheduled_job import ScheduledJobStatus, ScheduledJob
from ....memory import ImageSegment, MemoryContextBuilder
from ....v_db import meme_repo
from ....utils import get_http_client, SCHEDULED_CACHE_DIR

schedule_message_group = ToolGroupMeta(
    name="ScheduleMessage",
    description="定时消息工具组，包含用于调度和管理定时消息的工具",
)


# TODO 允许不预先生成message，而是在触发时动态生成
class RelativeTime(BaseModel):
    type: Literal["relative"] = "relative"
    minutes: int | None = Field(default=None, description="多少分钟后触发")
    hours: int | None = Field(default=None, description="多少小时后触发")
    days: int | None = Field(default=None, description="多少天后触发")

    @property
    def trigger_timestamp(self) -> int:
        now = datetime.now()
        delta = timedelta(
            minutes=self.minutes or 0,
            hours=self.hours or 0,
            days=self.days or 0,
        )
        trigger_time = now + delta
        return int(trigger_time.timestamp())


class AbsoluteTime(BaseModel):
    type: Literal["absolute"] = "absolute"
    date: str = Field(..., description="日期，格式为 YYYY-MM-DD")
    time: str = Field(..., description="时间，格式为 HH:MM")

    @property
    def trigger_timestamp(self) -> int:
        dt_str = f"{self.date} {self.time}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        return int(dt.timestamp())


TimeConfig = Union[RelativeTime, AbsoluteTime]


def _job_to_text(job: ScheduledJob, ctx: MemoryContextBuilder) -> str:
    """将 ScheduledJob 格式化为人类可读文本。"""
    _user = (
        ctx.ctx.alias_provider.get_alias(job.user_id)
        if ctx and job.user_id
        else job.user_id
    )
    _group = (
        ctx.ctx.alias_provider.get_alias(job.group_id)
        if ctx and job.group_id
        else job.group_id
    )

    target_desc = f"用户 {_user}" if _user else ""
    if _group:
        target_desc += f" 在群 {_group}"

    trigger_cfg = job.trigger_config
    trigger_type = trigger_cfg.get("type", "")
    if trigger_type == "date":
        run_ts = trigger_cfg.get("run_timestamp", 0)
        trigger_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_ts))
        trigger_desc = f"触发时间: {trigger_time_str}"
    else:
        trigger_desc = f"触发类型: {trigger_type}"

    desc = job.description or job.task_name
    return (
        f"定时任务 (job_id={job.job_id}) {trigger_desc} "
        f"状态: {job.status} {target_desc} 描述: {desc}"
    )


@schedule_message_group.tool(cost=0)
async def schedule_message(
    ctx: Context,
    message: str,
    schedule_config: TimeConfig,
    mentions: List[str] | None = None,
    meme: str | None = None,
) -> str | None:
    """创建定时消息。

    创建一条在指定时间发送的消息，支持文本、@提及和表情包。

    Args:
        message: 要发送的文本消息内容, 纯文本
        schedule_config: 时间配置，支持相对时间或绝对时间
        mentions: 要@的用户列表
        meme: 表情包UUID或图片引用

    Returns:
        str: 成功时返回提示信息
    """
    if ctx.deps.nb_runtime is None:
        raise ValueError(
            "nb_runtime is required in Context.deps for scheduling messages."
        )
    if not ctx.deps.ids.user_id:
        raise ValueError(
            "User ID is required in Context.deps.ids for scheduling messages."
        )

    reschedule_deadline(ctx, 15)

    unimsg = UniMessage()

    if mentions:
        for mention in mentions:
            mention = ctx.deps.context.resolve_aliases(mention)
            if mention:
                unimsg.at(mention)

    unimsg.text(message)

    if meme:
        try:
            _meme = ctx.deps.context.resolve_media_ref(meme, ImageSegment)
        except ValueError:
            _meme = None

        if _meme and _meme.available:
            if _meme.local_path:
                image_data = await _meme.get_bytes_async()
                unimsg.image(raw=image_data)
            elif _meme.url:
                image_data = await _meme.get_bytes_async(
                    get_http_client(), download=True
                )
                unimsg.image(raw=image_data)

        else:
            try:
                _meme = await meme_repo.get_meme_by_id(meme)
                if _meme:
                    _bytes = await _meme.get_bytes_async()
                    unimsg.image(raw=_bytes, name=f"{_meme.storage_id}.webp")
                else:
                    unimsg.text("\n哎呀图丢了")
            except (OSError, ValueError, RuntimeError):
                unimsg.text("\n哎呀图丢了")

    trigger_ts = schedule_config.trigger_timestamp
    params = SendStaticMessageParams(
        target_data=ctx.deps.nb_runtime.target.dump(),
        messages=unimsg.dump(media_save_dir=SCHEDULED_CACHE_DIR),
        user_id=ctx.deps.ids.user_id,
        group_id=ctx.deps.ids.group_id,
        agent_id=ctx.deps.ids.agent_id,
    )
    trigger = DateTriggerConfig(run_timestamp=trigger_ts)

    trigger_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(trigger_ts))
    await schedule_job(
        task_name="send_static_message",
        task_params=params,
        trigger=trigger,
        user_id=ctx.deps.ids.user_id,
        group_id=ctx.deps.ids.group_id,
        agent_id=ctx.deps.ids.agent_id,
        description=f"定时消息 [{trigger_time_str}]: {message[:50]}",
    )
    return "定时消息已设置！"


@schedule_message_group.tool(cost=3)
async def schedule_agent_task(
    ctx: Context, schedule_config: TimeConfig, instruction: str
) -> str:
    """在指定时间触发执行由你介入的特定任务。仅必要时使用。

    设置一个延时任务，在到达指定时间后，系统会以 `instruction` 作为输入
    再次唤起你。这适用于提醒、定时检查或需要在未来某个时间点执行的复杂操作。

    Args:
        schedule_config (TimeConfig): 时间配置，支持相对时间（如“5分钟后”）或绝对时间（如“2023-10-01 10:00”）。
        instruction (str): 到达触发时间时发送给智能体的指令或提醒内容。
    """
    if ctx.deps.nb_runtime is None:
        raise ValueError(
            "nb_runtime is required in Context.deps for scheduling messages."
        )
    if not ctx.deps.ids.user_id:
        raise ValueError(
            "User ID is required in Context.deps.ids for scheduling messages."
        )

    reschedule_deadline(ctx, 15)
    trigger_ts = schedule_config.trigger_timestamp
    trigger = DateTriggerConfig(run_timestamp=trigger_ts)
    params = InvokeAgentParams(
        user_id=ctx.deps.ids.user_id,
        group_id=ctx.deps.ids.group_id,
        agent_id=ctx.deps.ids.agent_id,
        prompt=instruction,
        target_data=ctx.deps.nb_runtime.target.dump(),
        session_data=ctx.deps.nb_runtime.session.dump(),
    )
    await schedule_job(
        task_name="invoke_agent",
        task_params=params,
        trigger=trigger,
        user_id=ctx.deps.ids.user_id,
        group_id=ctx.deps.ids.group_id,
        agent_id=ctx.deps.ids.agent_id,
        description=f"定时任务: {instruction[:50]}",
    )
    return "定时任务已设置！"


@schedule_message_group.tool(cost=0)
async def cancel_scheduled_task(ctx: Context, job_id: str) -> str:
    """取消已设置的定时任务。

    只能取消属于当前用户且状态为pending的定时任务。

    Args:
        job_id: 要取消的定时消息的job ID

    Returns:
        str: 操作结果描述，包含成功或失败的原因
    """
    reschedule_deadline(ctx, 15)
    job = await scheduled_job_repo.get_by_job_id(job_id)
    if not job:
        return f"未找到 job_id={job_id} 的定时消息记录。"

    if job.user_id != ctx.deps.ids.user_id:
        return f"你没有权限取消 job_id={job_id} 的定时消息, 因为该job不属于当前用户。"

    if job.status != ScheduledJobStatus.PENDING:
        return f"job_id={job_id} 的定时消息当前状态为 {job.status}，无法取消。"

    await cancel_job(job_id)
    return f"定时消息 job_id={job_id} 已取消。"


@schedule_message_group.tool(cost=0)
async def list_scheduled_tasks(
    ctx: Context, type: Literal["user", "group"], limit: int = 5
) -> str:
    """列出定时消息。

    查询当前用户或当前群组的定时任务列表。

    Args:
        type: 查询类型，"user"表示当前用户，"group"表示当前群组
        limit: 返回结果的最大数量，默认为5

    Returns:
        str: 定时任务列表文本
    """
    reschedule_deadline(ctx, 15)
    if type == "user":
        if not ctx.deps.ids.user_id:
            raise ValueError(
                "User ID is required in Context.deps.ids for listing user scheduled tasks."
            )
        jobs = await scheduled_job_repo.list_by_user(
            user_id=ctx.deps.ids.user_id, agent_id=ctx.deps.ids.agent_id, limit=limit
        )
    elif type == "group":
        if not ctx.deps.ids.group_id:
            raise ValueError(
                "Group ID is required in Context.deps.ids for listing group scheduled tasks."
            )
        jobs = await scheduled_job_repo.list_by_group(
            group_id=ctx.deps.ids.group_id, agent_id=ctx.deps.ids.agent_id, limit=limit
        )
    else:
        raise ValueError("Invalid type. Must be 'user' or 'group'.")

    if not jobs:
        return "当前用户没有任何定时任务。"

    return "\n\n".join([_job_to_text(job, ctx.deps.context) for job in jobs])
