from datetime import datetime, timedelta
from typing import List, Literal, Union

from pydantic import BaseModel, Field
from nonebot_plugin_alconna import UniMessage

from .base import ToolGroupMeta
from .utils import reschedule_deadline
from ...deps import Context
from ....scheduler import (
    schedule_message as _schedule_message,
    cancel_message as _cancel_message,
)
from ....db import scheduled_message_repo
from ....memory import ImageSegment
from ....v_db import meme_repo
from ....utils import get_http_client

schedule_message_group = ToolGroupMeta(
    name="ScheduleMessage",
    description="定时消息工具组，包含用于调度和管理定时消息的工具",
)


# TODO 允许不预先生成message，而是在触发时动态生成
class RelativeTime(BaseModel):
    type: Literal["relative"] = "relative"
    minutes: int | None = Field(None, description="多少分钟后触发")
    hours: int | None = Field(None, description="多少小时后触发")
    days: int | None = Field(None, description="多少天后触发")

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


# @schedule_message_group.tool
# async def call_later(ctx: Context, schedule_config: TimeConfig) -> str: ...


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

    await _schedule_message(
        target=ctx.deps.nb_runtime.target,
        session=ctx.deps.nb_runtime.session,
        unimsg=unimsg,
        trigger_time=schedule_config.trigger_timestamp,
    )
    return "定时消息已设置！"


@schedule_message_group.tool(cost=0)
async def cancel_scheduled_message(ctx: Context, job_id: str) -> str:
    """取消已设置的定时消息。

    只能取消属于当前用户且状态为pending的定时消息。

    Args:
        job_id: 要取消的定时消息的job ID

    Returns:
        str: 操作结果描述，包含成功或失败的原因
    """
    reschedule_deadline(ctx, 15)
    job = await scheduled_message_repo.get_by_job_id(job_id)
    if not job:
        return f"未找到 job_id={job_id} 的定时消息记录。"

    if job.user_id != ctx.deps.ids.user_id:
        return f"你没有权限取消 job_id={job_id} 的定时消息, 因为该job不属于当前用户。"

    if job.status != "pending":
        return f"job_id={job_id} 的定时消息当前状态为 {job.status}，无法取消。"

    await _cancel_message(job_id)
    return f"定时消息 job_id={job_id} 已取消。"


@schedule_message_group.tool(cost=0)
async def list_scheduled_messages(
    ctx: Context, type: Literal["user", "group"], limit: int = 10
) -> str:
    """列出定时消息。

    查询当前用户或当前群组的定时消息列表。

    Args:
        type: 查询类型，"user"表示当前用户，"group"表示当前群组
        limit: 返回结果的最大数量，默认为10

    Returns:
        str: 定时消息列表文本
    """
    reschedule_deadline(ctx, 15)
    if type == "user":
        if not ctx.deps.ids.user_id:
            raise ValueError(
                "User ID is required in Context.deps.ids for listing user scheduled messages."
            )
        schedule_messages = await scheduled_message_repo.list_by_user(
            user_id=ctx.deps.ids.user_id, agent_id=ctx.deps.ids.agent_id, limit=limit
        )
    elif type == "group":
        if not ctx.deps.ids.group_id:
            raise ValueError(
                "Group ID is required in Context.deps.ids for listing group scheduled messages."
            )
        schedule_messages = await scheduled_message_repo.list_by_group(
            group_id=ctx.deps.ids.group_id, agent_id=ctx.deps.ids.agent_id, limit=limit
        )
    else:
        raise ValueError("Invalid type. Must be 'user' or 'group'.")

    if not schedule_messages:
        return "当前用户没有任何定时消息。"

    return "\n\n".join([(msg.to_text(ctx.deps.context)) for msg in schedule_messages])
