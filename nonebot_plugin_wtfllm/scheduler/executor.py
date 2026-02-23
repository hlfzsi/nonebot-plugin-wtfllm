"""定时消息执行回调"""

__all__ = ["get_handle_func_by_type", "execute_scheduled_static_message"]
from typing import Callable, Awaitable

from nonebot_plugin_alconna import Target, UniMessage

# from ..llm import CHAT_AGENT, AgentDeps
from ..db import scheduled_message_repo, ScheduledMessage
from ..db.models.scheduled_message import ScheduledFunctionType
from ..stream_processing import convert_and_store_item
from ..msg_tracker import msg_tracker
from ..utils import logger, ensure_msgid_from_receipt
from ..services.func.easy_ban import is_banned


def get_handle_func_by_type(
    func_type: ScheduledFunctionType,
) -> Callable[[str], Awaitable[None]] | None:
    if func_type == ScheduledFunctionType.STATIC_MESSAGE:
        return execute_scheduled_static_message
    elif func_type == ScheduledFunctionType.DYNAMIC_MESSAGE:
        return execute_scheduled_dynamic_message
    else:
        return None


async def should_skip_scheduled_message(record: ScheduledMessage) -> bool:
    if record.status != "pending":
        logger.debug(
            f"Skipped non-pending job_id={record.job_id} (status={record.status})"
        )
        return True

    if await is_banned(record.user_id, record.group_id):
        logger.info(
            f"User {record.user_id} is banned in group {record.group_id}, skipping job_id={record.job_id}"
        )
        await scheduled_message_repo.mark_failed(record.job_id, "UserOrGroup is banned")
        return True

    return False


async def execute_scheduled_static_message(job_id: str) -> None:
    """APScheduler 作业回调：发送定时消息并更新状态

    Args:
        job_id: 对应的数据库记录 job_id
    """
    record = await scheduled_message_repo.get_by_job_id(job_id)
    if not record:
        logger.warning(f"Scheduled message not found for job_id={job_id}")
        return

    if await should_skip_scheduled_message(record):
        return

    try:
        target: Target = Target.load(record.target_data)
        unimsg: UniMessage = UniMessage.load(record.messages)
        receipt = await unimsg.send(target=target)
        sent_msg_id = ensure_msgid_from_receipt(receipt)

        if record.agent_id and record.user_id:
            await convert_and_store_item(
                agent_id=record.agent_id,
                uni_msg=unimsg,
                group_id=record.group_id,
                user_id=record.user_id,
                sender=record.agent_id,
                msg_id=sent_msg_id,
            )
            msg_tracker.track(
                agent_id=record.agent_id,
                user_id=record.user_id,
                group_id=record.group_id,
                msg_id=sent_msg_id,
            )
        else:
            logger.warning(
                f"Missing agent_id or user_id for job_id={job_id}, skipping message tracking"
            )
        await scheduled_message_repo.mark_completed(job_id)
        logger.info(f"Scheduled message sent: job_id={job_id}")
    except (RuntimeError, ValueError, OSError) as e:
        await scheduled_message_repo.mark_failed(job_id, str(e))
        logger.error(f"Scheduled message failed: job_id={job_id}, error={e}")


async def execute_scheduled_dynamic_message(job_id: str) -> None:
    """APScheduler 作业回调：发送定时动态生成消息并更新状态

    Args:
        job_id: 对应的数据库记录 job_id
    """
    raise NotImplementedError("Dynamic message execution logic is not implemented yet.")
    # TODO 定时动态消息执行逻辑待实现，核心是根据记录中的信息构造 AgentDeps 并调用 CHAT_AGENT.run(deps)
