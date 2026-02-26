__all__ = ["SendStaticMessageParams", "handle_send_static_message"]

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..registry import scheduled_task


class SendStaticMessageParams(BaseModel):
    """send_static_message 任务的参数模型"""

    target_data: Dict[str, Any] = Field(..., description="Target.dump() 输出")
    messages: List[Dict[str, Any]] = Field(..., description="UniMessage.dump() 输出")
    user_id: str = Field(..., description="创建任务的用户ID")
    group_id: Optional[str] = Field(default=None, description="关联群组ID")
    agent_id: str = Field(..., description="Bot/Agent ID")


@scheduled_task("send_static_message", SendStaticMessageParams)
async def handle_send_static_message(params: SendStaticMessageParams) -> None:
    """发送预构建的 UniMessage 到指定目标。"""
    # --- 延迟导入：避免 scheduler ↔ 业务层循环依赖 ---
    from nonebot_plugin_alconna import Target, UniMessage

    from ...msg_tracker import msg_tracker
    from ...services.func.easy_ban import is_banned
    from ...stream_processing import convert_and_store_item
    from ...utils import logger, ensure_msgid_from_receipt

    if await is_banned(params.user_id, params.group_id):
        raise RuntimeError(
            f"User {params.user_id} is banned in group {params.group_id}"
        )

    target: Target = Target.load(params.target_data)
    unimsg: UniMessage = UniMessage.load(params.messages)
    receipt = await unimsg.send(target=target)
    sent_msg_id = ensure_msgid_from_receipt(receipt)

    if params.agent_id and params.user_id:
        await convert_and_store_item(
            agent_id=params.agent_id,
            uni_msg=unimsg,
            group_id=params.group_id,
            user_id=params.user_id,
            sender=params.agent_id,
            msg_id=sent_msg_id,
        )
        msg_tracker.track(
            agent_id=params.agent_id,
            user_id=params.user_id,
            group_id=params.group_id,
            msg_id=sent_msg_id,
        )
    else:
        logger.warning("Missing agent_id or user_id, skipping message tracking")
