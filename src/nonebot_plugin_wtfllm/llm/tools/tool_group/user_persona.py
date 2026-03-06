from typing import Dict

from .base import ToolGroupMeta
from ...deps import Context
from ....utils import logger
from ....db import user_persona_repo

user_tool_group = ToolGroupMeta(
    name="UserPersona",
    description="用户信息与备注工具组，可以记录用户偏好和行为模式供后续交互参考。",
)


@user_tool_group.tool(cost=0)
async def get_user_persona(
    ctx: Context,
    user_id: str,
) -> str:
    """获取指定用户的完整个性化画像文本。

    Args:
        user_id (str): 用户ID

    Returns:
        包含交互风格、结构化偏好、整体印象及其他信息的Markdown格式文本。
    """
    persona_text = await user_persona_repo.get_persona_text(user_id, ctx.deps.context)

    logger.debug(f"Retrieved user persona for in group {ctx.deps.ids.group_id}.")

    return persona_text if persona_text else "暂无该用户的画像"


@user_tool_group.tool(cost=0)
async def update_user_persona(
    ctx: Context,
    user_id: str,
    interaction_style: str | None = None,
    note: str | None = None,
    structured_preferences: Dict[str, str] | None = None,
    impression: str | None = None,
    other: str | None = None,
) -> str:
    """更新用户画像与备注等个性化信息

    Args:
        user_id (str): 用户ID
        interaction_style (str | None): 用户的交互风格描述
        note (str | None): 备注
        structured_preferences (Dict[str, str] | None): 结构化的偏好，如喜欢的技术栈、饮食禁忌、性癖等
        impression (str | None): 你对用户的整体印象
        other (str | None): 其他补充信息

    Returns:
        更新后的用户画像的文本表示
    """
    real_id = ctx.deps.context.resolve_aliases(user_id) or user_id
    user_persona = await user_persona_repo.update_persona(
        user_id=real_id,
        interaction_style=interaction_style,
        structured_preferences=structured_preferences,
        impression=impression,
        other=other,
        note=note,
    )

    logger.debug(
        f"Updated user persona for user {real_id} in group {ctx.deps.ids.group_id}."
    )
    persona = user_persona.render_to_llm(user_id)

    return persona or "暂无该用户的画像"
