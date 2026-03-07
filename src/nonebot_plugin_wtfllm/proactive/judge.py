from .topic_interest import has_active_topic_interest_match


async def should_proactively_respond(
    *,
    agent_id: str,
    user_id: str,
    group_id: str | None,
    plain_text: str,
) -> bool:
    """统一的主动发言判断入口。"""
    return await has_active_topic_interest_match(
        agent_id=agent_id,
        user_id=user_id,
        group_id=group_id,
        plain_text=plain_text,
    )