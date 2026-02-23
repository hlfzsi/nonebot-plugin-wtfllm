"""用户 Repository

封装用户个性化的数据访问操作。
"""

__all__ = ["UserPersonaRepository"]
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

from .base import BaseRepository
from ..models import UserPersona
from ..engine import SESSION_MAKER, WRITE_LOCK

if TYPE_CHECKING:
    from ...memory import MemoryContextBuilder


class UserPersonaRepository(BaseRepository[UserPersona]):
    """用户画像数据访问层"""

    def __init__(self):
        super().__init__(UserPersona)

    async def get_or_create(self, user_id: str) -> UserPersona:
        """获取或创建用户画像

        Args:
            user_id: 用户ID

        Returns:
            用户画像对象
        """
        profile = await self.get_by_id(user_id)
        if not profile:
            profile = UserPersona(user_id=user_id, version=0)
            await self.save(profile)
        return profile

    async def update_persona(
        self,
        user_id: str,
        interaction_style: Optional[str] = None,
        note: Optional[str] = None,
        structured_preferences: Optional[Dict[str, Any]] = None,
        impression: Optional[str] = None,
        other: Optional[str] = None,
    ) -> UserPersona:
        """更新用户画像

        Args:
            user_id: 用户ID
            interaction_style: 交互风格
            structured_preferences: 结构化偏好
            impression: LLM 情绪态度
            other: 其他补充信息
            note: 备注
        Returns:
            更新后的用户画像对象
        """
        async with WRITE_LOCK:
            async with SESSION_MAKER() as session:
                user_profile = await session.get(UserPersona, user_id)

                if not user_profile:
                    user_profile = UserPersona(
                        user_id=user_id,
                        version=0,
                    )
                    session.add(user_profile)

                if interaction_style is not None:
                    user_profile.interaction_style = interaction_style
                if structured_preferences is not None:
                    user_profile.structured_preferences = structured_preferences
                if note is not None:
                    user_profile.note = note
                if impression is not None:
                    user_profile.impression = impression
                if other is not None:
                    user_profile.other = other

                # 更新时间戳和版本号
                user_profile.updated_at = int(time.time())
                user_profile.version += 1

                await session.flush()
                await session.refresh(user_profile)
                await session.commit()

        return user_profile

    async def get_persona_text(
        self,
        user_id: Optional[str] = None,
        ctx: Optional["MemoryContextBuilder"] = None,
        real_user_id: Optional[str] = None,
    ) -> str | None:
        """获取用户画像的文本表示

        Args:
            user_id: ref用户ID
            ctx: 内存上下文构建器
            real_user_id: 实际用户ID

        Returns:
            用户画像的 Markdown 格式文本
        """
        if real_user_id and user_id:
            query_user_id = real_user_id
        elif user_id and ctx:
            query_user_id = ctx.ctx.alias_provider.resolve_alias(user_id) or user_id
        else:
            raise ValueError(
                "Must provide both user_id and ctx, or both user_id and real_user_id."
            )

        async with SESSION_MAKER() as session:
            async with session.begin():
                user_profile = await session.get(UserPersona, query_user_id)
        return user_profile.render_to_llm(user_id) if user_profile else None
