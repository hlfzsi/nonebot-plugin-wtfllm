import time
from typing import Optional, Dict

from sqlmodel import SQLModel, Field, Column, JSON


class UserPersona(SQLModel, table=True):
    """用户模型"""

    user_id: str = Field(..., primary_key=True, description="用户的唯一标识符(user_id)")

    interaction_style: str | None = Field(
        default=None, description="LLM总结的用户的交互风格"
    )

    structured_preferences: Optional[Dict] = Field(
        default={},
        sa_column=Column(JSON),
        description="结构化的偏好，如喜欢的技术栈、饮食禁忌、性癖等",
    )

    impression: str | None = Field(default=None, description="LLM对用户的整体印象")

    note: str | None = Field(default=None, description="备注")

    other: str | None = Field(default=None, description="其他补充信息")

    updated_at: int | None = Field(
        default_factory=lambda: int(time.time()),
        description="用户档案更新的时间戳 (Unix 时间戳)",
    )

    version: int = Field(default=1, description="用户档案的版本号 , 每次更新应当加1")

    def render_to_llm(self, ref: str) -> Optional[str]:
        """将用户画像渲染为 LLM 可理解的文本格式

        Returns:
            用户画像的 Markdown 格式文本
        """
        parts = [f"用户: {ref}"]
        build_flag = False
        if self.interaction_style:
            parts.append(f"交互风格: {self.interaction_style}")
            build_flag = True
        if self.structured_preferences:
            prefs = ", ".join(
                f"{key}: {value}" for key, value in self.structured_preferences.items()
            )
            parts.append(f"结构化偏好: {prefs}")
            build_flag = True
        if self.impression:
            parts.append(f"整体印象: {self.impression}")
            build_flag = True
        if self.other:
            parts.append(f"其他信息: {self.other}")
            build_flag = True
        if self.note:
            parts.append(f"备注: {self.note}")
            build_flag = True

        return "\n".join(parts) if build_flag else None
