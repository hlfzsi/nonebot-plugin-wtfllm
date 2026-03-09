__all__ = ["NoteMemoryTable"]

from sqlmodel import SQLModel, Field


class NoteMemoryTable(SQLModel, table=True):
    """会话级短期备忘记录。"""

    storage_id: str = Field(primary_key=True, description="Note 唯一ID")
    content: str = Field(..., description="备忘内容")
    group_id: str | None = Field(default=None, index=True, description="所属群组ID")
    user_id: str | None = Field(default=None, index=True, description="所属用户ID")
    agent_id: str = Field(..., index=True, description="所属智能体ID")
    expires_at: int = Field(..., index=True, description="过期时间戳")
    created_at: int = Field(..., index=True, description="创建时间戳")
    updated_at: int = Field(..., index=True, description="最后更新时间戳")
    token_count: int = Field(default=0, description="缓存的 token 计数")
