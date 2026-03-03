"""向量数据库模块 - Qdrant 数据访问层

提供 Qdrant 客户端、模型、Repository 和生命周期管理的统一入口。

Example:
    ```python
    from v_db import core_memory_repo

    # 保存核心记忆
    await core_memory_repo.save_core_memory(memory)

    # 语义搜索
    results = await core_memory_repo.search_cross_session(
        agent_id="agent_main",
        query="Python 编程",
        limit=5,
    )
    for result in results:
        print(f"Score: {result.score}, Content: {result.item.content}")
    ```
"""

__all__ = [
    # 引擎
    "get_qdrant_client",
    # 生命周期
    "on_startup",
    "on_shutdown",
    # 模型基类
    "VectorModel",
    "MemePayload",
    "CoreMemoryPayload",
    "KnowledgeBasePayload",
    "TopicArchivePayload",
    # Repository 基类
    "VectorRepository",
    "SearchResult",
    # Repository 实现
    "MemeRepository",
    "CoreMemoryRepository",
    "KnowledgeBaseRepository",
    "TopicArchiveRepository",
    # Repository 单例
    "meme_repo",
    "core_memory_repo",
    "knowledge_base_repo",
    "topic_archive_repo",
]

from .engine import get_qdrant_client
from .lifecycle import on_startup, on_shutdown
from .models import (
    VectorModel,
    MemePayload,
    CoreMemoryPayload,
    KnowledgeBasePayload,
    TopicArchivePayload,
)
from .repositories import (
    VectorRepository,
    SearchResult,
    MemeRepository,
    CoreMemoryRepository,
    KnowledgeBaseRepository,
    TopicArchiveRepository,
)

# 全局 Repository 单例
meme_repo = MemeRepository()
core_memory_repo = CoreMemoryRepository()
knowledge_base_repo = KnowledgeBaseRepository()
topic_archive_repo = TopicArchiveRepository()
