"""向量数据模型

导出所有 Qdrant Payload 模型。
"""

__all__ = [
    "MODELS",
    "VectorModel",
    "MemePayload",
    "CoreMemoryPayload",
    "KnowledgeBasePayload",
]


from typing import Type
from .base import VectorModel
from .meme import MemePayload
from .core_memory import CoreMemoryPayload
from .knowledge_base import KnowledgeBasePayload


MODELS: list[Type[VectorModel]] = [MemePayload, CoreMemoryPayload, KnowledgeBasePayload]
