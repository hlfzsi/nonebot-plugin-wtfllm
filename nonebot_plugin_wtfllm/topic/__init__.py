"""话题聚类短期记忆增强模块"""

from ..config import APP_CONFIG
from ._types import SessionKey
from .manager import TopicManager

__all__ = [
    "topic_manager",
    "TopicManager",
    "SessionKey",
]

topic_manager = TopicManager(
    cluster_threshold=APP_CONFIG.topic_cluster_threshold,
    max_clusters=APP_CONFIG.topic_max_clusters,
    decay_seconds=APP_CONFIG.topic_decay_minutes * 60,
)
