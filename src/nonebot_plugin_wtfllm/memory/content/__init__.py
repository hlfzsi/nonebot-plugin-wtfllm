"""内容模块"""

from .message import Message
from .segments import (
    MEDIA_TYPES,
    AudioSegment,
    BaseSegment,
    FileSegment,
    ImageSegment,
    MentionSegment,
    Node,
    Segment,
    TextSegment,
    VideoSegment,
    ForwardSegment,
    UnknownSegment,
    HyperSegment,
    MediaBaseSegment,
    EmojiSegment,
)

# 解决循环前向引用
Node.model_rebuild(_types_namespace={"Message": Message})
ForwardSegment.model_rebuild(_types_namespace={"Message": Message})

__all__ = [
    "MEDIA_TYPES",
    "AudioSegment",
    "BaseSegment",
    "FileSegment",
    "ImageSegment",
    "MentionSegment",
    "Message",
    "Node",
    "Segment",
    "TextSegment",
    "VideoSegment",
    "UnknownSegment",
    "ForwardSegment",
    "HyperSegment",
    "MediaBaseSegment",
    "EmojiSegment",
]
