import uuid
from time import time
from typing import Dict, Callable, List, cast

from nonebot_plugin_alconna import UniMessage, MsgId
from nonebot_plugin_alconna.uniseg.segment import (
    Segment,
    Reply,
    Text,
    Image,
    At,
    AtAll,
    Video,
    Voice,
    File,
    Reference,
    RefNode,
    CustomNode,
    Hyper,
    Emoji,
)

from ..utils import logger, APP_CONFIG, MEDIA_DIR
from ..db import memory_item_repo
from .hyper_clean import clean_hyper_content
from ..memory import (
    GroupMemoryItem,
    PrivateMemoryItem,
    BaseSegment,
    TextSegment,
    ImageSegment,
    FileSegment,
    AudioSegment,
    MentionSegment,
    VideoSegment,
    ForwardSegment,
    UnknownSegment,
    HyperSegment,
    EmojiSegment,
    Node,
    Message,
    MemoryItemUnion,
)


def _to_text_segment(seg: Text, created_at: int | None) -> TextSegment:
    created_at = created_at or int(time())
    return TextSegment(content=seg.text, created_at=created_at)


def _to_emoji_segment(seg: Emoji, created_at: int | None) -> EmojiSegment:
    created_at = created_at or int(time())
    return EmojiSegment(
        name=seg.name if seg.name else seg.id,
        url=seg.url,
        created_at=created_at,
    )


def _to_unknown_segment(seg: Segment, created_at: int | None) -> UnknownSegment:
    created_at = created_at or int(time())
    return UnknownSegment(
        original_type=type(seg).__name__,
        created_at=created_at,
    )


def _to_image_segment(seg: Image, created_at: int | None) -> ImageSegment:
    created_at = created_at or int(time())
    _seg = None
    if seg.url:
        _seg = ImageSegment(url=seg.url, created_at=created_at)
    elif seg.raw_bytes:
        local_path = MEDIA_DIR / f"image_{uuid.uuid4().hex}"
        local_path.write_bytes(seg.raw_bytes)
        _seg = ImageSegment(local_path=local_path, created_at=created_at)

    if (docs := getattr(seg, "desc", None)) and _seg is not None:
        _seg.desc = docs

    if _seg is not None:
        return _seg

    raise ValueError("Image segment must have either url or raw_bytes")


def _to_file_segment(seg: File, created_at: int | None) -> FileSegment:
    created_at = created_at or int(time())
    _seg = None
    if seg.url:
        _seg = FileSegment(url=seg.url, filename=seg.name, created_at=created_at)
    elif seg.raw_bytes:
        local_path = MEDIA_DIR / f"file_{uuid.uuid4().hex}"
        local_path.write_bytes(seg.raw_bytes)
        _seg = FileSegment(
            local_path=local_path, filename=seg.name, created_at=created_at
        )

    if (docs := getattr(seg, "desc", None)) and _seg is not None:
        _seg.desc = docs

    if _seg is not None:
        return _seg

    raise ValueError("File segment must have either url or raw_bytes")


def _to_audio_segment(seg: Voice, created_at: int | None) -> AudioSegment:
    created_at = created_at or int(time())
    _seg = None
    if seg.url:
        _seg = AudioSegment(url=seg.url, created_at=created_at)
    elif seg.raw_bytes:
        local_path = MEDIA_DIR / f"audio_{uuid.uuid4().hex}"
        local_path.write_bytes(seg.raw_bytes)
        _seg = AudioSegment(local_path=local_path, created_at=created_at)

    if (docs := getattr(seg, "desc", None)) and _seg is not None:
        _seg.desc = docs

    if _seg is not None:
        return _seg

    raise ValueError("Voice segment must have either url or raw_bytes")


def _to_mention_segment(seg: At | AtAll, created_at: int | None) -> MentionSegment:
    created_at = created_at or int(time())
    if isinstance(seg, AtAll):
        return MentionSegment(at_all=True, created_at=created_at)
    return MentionSegment(user_id=seg.target, created_at=created_at)


def _to_video_segment(seg: Video, created_at: int | None) -> VideoSegment:
    created_at = created_at or int(time())
    _seg = None
    if seg.url:
        _seg = VideoSegment(url=seg.url, created_at=created_at)
    elif seg.raw_bytes:
        local_path = MEDIA_DIR / f"video_{uuid.uuid4().hex}"
        local_path.write_bytes(seg.raw_bytes)
        _seg = VideoSegment(local_path=local_path, created_at=created_at)

    if (docs := getattr(seg, "desc", None)) and _seg is not None:
        _seg.desc = docs

    if _seg is not None:
        return _seg

    raise ValueError("Video segment must have either url or raw_bytes")


def _to_hyper_segment(seg: Hyper, created_at: int | None) -> HyperSegment:
    created_at = created_at or int(time())

    content = clean_hyper_content(seg.raw, seg.format)

    return HyperSegment(
        format=seg.format,
        content=content,
        created_at=created_at,
    )


def _to_forward_segment(seg: Reference, created_at: int | None) -> ForwardSegment:
    created_at = created_at or int(time())

    if APP_CONFIG.ignore_reference:
        placeholder_node = Node(
            sender="system",
            group_id=None,
            content=Message.create([TextSegment(content="[合并转发消息已被忽略]")]),
        )
        return ForwardSegment(children=[placeholder_node], created_at=created_at)

    nodes: List[Node] = []
    for child in seg.children:
        if isinstance(child, CustomNode):
            content = child.content
            child_created_at = int(child.time.timestamp())
            if isinstance(content, str):
                memory_msg = Message.create(
                    [TextSegment(content=content, created_at=child_created_at)]
                )
            else:
                memory_msg = Message.create()
                for uni_seg in content:
                    method = UNISEG_TO_MEMORYSEG_MAP.get(type(uni_seg))
                    if method is None:
                        method = _to_unknown_segment
                        logger.debug(
                            f"Unsupported segment type in forward node: {type(uni_seg)}"
                        )
                    memory_seg = method(uni_seg, child_created_at)
                    memory_msg += memory_seg  # pyright: ignore[reportOperatorIssue]

            nodes.append(
                Node(
                    sender=child.uid,
                    group_id=child.context,
                    content=memory_msg,
                )
            )
        elif isinstance(child, RefNode):
            nodes.append(
                Node(
                    sender="unknown",
                    group_id=child.context,
                    content=Message.create([TextSegment(content="[未知引用消息]")]),
                )
            )
        else:
            logger.debug(f"Unknown forward node type: {type(child)}")

    return ForwardSegment(children=nodes, created_at=created_at)


UNISEG_TO_MEMORYSEG_MAP: Dict[
    type[Segment], Callable[[Segment, int | None], BaseSegment]
] = {
    Text: cast(Callable[[Segment, int | None], BaseSegment], _to_text_segment),
    Emoji: cast(Callable[[Segment, int | None], BaseSegment], _to_emoji_segment),
    Image: cast(Callable[[Segment, int | None], BaseSegment], _to_image_segment),
    File: cast(Callable[[Segment, int | None], BaseSegment], _to_file_segment),
    Voice: cast(Callable[[Segment, int | None], BaseSegment], _to_audio_segment),
    At: cast(Callable[[Segment, int | None], BaseSegment], _to_mention_segment),
    AtAll: cast(Callable[[Segment, int | None], BaseSegment], _to_mention_segment),
    Video: cast(Callable[[Segment, int | None], BaseSegment], _to_video_segment),
    Hyper: cast(Callable[[Segment, int | None], BaseSegment], _to_hyper_segment),
    Reference: cast(Callable[[Segment, int | None], BaseSegment], _to_forward_segment),
}


def extract_memeorymsg_from_unimsg(
    unimsg: UniMessage[Segment],
) -> Message:
    memory_msg = Message.create()

    for seg in unimsg:
        if isinstance(seg, Reply):
            continue
        method = UNISEG_TO_MEMORYSEG_MAP.get(type(seg))
        if method is None:
            method = _to_unknown_segment
            logger.debug(f"Unsupported segment type for memory extraction: {type(seg)}")
        memory_seg = method(seg, None)
        memory_msg += memory_seg  # pyright: ignore[reportOperatorIssue]

    return memory_msg


def extract_memoryitem_from_unimsg(
    unimsg: UniMessage[Segment],
    sender: str,
    group_id: str | None,
    user_id: str | None,
    agent_id: str,
    message_id: MsgId,
) -> MemoryItemUnion:
    if not sender or (group_id is None and user_id is None):
        raise ValueError(
            "Sender ID or Group ID or User ID is required to extract MemoryItem"
        )
    memory_msg = extract_memeorymsg_from_unimsg(unimsg)
    related_msg_id = None
    if unimsg.has(Reply):
        reply = unimsg.get(Reply)[0]
        reply = cast(Reply, reply)
        related_msg_id = reply.id  # 当作是消息id

    if group_id is None and user_id is not None:
        return PrivateMemoryItem(
            message_id=message_id,
            sender=sender,
            user_id=user_id,
            agent_id=agent_id,
            related_message_id=related_msg_id,
            content=memory_msg,
        )
    elif group_id is not None:
        return GroupMemoryItem(
            message_id=message_id,
            sender=sender,
            group_id=group_id,
            agent_id=agent_id,
            related_message_id=related_msg_id,
            content=memory_msg,
        )
    else:
        raise ValueError(
            "Either Group ID or User ID must be provided to extract MemoryItem"
        )


async def convert_and_store_item(
    agent_id: str,
    uni_msg: UniMessage,
    group_id: str | None,
    user_id: str | None,
    sender: str,
    msg_id: MsgId | str,
) -> MemoryItemUnion:
    """将 UniMessage 转换为 MemoryItem"""
    msg_id = str(msg_id)
    item = extract_memoryitem_from_unimsg(
        unimsg=uni_msg,
        sender=sender,
        user_id=user_id,
        group_id=group_id,
        agent_id=agent_id,
        message_id=msg_id,
    )
    await item.content.ensure_local()
    await memory_item_repo.save_memory_item(item)
    return item
