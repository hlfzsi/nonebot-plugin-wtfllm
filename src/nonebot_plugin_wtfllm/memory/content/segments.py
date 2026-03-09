"""Segment 类型定义"""

import asyncio
import os
import uuid
from pathlib import Path
from abc import ABC, abstractmethod
from time import time
from typing import TYPE_CHECKING, Annotated, Any, List, Literal, Union, cast

import aiofiles
import httpx
import orjson
import pybase64
import filetype
from lxml import etree
from pydantic import (
    BaseModel,
    Field,
    Tag,
    PrivateAttr,
    field_serializer,
    field_validator,
)
from .condense import condense_text, condense_forward
from ...media import convert_to_webp
from ...utils import MEDIA_DIR, get_http_client, APP_CONFIG

if TYPE_CHECKING:
    from ..context import LLMContext
    from .message import Message

MEDIA_TYPES = Literal[
    "text", "mention", "image", "file", "audio", "video", "forward", "unknown", "hyper"
]


class BaseSegment(BaseModel, ABC):
    """Segment 基类"""

    type: Any = Field(..., description="片段类型")
    created_at: int = Field(
        default_factory=lambda: int(time()), description="创建时间戳"
    )

    _message_id: str | None = PrivateAttr(default=None)

    def to_llm_context(
        self, ctx: "LLMContext", message_id: str, memory_ref: int | None = None
    ) -> str:
        """转换为 LLM 上下文格式"""
        self._message_id = message_id
        return self._format_content(ctx, memory_ref)

    @property
    def message_id(self) -> str:
        """获取所属消息ID"""
        if self._message_id is None:
            raise ValueError("Message ID has not been set for this segment")
        return self._message_id

    @property
    @abstractmethod
    def unique_key(self) -> str:
        """获取唯一键"""
        ...

    @abstractmethod
    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        """子类实际上的转换为 LLM 上下文格式"""
        ...

    def __eq__(self, value: object) -> bool:
        if type(self) is not type(value):
            return False
        value = cast(BaseSegment, value)
        return self.unique_key == value.unique_key

    def __hash__(self) -> int:
        return hash(self.unique_key)


class MediaBaseSegment(BaseSegment, ABC):
    """媒体片段基类"""

    url: str | None = Field(default=None, description="媒体URL")
    local_path: Path | None = Field(default=None, description="本地文件路径")
    desc: str | None = Field(default=None, description="媒体描述")

    expired: bool = Field(
        default=False, description="是否已过期，过期的媒体不应再被使用"
    )

    _bytes: bytes | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if not self.url and not self.local_path and not self.expired:
            raise ValueError(
                "MediaBaseSegment must have either 'url' or 'local_path' or be expired"
            )

    @field_serializer("local_path")
    def serialize_local_path(self, v: Path | None) -> Path | None:
        if v is None:
            return None
        _path = v
        if _path.is_relative_to(MEDIA_DIR):
            path = _path.relative_to(MEDIA_DIR)
        elif _path.is_absolute():
            path = _path
        else:
            path = _path.resolve()

        return path

    @field_validator("local_path", mode="after")
    @classmethod
    def validate_local_path(cls, v: Path | None) -> Path | None:
        if v is None:
            return None
        _path = v
        if _path.is_absolute():
            path = _path
        else:
            path = MEDIA_DIR / _path
        return path

    @property
    def available(self) -> bool:
        """判断媒体是否可用"""
        if self.expired:
            return False
        elif self.local_path and os.path.exists(self.local_path):
            return True
        elif self.url:
            return True
        return False

    @abstractmethod
    async def ensure_local(
        self,
        client: httpx.AsyncClient | None = None,
        path: Path | None = None,
    ) -> None:
        """
        将媒体保存到本地路径

        Args:
            client (httpx.AsyncClient | None): HTTP 客户端，若为 None 则使用全局客户端
            path (Path | None): 本地完整保存路径，若为 None 则自动生成
        """
        ...

    async def get_bytes_async(
        self, client: httpx.AsyncClient | None = None, download: bool = False
    ) -> bytes:
        """获取媒体的字节内容"""
        if self._bytes:
            return self._bytes
        if self.local_path:
            async with aiofiles.open(self.local_path, "rb") as f:
                data = await f.read()
                self._bytes = data
                return data
        elif self.url and download:
            if client is None:
                client = get_http_client()
            resp = await client.get(self.url)
            resp.raise_for_status()
            self._bytes = resp.content
            return self._bytes
        elif self.url and not download:
            raise ValueError("Cannot get bytes from URL without downloading")
        else:
            raise ValueError("Cannot get bytes without local_path or url")

    async def get_mime_type_async(
        self, client: httpx.AsyncClient | None = None, download: bool = False
    ) -> str:
        """获取媒体的 MIME 类型"""
        _bytes = await self.get_bytes_async(client, download)
        kind = filetype.guess(_bytes)
        return kind.mime if kind else "application/octet-stream"

    async def get_extension_async(
        self, client: httpx.AsyncClient | None = None, download: bool = False
    ) -> str:
        """获取媒体的扩展名"""
        _bytes = await self.get_bytes_async(client, download)
        kind = filetype.guess(_bytes)
        return kind.extension if kind else "bin"

    async def get_data_uri_async(
        self, client: httpx.AsyncClient | None = None, download: bool = False
    ) -> str:
        """获取图片的 Data URI"""
        _bytes = await self.get_bytes_async(client, download)
        mime = await self.get_mime_type_async(client, download)
        b64_data = pybase64.b64encode(_bytes).decode("utf-8")
        return f"data:{mime};base64,{b64_data}"

    def unbound_local(self, expired: bool = True) -> None:
        """解除本地文件绑定，并删除文件"""
        if self.local_path and os.path.exists(self.local_path):
            os.remove(self.local_path)
        self.local_path = None
        self.expired = expired

    @property
    def unique_key(self) -> str:
        """获取唯一键"""
        return f"self.type:{self.type}-url:{self.url}-local_path:{self.local_path}-desc:{self.desc}-created_at:{self.created_at}"

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        ref = ctx.ref_provider.next_media_ref(self, memory_ref)
        desc = self.desc
        if ctx.condense and desc:
            desc, _ = condense_text(desc, APP_CONFIG.memory_item_max_chars)
        return f"[{ref}{' - ' + desc if desc else ''}]"


class UnknownSegment(BaseSegment):
    """未知片段类型"""

    type: Literal["unknown"] = Field(default="unknown", description="片段类型")

    original_type: str | None = Field(..., description="原始片段类型")

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        return "[未知格式消息]"

    @property
    def unique_key(self) -> str:
        return f"self.type:{self.type}-original_type:{self.original_type}-created_at:{self.created_at}"


class TextSegment(BaseSegment):
    """文本片段"""

    type: Literal["text"] = Field(default="text", description="片段类型")
    content: str = Field(..., description="文本内容")

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        if ctx.condense:
            text, _ = condense_text(self.content, APP_CONFIG.memory_item_max_chars)
            return text
        else:
            return self.content

    @property
    def unique_key(self) -> str:
        return (
            f"self.type:{self.type}-content:{self.content}-created_at:{self.created_at}"
        )


class EmojiSegment(BaseSegment):
    """表情片段"""

    type: Literal["emoji"] = Field(default="emoji", description="片段类型")
    name: str = Field(..., description="表情名称")
    url: str | None = Field(default=None, description="表情URL")

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        return f"[表情: {self.name}]"

    @property
    def unique_key(self) -> str:
        return f"self.type:{self.type}-name:{self.name}-url:{self.url}-created_at:{self.created_at}"


class Node(BaseModel):
    """消息树节点"""

    sender: str = Field(..., description="发送者ID")
    group_id: str | None = Field(default=None, description="群ID")
    content: "Message" = Field(..., description="消息内容")

    @property
    def created_at(self) -> int:
        """获取节点中最早片段的创建时间戳"""
        return self.content.created_at


class ForwardSegment(BaseSegment):
    """合并消息片段/消息树"""

    type: Literal["forward"] = Field(default="forward", description="片段类型")
    children: List[Node] = Field(..., description="合并的消息内容列表")

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        if not self.children:
            return "[合并转发消息, 共0条, 空]"

        if ctx.condense:
            condensed = condense_forward(
                self.children,
                ctx,
                self.message_id or "",
                memory_ref,
                APP_CONFIG.memory_item_max_chars,
            )
            if condensed is not None:
                return condensed

        lines: List[str] = [f"[合并转发消息, 共{len(self.children)}条:]"]
        for node in self.children:
            sender_alias = ctx.alias_provider.get_alias(node.sender) or node.sender
            content_str = node.content.to_llm_context(
                ctx, self.message_id or "", memory_ref
            )
            lines.append(f"  > {sender_alias}: {content_str}")
        lines.append("[合并转发结束]")
        return "\n".join(lines)

    @property
    def unique_key(self) -> str:
        children_keys = "-".join(
            f"{node.sender}:{node.created_at}" for node in self.children
        )
        return f"self.type:{self.type}-children:[{children_keys}]-created_at:{self.created_at}"


class MentionSegment(BaseSegment):
    """提及片段"""

    type: Literal["mention"] = Field(default="mention", description="片段类型")
    user_id: str | None = Field(default=None, description="被提及的用户ID")
    at_all: bool = Field(default=False, description="是否为 @all 提及")

    def model_post_init(self, __context: Any) -> None:
        if not self.user_id and not self.at_all:
            raise ValueError(
                "MentionSegment must have either 'user_id' or 'at_all' set"
            )

        if self.user_id and self.at_all:
            raise ValueError(
                "MentionSegment cannot have both 'user_id' and 'at_all' set"
            )

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        if not self.user_id and self.at_all:
            return "<@全体成员>"
        elif self.user_id:
            alias = ctx.alias_provider.get_alias(self.user_id) or self.user_id
            return f"<@{alias}>"
        else:
            raise ValueError(
                "MentionSegment must have either 'user_id' or 'at_all' set"
            )

    @property
    def unique_key(self) -> str:
        return f"self.type:{self.type}-user_id:{self.user_id}-at_all:{self.at_all}-created_at:{self.created_at}"


class ImageSegment(MediaBaseSegment):
    """图片片段"""

    type: Literal["image"] = Field(default="image", description="片段类型")

    async def ensure_local(
        self, client: httpx.AsyncClient | None = None, path: Path | None = None
    ) -> None:
        if self.local_path:
            return

        _bytes = await self.get_bytes_async(client, download=True)

        webp_bytes = await asyncio.to_thread(
            convert_to_webp, _bytes, quality=85, max_size=(800, 600)
        )
        path = path or MEDIA_DIR / f"image_{uuid.uuid4().hex}.webp"
        async with aiofiles.open(path, "wb") as f:
            await f.write(webp_bytes)
        self.local_path = path
        self._bytes = webp_bytes


class VideoSegment(MediaBaseSegment):
    """视频片段"""

    type: Literal["video"] = Field(default="video", description="片段类型")
    duration: int | None = Field(default=None, description="视频时长，单位秒")

    @property
    def unique_key(self) -> str:
        return super().unique_key + f"-duration:{self.duration}"

    async def ensure_local(
        self, client: httpx.AsyncClient | None = None, path: Path | None = None
    ) -> None: ...


class FileSegment(MediaBaseSegment):
    """文件片段"""

    type: Literal["file"] = Field(default="file", description="片段类型")
    filename: str = Field(..., description="文件名")

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        ref = ctx.ref_provider.next_media_ref(self, memory_ref)
        return f"[{ref}: {self.filename}{' - ' + self.desc if self.desc else ''}]"

    @property
    def unique_key(self) -> str:
        return super().unique_key + f"-filename:{self.filename}"

    async def ensure_local(
        self, client: httpx.AsyncClient | None = None, path: Path | None = None
    ) -> None: ...


class AudioSegment(MediaBaseSegment):
    """音频片段"""

    type: Literal["audio"] = Field(default="audio", description="片段类型")
    duration: int | None = Field(default=None, description="音频时长，单位秒")

    @property
    def unique_key(self) -> str:
        return super().unique_key + f"-duration:{self.duration}"

    async def ensure_local(
        self, client: httpx.AsyncClient | None = None, path: Path | None = None
    ) -> None:
        _bytes = await self.get_bytes_async(client, download=True)
        extension = filetype.guess_extension(_bytes)
        path = path or MEDIA_DIR / f"audio_{uuid.uuid4().hex}.{extension}"
        async with aiofiles.open(path, "wb") as f:
            await f.write(_bytes)
        self.local_path = path
        self._bytes = _bytes


class HyperSegment(BaseSegment):
    """超级消息片段"""

    type: Literal["hyper"] = Field(default="hyper", description="片段类型")
    format: Literal["xml", "json"] = Field(..., description="超级消息格式")
    content: str = Field(..., description="超级消息内容")

    def _format_content(self, ctx: "LLMContext", memory_ref: int | None = None) -> str:
        header = f"[Rich Message: {self.format.upper()}]"
        content = self.content
        if self.format == "json":
            try:
                data = orjson.loads(self.content)
                content = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
            except (orjson.JSONDecodeError, ValueError):
                pass

        elif self.format == "xml":
            try:
                parser = etree.XMLParser(recover=True)
                root = etree.fromstring(self.content.encode("utf-8"), parser=parser)
                content = etree.tostring(root, pretty_print=True, encoding="unicode")
                return f"[Rich Message: XML]\n```xml\n{content}\n```"
            except (etree.XMLSyntaxError, UnicodeDecodeError):
                pass

        return f"{header}\n```{self.format}\n{content}\n```"

    @property
    def unique_key(self) -> str:
        return f"self.type:{self.type}-format:{self.format}-content:{self.content}-created_at:{self.created_at}"


Segment = Annotated[
    Union[
        Annotated[TextSegment, Tag("text")],
        Annotated[EmojiSegment, Tag("emoji")],
        Annotated[MentionSegment, Tag("mention")],
        Annotated[ImageSegment, Tag("image")],
        Annotated[VideoSegment, Tag("video")],
        Annotated[FileSegment, Tag("file")],
        Annotated[AudioSegment, Tag("audio")],
        Annotated[UnknownSegment, Tag("unknown")],
        Annotated[ForwardSegment, Tag("forward")],
        Annotated[HyperSegment, Tag("hyper")],
    ],
    Field(discriminator="type"),
]
