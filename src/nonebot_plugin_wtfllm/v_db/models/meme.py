"""Meme 的 Qdrant Payload 模型"""

__all__ = ["MemePayload"]

import asyncio
import hashlib
import time
import uuid
from pathlib import Path
from typing import ClassVar, List, Self

import pybase64
import aiofiles
import httpx
from PIL import Image
from pydantic import ConfigDict, Field, field_validator
from qdrant_client import models

from ...media import convert_to_webp
from ...utils import MEMES_DIR
from .base import VectorModel


class MemePayload(VectorModel):
    """Meme 的 Qdrant Payload 模型

    Attributes:
        storage_id: Meme 存储唯一ID（作为 Qdrant Point ID）
        file_path: 相对于 MEMES_DIR 的文件路径，格式: {storage_id}.webp
        docs: Meme 的描述文本/适用场景
        tags: 与 Meme 相关的标签列表
        created_at: Meme 创建时间戳
        uploader_id: 上传者的真实用户ID
        raw_message_id: 关联的原始消息ID
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    collection_name: ClassVar[str] = "wtfllm_meme"
    indexes: ClassVar[dict[str, models.PayloadSchemaType]] = {
        "tags": models.PayloadSchemaType.KEYWORD,
        "uploader_id": models.PayloadSchemaType.KEYWORD,
        "raw_message_id": models.PayloadSchemaType.KEYWORD,
        "created_at": models.PayloadSchemaType.INTEGER,
    }
    point_id_field: ClassVar[str] = "storage_id"

    storage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Meme 存储唯一ID"
    )
    file_path: str = Field(
        ..., description="相对于 MEMES_DIR 的文件路径，格式: {storage_id}.webp"
    )
    raw_message_id: str = Field(..., description="关联的原始消息ID")
    docs: str = Field(..., description="Meme 的描述文本/适用场景")
    tags: List[str] = Field(default_factory=list, description="与 Meme 相关的标签列表")
    created_at: int = Field(
        default_factory=lambda: int(time.time()), description="Meme 创建时间戳"
    )
    uploader_id: str = Field(..., description="上传者的用户ID")

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """标签标准化：小写、去重、排序"""
        if not v:
            return []
        return sorted(set(tag.strip().lower() for tag in v if tag.strip()))

    @field_validator("docs", mode="before")
    @classmethod
    def validate_docs(cls, v: str) -> str:
        """确保 docs 非空"""
        if not v or not v.strip():
            raise ValueError("docs cannot be empty")
        return v.strip()

    @property
    def point_id(self) -> str:
        """返回 storage_id 作为 Qdrant Point ID"""
        return self.storage_id

    def get_text_for_embedding(self) -> str:
        """返回 docs 作为向量嵌入文本内容"""
        return self.docs

    @property
    def full_path(self) -> Path:
        """获取完整文件路径"""
        return MEMES_DIR / self.file_path

    def get_bytes(self) -> bytes:
        """读取图片数据"""
        return self.full_path.read_bytes()

    async def get_bytes_async(self) -> bytes:
        """读取图片数据"""
        async with aiofiles.open(self.full_path, "rb") as f:
            return await f.read()

    @property
    def bytes_base64(self) -> str:
        """Base64 编码的图片数据"""
        return pybase64.b64encode(self.get_bytes()).decode("utf-8")

    @property
    def data_uri(self) -> str:
        """获取data URI"""
        return f"data:image/webp;base64,{self.bytes_base64}"

    async def delete_file(self) -> bool:
        """删除关联文件"""
        try:
            self.full_path.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def has_tag(self, tag: str) -> bool:
        """检查是否包含指定标签（大小写不敏感）"""
        return tag.strip().lower() in self.tags

    def has_any_tag(self, tags: List[str]) -> bool:
        """检查是否包含任意一个指定标签"""
        normalized_tags = {t.strip().lower() for t in tags}
        return bool(normalized_tags & set(self.tags))

    def has_all_tags(self, tags: List[str]) -> bool:
        """检查是否包含所有指定标签"""
        normalized_tags = {t.strip().lower() for t in tags}
        return normalized_tags.issubset(set(self.tags))

    def add_tags(self, new_tags: List[str]) -> Self:
        """添加标签并返回新实例"""
        combined = sorted(
            set(self.tags + [t.strip().lower() for t in new_tags if t.strip()])
        )
        return self.model_copy(update={"tags": combined})

    def remove_tags(self, tags_to_remove: List[str]) -> Self:
        """移除标签并返回新实例"""
        to_remove = {t.strip().lower() for t in tags_to_remove}
        filtered = [t for t in self.tags if t not in to_remove]
        return self.model_copy(update={"tags": filtered})

    def get_image_dimensions(self) -> tuple[int, int]:
        """获取图片宽高 (width, height)"""
        with Image.open(self.full_path) as img:
            return img.size

    async def get_image_dimensions_async(self) -> tuple[int, int]:
        """异步获取图片宽高"""
        return await asyncio.to_thread(self.get_image_dimensions)

    def to_thumbnail_bytes(
        self, max_size: tuple[int, int] = (128, 128), quality: int = 60
    ) -> bytes:
        """生成缩略图"""
        return convert_to_webp(
            self.full_path.read_bytes(), quality=quality, max_size=max_size
        )

    @staticmethod
    def convert_to_webp(image_bytes: bytes, quality: int = 95) -> bytes:
        """将图片转换为 WebP 格式

        Args:
            image_bytes: 原始图片的二进制数据
            quality: WebP 压缩质量 (0-100)

        Returns:
            转换后的 WebP 图片二进制数据

        Raises:
            IOError: 如果图片无法被识别或处理
        """
        return convert_to_webp(image_bytes, quality=quality, max_size=(600, 450))

    @staticmethod
    def _generate_storage_id(data: bytes) -> str:
        """根据图片数据生成唯一的 storage_id"""
        return str(uuid.uuid5(uuid.NAMESPACE_OID, hashlib.sha256(data).hexdigest()))

    @classmethod
    async def from_url(
        cls,
        url: str,
        raw_message_id: str,
        docs: str,
        tags: List[str],
        uploader_id: str,
    ) -> "MemePayload":
        """从 URL 创建 MemePayload 实例

        Args:
            url: Meme 图片的 URL
            raw_message_id: 关联的原始消息ID
            docs: Meme 的描述文本
            tags: 与 Meme 相关的标签列表
            uploader_id: 上传者的真实用户ID

        Returns:
            MemePayload 实例
        """

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            bytes_ = response.content

        webp_bytes = await asyncio.to_thread(cls.convert_to_webp, bytes_)

        storage_id = cls._generate_storage_id(bytes_)
        file_name = f"{storage_id}.webp"

        file_full_path = MEMES_DIR / file_name

        async with aiofiles.open(file_full_path, "wb") as f:
            await f.write(webp_bytes)

        return cls(
            storage_id=storage_id,
            file_path=file_name,
            raw_message_id=raw_message_id,
            docs=docs,
            tags=tags,
            uploader_id=uploader_id,
        )

    @classmethod
    async def from_path(
        cls,
        path: str,
        raw_message_id: str,
        docs: str,
        tags: List[str],
        uploader_id: str,
    ) -> "MemePayload":
        """从本地文件路径创建 MemePayload 实例

        Args:
            path: 本地文件路径
            raw_message_id: 关联的原始消息ID
            docs: Meme 的描述文本
            tags: 与 Meme 相关的标签列表
            uploader_id: 上传者的真实用户ID

        Returns:
            MemePayload 实例
        """
        async with aiofiles.open(path, "rb") as f:
            bytes_ = await f.read()

        storage_id = cls._generate_storage_id(bytes_)
        file_name = f"{storage_id}.webp"

        webp_bytes = await asyncio.to_thread(cls.convert_to_webp, bytes_)

        file_full_path = MEMES_DIR / file_name

        async with aiofiles.open(file_full_path, "wb") as f:
            await f.write(webp_bytes)

        return cls(
            storage_id=storage_id,
            file_path=file_name,
            raw_message_id=raw_message_id,
            docs=docs,
            tags=tags,
            uploader_id=uploader_id,
        )

    @classmethod
    async def from_raw(
        cls,
        raw: bytes,
        raw_message_id: str,
        docs: str,
        tags: List[str],
        uploader_id: str,
    ) -> "MemePayload":
        """从原始二进制数据创建 MemePayload 实例

        Args:
            raw: 原始图片的二进制数据
            raw_message_id: 关联的原始消息ID
            docs: Meme 的描述文本
            tags: 与 Meme 相关的标签列表
            uploader_id: 上传者的真实用户ID
        Returns:
            MemePayload 实例
        """
        webp_bytes = await asyncio.to_thread(cls.convert_to_webp, raw)

        storage_id = cls._generate_storage_id(webp_bytes)
        file_name = f"{storage_id}.webp"

        file_full_path = MEMES_DIR / file_name
        async with aiofiles.open(file_full_path, "wb") as f:
            await f.write(webp_bytes)

        return cls(
            storage_id=storage_id,
            file_path=file_name,
            raw_message_id=raw_message_id,
            docs=docs,
            tags=tags,
            uploader_id=uploader_id,
        )
