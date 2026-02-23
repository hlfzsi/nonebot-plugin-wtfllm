"""v_db/models 扩展单元测试

覆盖 MemePayload 工厂方法及文件操作。
"""

import asyncio
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest
from PIL import Image

from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload


def _make_png_bytes(width: int = 10, height: int = 10) -> bytes:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_rgba_png_bytes() -> bytes:
    image = Image.new("RGBA", (20, 20), color=(0, 0, 255, 128))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestMemePayloadValidation:
    """MemePayload 字段验证测试"""

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="docs cannot be empty"):
            MemePayload(
                file_path="test.webp",
                raw_message_id="r1",
                docs="   ",
                uploader_id="u1",
            )

    def test_tags_dedup_and_sort(self):
        payload = MemePayload(
            file_path="test.webp",
            raw_message_id="r1",
            docs="doc",
            tags=["Zebra", "alpha", "zebra", "ALPHA"],
            uploader_id="u1",
        )
        assert payload.tags == ["alpha", "zebra"]

    def test_empty_tags(self):
        payload = MemePayload(
            file_path="test.webp",
            raw_message_id="r1",
            docs="doc",
            tags=[],
            uploader_id="u1",
        )
        assert payload.tags == []

    def test_tags_with_whitespace_only(self):
        payload = MemePayload(
            file_path="test.webp",
            raw_message_id="r1",
            docs="doc",
            tags=["  ", "", "valid"],
            uploader_id="u1",
        )
        assert payload.tags == ["valid"]


class TestMemePayloadPointId:
    """MemePayload point_id 测试"""

    def test_point_id_is_storage_id(self):
        payload = MemePayload(
            storage_id="custom-uuid",
            file_path="custom.webp",
            raw_message_id="r1",
            docs="doc",
            uploader_id="u1",
        )
        assert payload.point_id == "custom-uuid"


class TestMemePayloadEmbeddingText:
    """MemePayload get_text_for_embedding 测试"""

    def test_returns_docs(self):
        payload = MemePayload(
            file_path="test.webp",
            raw_message_id="r1",
            docs="embedding target",
            uploader_id="u1",
        )
        assert payload.get_text_for_embedding() == "embedding target"


class TestMemePayloadTagHelpers:
    """MemePayload 标签辅助方法测试"""

    @pytest.fixture
    def tagged_payload(self):
        return MemePayload(
            file_path="test.webp",
            raw_message_id="r1",
            docs="doc",
            tags=["cat", "dog", "funny"],
            uploader_id="u1",
        )

    def test_has_tag_case_insensitive(self, tagged_payload):
        assert tagged_payload.has_tag("CAT") is True
        assert tagged_payload.has_tag("bird") is False

    def test_has_any_tag(self, tagged_payload):
        assert tagged_payload.has_any_tag(["bird", "cat"]) is True
        assert tagged_payload.has_any_tag(["bird", "fish"]) is False

    def test_has_all_tags(self, tagged_payload):
        assert tagged_payload.has_all_tags(["cat", "dog"]) is True
        assert tagged_payload.has_all_tags(["cat", "bird"]) is False

    def test_add_tags_returns_new(self, tagged_payload):
        new_payload = tagged_payload.add_tags(["Bird", "cat"])
        assert "bird" in new_payload.tags
        assert "cat" in new_payload.tags
        # 原实例不变
        assert "bird" not in tagged_payload.tags

    def test_remove_tags_returns_new(self, tagged_payload):
        new_payload = tagged_payload.remove_tags(["dog"])
        assert "dog" not in new_payload.tags
        assert "dog" in tagged_payload.tags


class TestMemePayloadConvertToWebp:
    """MemePayload.convert_to_webp 静态方法测试"""

    def test_convert_rgb_png(self):
        data = _make_png_bytes()
        webp = MemePayload.convert_to_webp(data, quality=80)
        assert isinstance(webp, bytes)
        assert webp[:4] == b"RIFF"  # WebP magic bytes

    def test_convert_rgba_png(self):
        data = _make_rgba_png_bytes()
        webp = MemePayload.convert_to_webp(data, quality=80)
        assert isinstance(webp, bytes)
        assert webp[:4] == b"RIFF"


class TestMemePayloadFileOperations:
    """MemePayload 文件读写操作测试（使用临时目录）"""

    @pytest.fixture
    def meme_with_file(self, tmp_path):
        """创建带实际文件的 MemePayload"""
        file_name = "test_meme.webp"
        # 把真实 webp 写到 tmp_path
        png_data = _make_png_bytes(50, 50)
        webp_data = MemePayload.convert_to_webp(png_data)
        (tmp_path / file_name).write_bytes(webp_data)

        payload = MemePayload(
            storage_id="file_test",
            file_path=file_name,
            raw_message_id="r1",
            docs="file test",
            uploader_id="u1",
        )

        # Patch MEMES_DIR
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            yield payload, tmp_path

    def test_full_path(self, meme_with_file):
        payload, tmp_path = meme_with_file
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            assert payload.full_path == tmp_path / "test_meme.webp"

    def test_get_bytes(self, meme_with_file):
        payload, tmp_path = meme_with_file
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            data = payload.get_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0

    @pytest.mark.asyncio
    async def test_get_bytes_async(self, meme_with_file):
        payload, tmp_path = meme_with_file
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            data = await payload.get_bytes_async()
            assert isinstance(data, bytes)
            assert len(data) > 0

    def test_get_image_dimensions(self, meme_with_file):
        payload, tmp_path = meme_with_file
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            w, h = payload.get_image_dimensions()
            assert w > 0
            assert h > 0

    def test_to_thumbnail_bytes(self, meme_with_file):
        payload, tmp_path = meme_with_file
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            thumb = payload.to_thumbnail_bytes(max_size=(32, 32))
            assert isinstance(thumb, bytes)
            assert thumb[:4] == b"RIFF"

    @pytest.mark.asyncio
    async def test_delete_file(self, meme_with_file):
        payload, tmp_path = meme_with_file
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            assert payload.full_path.exists()
            result = await payload.delete_file()
            assert result is True
            assert not payload.full_path.exists()

    @pytest.mark.asyncio
    async def test_delete_file_missing(self, tmp_path):
        payload = MemePayload(
            storage_id="missing_file",
            file_path="nonexistent.webp",
            raw_message_id="r1",
            docs="missing",
            uploader_id="u1",
        )
        with patch(
            "nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path
        ):
            result = await payload.delete_file()
            assert result is True  # missing_ok=True


class TestMemePayloadToPayload:
    """MemePayload to_payload / from_payload 往返测试"""

    def test_roundtrip(self):
        payload = MemePayload(
            storage_id="rt_meme",
            file_path="rt_meme.webp",
            raw_message_id="r1",
            docs="roundtrip test",
            tags=["test", "round"],
            uploader_id="u1",
            created_at=12345,
        )

        data = payload.to_payload()
        restored = MemePayload.from_payload(data)

        assert restored.storage_id == payload.storage_id
        assert restored.file_path == payload.file_path
        assert restored.docs == payload.docs
        assert restored.tags == payload.tags
        assert restored.uploader_id == payload.uploader_id
        assert restored.created_at == payload.created_at


class TestMemePayloadClassVars:
    """MemePayload 类变量测试"""

    def test_collection_name(self):
        assert MemePayload.collection_name == "wtfllm_meme"

    def test_indexes(self):
        indexes = MemePayload.indexes
        assert "tags" in indexes
        assert "uploader_id" in indexes
        assert "raw_message_id" in indexes
        assert "created_at" in indexes


# ===== VectorModel 基类测试 =====

from abc import abstractmethod
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.v_db.models.base import VectorModel


class _TestModel(VectorModel):
    """用于测试的具体 VectorModel 子类"""

    collection_name = "test_coll"
    indexes = {}
    name: str
    value: int

    @property
    def point_id(self) -> str:
        return self.name

    def get_text_for_embedding(self) -> str:
        return f"{self.name} {self.value}"


class TestVectorModelSubclass:
    """VectorModel __init_subclass__ 验证测试"""

    def test_missing_collection_name_raises(self):
        with pytest.raises(TypeError, match="must define.*'collection_name'"):

            class _Bad(VectorModel):
                indexes = {"field": "keyword"}

                @property
                def point_id(self) -> str:
                    return "id"

                def get_text_for_embedding(self) -> str:
                    return ""

    def test_missing_indexes_raises(self):
        with pytest.raises(TypeError, match="must define.*'indexes'"):

            class _Bad(VectorModel):
                collection_name = "coll"

                @property
                def point_id(self) -> str:
                    return "id"

                def get_text_for_embedding(self) -> str:
                    return ""

    def test_empty_collection_name_raises(self):
        with pytest.raises(TypeError, match="non-empty value for 'collection_name'"):

            class _Bad(VectorModel):
                collection_name = ""
                indexes = {}

                @property
                def point_id(self) -> str:
                    return "id"

                def get_text_for_embedding(self) -> str:
                    return ""

    def test_abstract_subclass_skips_validation(self):
        """抽象子类（未实现所有抽象方法）不触发 class attribute 校验"""

        # Should NOT raise despite missing collection_name/indexes
        class _AbstractChild(VectorModel):
            @abstractmethod
            def some_extra_method(self):
                ...

    def test_valid_subclass(self):
        """合法子类可以正常定义并实例化"""
        instance = _TestModel(name="hello", value=42)
        assert instance.point_id == "hello"
        assert instance.get_text_for_embedding() == "hello 42"


class TestVectorModelPayload:
    """VectorModel to_payload / from_payload 往返测试"""

    def test_to_payload_roundtrip(self):
        original = _TestModel(name="abc", value=99)
        payload = original.to_payload()

        assert isinstance(payload, dict)
        assert payload["name"] == "abc"
        assert payload["value"] == 99

        restored = _TestModel.from_payload(payload)
        assert restored.name == original.name
        assert restored.value == original.value
        assert restored.point_id == original.point_id
        assert restored.get_text_for_embedding() == original.get_text_for_embedding()


class TestInitCollection:
    """VectorModel.init_collection 测试"""

    @pytest.mark.asyncio
    async def test_existing_collection(self, mock_qdrant_client):
        """集合已存在时不应调用 create_collection"""
        mock_qdrant_client.collection_exists = AsyncMock(return_value=True)
        mock_qdrant_client.get_collection = AsyncMock(
            return_value=MagicMock(payload_schema={})
        )

        with patch(
            "nonebot_plugin_wtfllm.v_db.models.base.get_qdrant_client",
            return_value=mock_qdrant_client,
        ):
            await VectorModel.init_collection("test_coll", None)

        mock_qdrant_client.collection_exists.assert_awaited_once_with("test_coll")
        mock_qdrant_client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_new_collection(self, mock_qdrant_client):
        """集合不存在时应调用 create_collection"""
        mock_qdrant_client.collection_exists = AsyncMock(return_value=False)

        with patch(
            "nonebot_plugin_wtfllm.v_db.models.base.get_qdrant_client",
            return_value=mock_qdrant_client,
        ):
            await VectorModel.init_collection("new_coll", None)

        mock_qdrant_client.create_collection.assert_awaited_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "new_coll"

    @pytest.mark.asyncio
    async def test_with_indexes(self, mock_qdrant_client):
        """已有集合 + payload_indexes 时，应为缺失索引调用 create_payload_index"""
        mock_qdrant_client.collection_exists = AsyncMock(return_value=True)

        # 模拟已有 "existing_field" 索引，但缺少 "new_field"
        mock_qdrant_client.get_collection = AsyncMock(
            return_value=MagicMock(payload_schema={"existing_field": MagicMock()})
        )
        mock_qdrant_client.create_payload_index = AsyncMock()

        payload_indexes = {
            "existing_field": "keyword",
            "new_field": "integer",
        }

        with patch(
            "nonebot_plugin_wtfllm.v_db.models.base.get_qdrant_client",
            return_value=mock_qdrant_client,
        ):
            await VectorModel.init_collection("idx_coll", payload_indexes)

        # create_payload_index 应只对 "new_field" 调用一次
        mock_qdrant_client.create_payload_index.assert_awaited_once()
        call_kwargs = mock_qdrant_client.create_payload_index.call_args
        assert call_kwargs.kwargs["field_name"] == "new_field"
        assert call_kwargs.kwargs["field_schema"] == "integer"


# ===================== MemePayload =====================

from unittest.mock import MagicMock


class TestMemePayloadBytesBase64:
    def test_bytes_base64(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        meme = MemePayload(
            storage_id="test_id", file_path="test.webp",
            raw_message_id="msg1", docs="test", tags=["a"],
            uploader_id="u1",
        )
        with patch.object(MemePayload, "get_bytes", return_value=b"test_data"):
            result = meme.bytes_base64
            assert isinstance(result, str)
            # Should be valid base64
            import base64
            decoded = base64.b64decode(result)
            assert decoded == b"test_data"

    def test_data_uri(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        meme = MemePayload(
            storage_id="test_id", file_path="test.webp",
            raw_message_id="msg1", docs="test", tags=["a"],
            uploader_id="u1",
        )
        with patch.object(MemePayload, "get_bytes", return_value=b"test_data"):
            result = meme.data_uri
            assert result.startswith("data:image/webp;base64,")


class TestMemePayloadDeleteFile:
    @pytest.mark.asyncio
    async def test_delete_file_oserror(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        meme = MemePayload(
            storage_id="test_id", file_path="test.webp",
            raw_message_id="msg1", docs="test", tags=["a"],
            uploader_id="u1",
        )
        with patch.object(type(meme), "full_path", new_callable=lambda: property(lambda self: MagicMock(unlink=MagicMock(side_effect=OSError("perm"))))):
            result = await meme.delete_file()
            assert result is False


class TestMemePayloadGenerateStorageId:
    def test_deterministic(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        id1 = MemePayload._generate_storage_id(b"hello")
        id2 = MemePayload._generate_storage_id(b"hello")
        assert id1 == id2

    def test_different_data(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        id1 = MemePayload._generate_storage_id(b"hello")
        id2 = MemePayload._generate_storage_id(b"world")
        assert id1 != id2

    def test_returns_string(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        result = MemePayload._generate_storage_id(b"test")
        assert isinstance(result, str)


class TestMemePayloadGetImageDimensionsAsync:
    @pytest.mark.asyncio
    async def test_calls_to_thread(self):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload
        meme = MemePayload(
            storage_id="test_id", file_path="test.webp",
            raw_message_id="msg1", docs="test", tags=["a"],
            uploader_id="u1",
        )
        with patch("nonebot_plugin_wtfllm.v_db.models.meme.asyncio.to_thread", new_callable=AsyncMock, return_value=(100, 200)) as mock_thread:
            result = await meme.get_image_dimensions_async()
            assert result == (100, 200)


class TestMemePayloadFromUrl:
    @pytest.mark.asyncio
    async def test_from_url(self, tmp_path):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload

        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("nonebot_plugin_wtfllm.v_db.models.meme.httpx.AsyncClient", return_value=mock_client), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.asyncio.to_thread", new_callable=AsyncMock, return_value=b"webp_data"), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.aiofiles") as mock_aiofiles:

            mock_file = AsyncMock()
            mock_aiofiles.open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
            mock_aiofiles.open.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await MemePayload.from_url(
                url="http://example.com/meme.jpg",
                raw_message_id="msg1",
                docs="test meme",
                tags=["funny"],
                uploader_id="u1",
            )
            assert isinstance(result, MemePayload)
            assert result.docs == "test meme"


class TestMemePayloadFromPath:
    @pytest.mark.asyncio
    async def test_from_path(self, tmp_path):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload

        with patch("nonebot_plugin_wtfllm.v_db.models.meme.asyncio.to_thread", new_callable=AsyncMock, return_value=b"webp_data"), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.aiofiles") as mock_aiofiles:

            mock_file_read = AsyncMock()
            mock_file_read.read = AsyncMock(return_value=b"raw_image_data")
            mock_file_write = AsyncMock()

            call_count = [0]
            async def _aenter_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_file_read
                return mock_file_write

            ctx = AsyncMock()
            ctx.__aenter__ = _aenter_side_effect
            ctx.__aexit__ = AsyncMock(return_value=None)
            mock_aiofiles.open.return_value = ctx

            result = await MemePayload.from_path(
                path="/tmp/test.jpg",
                raw_message_id="msg1",
                docs="test",
                tags=["test"],
                uploader_id="u1",
            )
            assert isinstance(result, MemePayload)


class TestMemePayloadFromRaw:
    @pytest.mark.asyncio
    async def test_from_raw(self, tmp_path):
        from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload

        with patch("nonebot_plugin_wtfllm.v_db.models.meme.asyncio.to_thread", new_callable=AsyncMock, return_value=b"webp_data"), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.MEMES_DIR", tmp_path), \
             patch("nonebot_plugin_wtfllm.v_db.models.meme.aiofiles") as mock_aiofiles:

            mock_file = AsyncMock()
            mock_aiofiles.open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
            mock_aiofiles.open.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await MemePayload.from_raw(
                raw=b"raw_image_bytes",
                raw_message_id="msg1",
                docs="test",
                tags=["test"],
                uploader_id="u1",
            )
            assert isinstance(result, MemePayload)
