"""llm/tools/tool_group/memes.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.memes import (
    save_meme as _save_wrapped,
    search_memes as _search_wrapped,
    list_memes as _list_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs

_save_meme = _save_wrapped.__wrapped__
_search_memes = _search_wrapped.__wrapped__
_list_memes = _list_wrapped.__wrapped__

MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.memes"


def _make_ctx(user_id="u1", group_id=None):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id=user_id, group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"Memes"},
    )
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


# ===================== save_meme =====================


class TestSaveMeme:
    @pytest.mark.asyncio
    async def test_duplicate_ref_returns_early(self):
        ctx = _make_ctx()
        ctx.deps.caches["meme_save_IMG:1"] = True

        result = await _save_meme(ctx, img_ref="IMG:1", description="test", tags=["tag"])
        assert "重复" in result

    @pytest.mark.asyncio
    async def test_ref_not_found(self):
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=None)

        result = await _save_meme(ctx, img_ref="IMG:99", description="test", tags=["tag"])
        assert "无法找到" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_image_expired(self, mock_repo):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = False
        mock_seg.message_id = "m1"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)

        mock_item = MagicMock()
        mock_item.content.deep_find_node.return_value = None
        mock_item.sender = "uploader1"
        mock_repo.get_by_message_id = AsyncMock(return_value=mock_item)

        result = await _save_meme(ctx, img_ref="IMG:1", description="test", tags=["tag"])
        assert "过期" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.meme_repo")
    @patch(f"{MODULE}.MemePayload")
    @patch(f"{MODULE}.memory_item_repo")
    async def test_saves_from_local_path(self, mock_mem_repo, MockPayload, mock_meme_repo):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = "/tmp/meme.webp"
        mock_seg.url = None
        mock_seg.message_id = "m1"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)

        mock_item = MagicMock()
        mock_item.content.deep_find_node.return_value = None
        mock_item.sender = "uploader1"
        mock_mem_repo.get_by_message_id = AsyncMock(return_value=mock_item)

        mock_payload = MagicMock()
        MockPayload.from_path = AsyncMock(return_value=mock_payload)
        mock_meme_repo.save_meme = AsyncMock()

        result = await _save_meme(ctx, img_ref="IMG:1", description="cat", tags=["funny"])
        assert "已保存" in result
        MockPayload.from_path.assert_called_once()
        mock_meme_repo.save_meme.assert_called_once_with(mock_payload)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.meme_repo")
    @patch(f"{MODULE}.MemePayload")
    @patch(f"{MODULE}.memory_item_repo")
    async def test_saves_from_url(self, mock_mem_repo, MockPayload, mock_meme_repo):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = "https://example.com/meme.jpg"
        mock_seg.message_id = "m1"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)

        mock_item = MagicMock()
        mock_item.content.deep_find_node.return_value = None
        mock_item.sender = "uploader1"
        mock_mem_repo.get_by_message_id = AsyncMock(return_value=mock_item)

        mock_payload = MagicMock()
        MockPayload.from_url = AsyncMock(return_value=mock_payload)
        mock_meme_repo.save_meme = AsyncMock()

        result = await _save_meme(ctx, img_ref="IMG:1", description="cat", tags=["funny"])
        assert "已保存" in result
        MockPayload.from_url.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{MODULE}.memory_item_repo")
    async def test_no_url_or_path_raises(self, mock_mem_repo):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = None
        mock_seg.message_id = "m1"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)

        mock_item = MagicMock()
        mock_item.content.deep_find_node.return_value = None
        mock_item.sender = "uploader1"
        mock_mem_repo.get_by_message_id = AsyncMock(return_value=mock_item)

        with pytest.raises(ValueError, match="URL"):
            await _save_meme(ctx, img_ref="IMG:1", description="cat", tags=["tag"])


# ===================== search_memes =====================


class TestSearchMemes:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_text_only(self, mock_repo, mock_config):
        mock_config.llm_support_vision = False
        mock_result = MagicMock()
        mock_result.item.storage_id = "uuid-1"
        mock_result.item.docs = "funny cat"
        mock_repo.search_by_text = AsyncMock(return_value=[mock_result])

        ctx = _make_ctx()
        result = await _search_memes(ctx, query="cat", tags=None)
        assert "uuid-1" in result.return_value
        mock_repo.search_by_text.assert_called_once_with("cat", 5)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_with_tags(self, mock_repo, mock_config):
        mock_config.llm_support_vision = False
        mock_result = MagicMock()
        mock_result.item.storage_id = "uuid-2"
        mock_result.item.docs = "doge"
        mock_repo.search_by_text_with_tags = AsyncMock(return_value=[mock_result])

        ctx = _make_ctx()
        result = await _search_memes(ctx, query="dog", tags=["meme"])
        assert "uuid-2" in result.return_value
        mock_repo.search_by_text_with_tags.assert_called_once_with("dog", ["meme"], 5)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_vision_includes_images(self, mock_repo, mock_config):
        mock_config.llm_support_vision = True
        mock_meme = MagicMock()
        mock_meme.storage_id = "uuid-3"
        mock_meme.docs = "pepe"
        mock_meme.get_bytes_async = AsyncMock(return_value=b"img-bytes")
        mock_result = MagicMock()
        mock_result.item = mock_meme
        mock_repo.search_by_text = AsyncMock(return_value=[mock_result])

        ctx = _make_ctx()
        result = await _search_memes(ctx, query="frog")
        assert len(result.content) == 1

    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_empty_results(self, mock_repo, mock_config):
        mock_config.llm_support_vision = False
        mock_repo.search_by_text = AsyncMock(return_value=[])

        ctx = _make_ctx()
        result = await _search_memes(ctx, query="nothing")
        assert result.return_value == ""


# ===================== list_memes =====================


class TestListMemes:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_without_uploader(self, mock_repo, mock_config):
        mock_config.llm_support_vision = False
        mock_meme = MagicMock()
        mock_meme.storage_id = "uuid-4"
        mock_meme.docs = "emoji"
        mock_repo.get_recent = AsyncMock(return_value=[mock_meme])

        ctx = _make_ctx()
        result = await _list_memes(ctx, limit=10)
        assert "uuid-4" in result.return_value
        mock_repo.get_recent.assert_called_once_with(limit=10, uploader_id=None)

    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_with_uploader_alias(self, mock_repo, mock_config):
        mock_config.llm_support_vision = False
        mock_repo.get_recent = AsyncMock(return_value=[])

        ctx = _make_ctx()
        ctx.deps.context.resolve_aliases = MagicMock(return_value="real_uid")

        result = await _list_memes(ctx, limit=5, uploader_id="alias_name")
        ctx.deps.context.resolve_aliases.assert_called_once_with("alias_name")
        mock_repo.get_recent.assert_called_once_with(limit=5, uploader_id="real_uid")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.APP_CONFIG")
    @patch(f"{MODULE}.meme_repo")
    async def test_empty_list(self, mock_repo, mock_config):
        mock_config.llm_support_vision = False
        mock_repo.get_recent = AsyncMock(return_value=[])

        ctx = _make_ctx()
        result = await _list_memes(ctx, limit=10)
        assert result.return_value == ""
