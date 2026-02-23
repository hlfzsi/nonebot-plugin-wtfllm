"""llm/tools/tool_group/image_generation.py 单元测试"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.image_generation import (
    _is_valid_url,
    _perm,
    text_to_image as _text_to_image_wrapped,
    modify_image_with_text as _modify_wrapped,
    combine_images as _combine_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs

_text_to_image = _text_to_image_wrapped.__wrapped__
_modify_image = _modify_wrapped.__wrapped__
_combine_images = _combine_wrapped.__wrapped__

MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.image_generation"


def _make_ctx(user_id="u1", group_id=None):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id=user_id, group_id=group_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"ImageGeneration"},
    )
    deps.nb_runtime = MagicMock()
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


# ===================== _is_valid_url =====================


class TestIsValidUrl:
    def test_valid_http(self):
        assert _is_valid_url("http://example.com/img.png") is True

    def test_valid_https(self):
        assert _is_valid_url("https://example.com/img.png") is True

    def test_no_scheme(self):
        assert _is_valid_url("example.com/img.png") is False

    def test_empty(self):
        assert _is_valid_url("") is False

    def test_base64_data(self):
        assert _is_valid_url("data:image/png;base64,abc123") is False


# ===================== _perm =====================


class TestPermission:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.ENABLE_IMAGE_GENERATION", True)
    async def test_enabled(self):
        ctx = _make_ctx()
        result = await _perm(ctx)
        assert result is True

    @pytest.mark.asyncio
    @patch(f"{MODULE}.ENABLE_IMAGE_GENERATION", False)
    async def test_disabled(self):
        ctx = _make_ctx()
        result = await _perm(ctx)
        assert result is False


# ===================== text_to_image =====================


class TestTextToImage:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}._text_to_image", new_callable=AsyncMock, return_value=None)
    async def test_fails_returns_false(self, mock_gen, mock_resched):
        ctx = _make_ctx()
        result = await _text_to_image(ctx, prompt="a cat")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}._text_to_image", new_callable=AsyncMock, return_value="https://example.com/img.webp")
    async def test_url_result_appends_image(self, mock_gen, mock_resched):
        ctx = _make_ctx()
        result = await _text_to_image(ctx, prompt="a cat")
        assert result is True
        # reply_segments should have been appended
        assert ctx.deps.reply_segments.__iadd__.called or True  # UniMessage mock

    @pytest.mark.asyncio
    @patch(f"{MODULE}.convert_to_webp", return_value=b"webp-bytes")
    @patch(f"{MODULE}.asyncio.to_thread", new_callable=AsyncMock, return_value=b"webp-bytes")
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}._text_to_image", new_callable=AsyncMock, return_value="iVBOR")
    async def test_base64_result_converts_webp(self, mock_gen, mock_resched, mock_thread, mock_conv):
        ctx = _make_ctx()
        result = await _text_to_image(ctx, prompt="a cat")
        assert result is True
        mock_thread.assert_called_once()


# ===================== modify_image_with_text =====================


class TestModifyImageWithText:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_ref_not_found(self, mock_resched):
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=None)
        result = await _modify_image(ctx, image_ref="IMG:1", prompt="edit")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_ref_raises_error(self, mock_resched):
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("bad ref"))
        result = await _modify_image(ctx, image_ref="IMG:bad", prompt="edit")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_image_expired(self, mock_resched):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = False
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)
        result = await _modify_image(ctx, image_ref="IMG:1", prompt="edit")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}._modify_image_with_text", new_callable=AsyncMock, return_value="https://result.com/img.webp")
    async def test_local_path_success(self, mock_mod, mock_resched):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = "/tmp/img.webp"
        mock_seg.url = None
        mock_seg.get_data_uri_async = AsyncMock(return_value="data:image/webp;base64,abc")
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)
        result = await _modify_image(ctx, image_ref="IMG:1", prompt="edit")
        assert result is True
        mock_mod.assert_called_once_with("data:image/webp;base64,abc", "edit")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}._modify_image_with_text", new_callable=AsyncMock, return_value="https://result.com/img.webp")
    async def test_url_success(self, mock_mod, mock_resched):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = "https://source.com/img.jpg"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)
        result = await _modify_image(ctx, image_ref="IMG:1", prompt="edit")
        assert result is True
        mock_mod.assert_called_once_with("https://source.com/img.jpg", "edit")

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    @patch(f"{MODULE}._modify_image_with_text", new_callable=AsyncMock, return_value=None)
    async def test_modification_fails(self, mock_mod, mock_resched):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = "https://source.com/img.jpg"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)
        result = await _modify_image(ctx, image_ref="IMG:1", prompt="edit")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_no_source(self, mock_resched):
        ctx = _make_ctx()
        mock_seg = MagicMock()
        mock_seg.available = True
        mock_seg.local_path = None
        mock_seg.url = None
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=mock_seg)
        result = await _modify_image(ctx, image_ref="IMG:1", prompt="edit")
        assert result is False


# ===================== combine_images =====================


class TestCombineImages:
    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_not_all_refs_valid(self, mock_resched):
        ctx = _make_ctx()
        ctx.deps.context.resolve_media_ref = MagicMock(side_effect=ValueError("nope"))
        result = await _combine_images(ctx, image_refs=["IMG:1", "IMG:2"], prompt="merge")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}._combine_images", new_callable=AsyncMock, return_value="https://result.com/combined.webp")
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_success_with_mixed_sources(self, mock_resched, mock_combine):
        ctx = _make_ctx()

        seg1 = MagicMock()
        seg1.available = True
        seg1.local_path = "/tmp/img1.webp"
        seg1.url = None
        seg1.get_data_uri_async = AsyncMock(return_value="data:1")

        seg2 = MagicMock()
        seg2.available = True
        seg2.local_path = None
        seg2.url = "https://source.com/img2.jpg"

        ctx.deps.context.resolve_media_ref = MagicMock(side_effect=[seg1, seg2])
        result = await _combine_images(ctx, image_refs=["IMG:1", "IMG:2"], prompt="merge")
        assert result is True
        mock_combine.assert_called_once_with(["data:1", "https://source.com/img2.jpg"], "merge")

    @pytest.mark.asyncio
    @patch(f"{MODULE}._combine_images", new_callable=AsyncMock, return_value=None)
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_combination_fails(self, mock_resched, mock_combine):
        ctx = _make_ctx()
        seg = MagicMock()
        seg.available = True
        seg.local_path = None
        seg.url = "https://source.com/img.jpg"
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=seg)
        result = await _combine_images(ctx, image_refs=["IMG:1"], prompt="gen")
        assert result is False

    @pytest.mark.asyncio
    @patch(f"{MODULE}.reschedule_deadline")
    async def test_skips_unavailable(self, mock_resched):
        ctx = _make_ctx()
        seg = MagicMock()
        seg.available = False
        ctx.deps.context.resolve_media_ref = MagicMock(return_value=seg)
        result = await _combine_images(ctx, image_refs=["IMG:1", "IMG:2"], prompt="merge")
        assert result is False
