"""llm/tools/tool_group/web_search.py 单元测试"""

import asyncio

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.web_search import (
    web_search as _search_wrapped,
    web_fetch as _fetch_wrapped,
)

_search = _search_wrapped.__wrapped__
_fetch = _fetch_wrapped.__wrapped__

import nonebot_plugin_wtfllm.llm.deps as _deps

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_ctx():
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id="u1", agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"WebSearch"},
    )
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


MODULE = "nonebot_plugin_wtfllm.llm.tools.tool_group.web_search"


class TestWebSearch:
    @pytest.mark.asyncio
    @patch(f"{MODULE}._search_cache", new_callable=dict)
    async def test_cache_hit(self, mock_cache):
        mock_cache["cached_query"] = '{"results": "cached"}'
        ctx = _make_ctx()
        # Need to use patch to inject our cache
        with patch(f"{MODULE}._search_cache", mock_cache):
            result = await _search(ctx, query="cached_query")
        assert "cached" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}._search_cache", {})
    @patch(f"{MODULE}._ddgs")
    async def test_search_success(self, mock_ddgs):
        mock_ddgs.text = MagicMock(
            return_value=[{"title": "Test", "href": "http://test.com", "body": "content"}]
        )
        ctx = _make_ctx()
        result = await _search(ctx, query="test query")
        assert "Test" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}._search_cache", {})
    @patch(f"{MODULE}._ddgs")
    async def test_search_empty_results(self, mock_ddgs):
        mock_ddgs.text = MagicMock(return_value=[])
        ctx = _make_ctx()
        result = await _search(ctx, query="empty")
        assert "未找到" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}._search_cache", {})
    @patch(f"{MODULE}._ddgs")
    async def test_search_timeout(self, mock_ddgs):
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(100)

        mock_ddgs.text = MagicMock(side_effect=asyncio.TimeoutError)
        ctx = _make_ctx()
        result = await _search(ctx, query="timeout")
        assert "出错" in result


class TestWebFetch:
    @pytest.mark.asyncio
    async def test_invalid_url_scheme(self):
        ctx = _make_ctx()
        result = await _fetch(ctx, url_or_href="file:///etc/passwd")
        assert "无效" in result

    @pytest.mark.asyncio
    async def test_no_netloc(self):
        ctx = _make_ctx()
        result = await _fetch(ctx, url_or_href="not-a-url")
        assert "无效" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.fetch_url", return_value=None)
    async def test_fetch_empty_download(self, mock_fetch):
        ctx = _make_ctx()
        result = await _fetch(ctx, url_or_href="https://example.com")
        assert "无法抓取" in result

    @pytest.mark.asyncio
    @patch(f"{MODULE}.extract", return_value="Extracted content")
    @patch(f"{MODULE}.fetch_url", return_value="<html>content</html>")
    async def test_fetch_success(self, mock_fetch, mock_extract):
        ctx = _make_ctx()
        result = await _fetch(ctx, url_or_href="https://example.com/article")
        assert result == "Extracted content"

    @pytest.mark.asyncio
    @patch(f"{MODULE}.extract", return_value=None)
    @patch(f"{MODULE}.fetch_url", return_value="<html>empty</html>")
    async def test_fetch_no_extract(self, mock_fetch, mock_extract):
        ctx = _make_ctx()
        result = await _fetch(ctx, url_or_href="https://example.com")
        assert "无法提取" in result
