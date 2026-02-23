"""llm/tools/tool_group/files.py 单元测试"""

import pytest
from unittest.mock import MagicMock, patch

from nonebot_plugin_wtfllm.llm.tools.tool_group.files import (
    _perm,
    list_files_in_directory as _list_wrapped,
    read_local_file as _read_wrapped,
    search_file as _search_wrapped,
)

import nonebot_plugin_wtfllm.llm.deps as _deps

_list_files = _list_wrapped.__wrapped__
_read_file = _read_wrapped.__wrapped__
_search = _search_wrapped.__wrapped__

AgentDeps = _deps.AgentDeps
IDs = _deps.IDs


def _make_ctx(user_id="u1"):
    from nonebot_plugin_wtfllm.memory import MemoryContextBuilder

    mock_ctx_builder = MagicMock(spec=MemoryContextBuilder)
    deps = AgentDeps(
        ids=IDs(user_id=user_id, agent_id="a1"),
        context=mock_ctx_builder,
        active_tool_groups={"files"},
    )
    ctx = MagicMock()
    ctx.deps = deps
    return ctx


class TestPerm:
    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.files.APP_CONFIG")
    async def test_admin_user(self, mock_config):
        mock_config.admin_users = ["admin1"]
        ctx = _make_ctx(user_id="admin1")
        assert await _perm(ctx) is True

    @pytest.mark.asyncio
    @patch("nonebot_plugin_wtfllm.llm.tools.tool_group.files.APP_CONFIG")
    async def test_non_admin_user(self, mock_config):
        mock_config.admin_users = ["admin1"]
        ctx = _make_ctx(user_id="normal_user")
        assert await _perm(ctx) is False


class TestListFiles:
    def test_list_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        ctx = _make_ctx()
        result = _list_files(ctx, directory_path=str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result

    def test_empty_directory(self, tmp_path):
        ctx = _make_ctx()
        result = _list_files(ctx, directory_path=str(tmp_path))
        assert "目录为空" in result

    def test_nonexistent_directory(self):
        ctx = _make_ctx()
        result = _list_files(ctx, directory_path="/nonexistent/path/xyz")
        assert "无法访问" in result


class TestReadFile:
    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("file content", encoding="utf-8")
        ctx = _make_ctx()
        result = await _read_file(ctx, file_path=str(f))
        assert result == "file content"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        ctx = _make_ctx()
        result = await _read_file(ctx, file_path="/nonexistent/file.txt")
        assert "无法读取" in result


class TestSearchFile:
    @pytest.mark.asyncio
    async def test_search_finds_match(self, tmp_path):
        (tmp_path / "match.txt").write_text("hello world keyword here", encoding="utf-8")
        (tmp_path / "no_match.txt").write_text("nothing here", encoding="utf-8")
        ctx = _make_ctx()
        result = await _search(ctx, dir_path=str(tmp_path), keyword="keyword")
        assert "match.txt" in result
        assert "no_match.txt" not in result

    @pytest.mark.asyncio
    async def test_search_no_match(self, tmp_path):
        (tmp_path / "file.txt").write_text("abc", encoding="utf-8")
        ctx = _make_ctx()
        result = await _search(ctx, dir_path=str(tmp_path), keyword="xyz")
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_search_nonexistent_dir(self):
        ctx = _make_ctx()
        result = await _search(ctx, dir_path="/nonexistent/dir", keyword="test")
        # os.walk 在不存在目录时不抛异常，直接返回空
        assert "未找到" in result or "无法访问" in result
