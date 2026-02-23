"""memory/items/tool_call_summary.py 单元测试"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.items.tool_call_summary import ToolCallSummaryBlock


class TestToolCallSummaryProperties:
    def test_source_id(self):
        block = ToolCallSummaryBlock()
        assert block.source_id.startswith("tool-call-summary-")

    def test_priority(self):
        block = ToolCallSummaryBlock()
        assert block.priority == 1

    def test_sort_key(self):
        block = ToolCallSummaryBlock()
        key = block.sort_key
        assert key[0] == 0
        assert key[1] == block.source_id


class TestToolCallSummaryHash:
    def test_hash_is_identity_based(self):
        b1 = ToolCallSummaryBlock()
        b2 = ToolCallSummaryBlock()
        assert hash(b1) != hash(b2)
        assert hash(b1) == hash(b1)


class TestRegisterAllAlias:
    def test_is_noop(self):
        block = ToolCallSummaryBlock()
        ctx = MagicMock()
        block.register_all_alias(ctx)


class TestToLLMContext:
    def test_empty_tool_names(self):
        block = ToolCallSummaryBlock(tool_names=[])
        ctx = MagicMock()
        assert block.to_llm_context(ctx) == ""

    def test_single_tool(self):
        block = ToolCallSummaryBlock(tool_names=["search"])
        ctx = MagicMock()
        result = block.to_llm_context(ctx)
        assert "- search" in result

    def test_deduplication_and_sorting(self):
        block = ToolCallSummaryBlock(tool_names=["z_tool", "a_tool", "z_tool", "m_tool"])
        ctx = MagicMock()
        result = block.to_llm_context(ctx)
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert lines == ["- a_tool", "- m_tool", "- z_tool"]

    def test_with_prefix_and_suffix(self):
        block = ToolCallSummaryBlock(
            tool_names=["tool1"],
            prefix="== Tools ==",
            suffix="== End ==",
        )
        ctx = MagicMock()
        result = block.to_llm_context(ctx)
        assert result.startswith("== Tools ==")
        assert result.endswith("== End ==")
        assert "- tool1" in result

    def test_prefix_only(self):
        block = ToolCallSummaryBlock(tool_names=["t"], prefix="Header")
        ctx = MagicMock()
        result = block.to_llm_context(ctx)
        assert result.startswith("Header")

    def test_suffix_only(self):
        block = ToolCallSummaryBlock(tool_names=["t"], suffix="Footer")
        ctx = MagicMock()
        result = block.to_llm_context(ctx)
        assert result.endswith("Footer")
