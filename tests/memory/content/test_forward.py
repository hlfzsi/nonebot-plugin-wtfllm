# tests/memory/content/test_forward.py
"""ForwardSegment / Node 及递归工具函数的单元测试"""

import pytest
from unittest.mock import MagicMock

from nonebot_plugin_wtfllm.memory.content.segments import (
    TextSegment,
    ImageSegment,
    MentionSegment,
    ForwardSegment,
    Node,
)
from nonebot_plugin_wtfllm.memory.content.message import Message

from nonebot_plugin_wtfllm.memory.context import LLMContext

# NOTE: TestDeepFindAndUpdateSegment / TestDeepFindSegmentNode / TestDeepGetSegments
# 主要测试 Message 实例方法；末尾另有 backward-compat 测试覆盖 utils 委托函数。


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_ctx() -> LLMContext:
    """创建具备完整 provider 的 LLMContext"""
    return LLMContext.create()


def _make_forward_with_image() -> tuple[ForwardSegment, ImageSegment]:
    """创建包含图片的转发消息"""
    img = ImageSegment(url="https://example.com/img.png", created_at=100)
    node = Node(
        sender="user_A",
        content=Message.create([TextSegment(content="看这张图"), img]),
    )
    fwd = ForwardSegment(children=[node], created_at=200)
    return fwd, img


def _make_nested_forward() -> tuple[ForwardSegment, ImageSegment]:
    """创建嵌套转发（转发内再嵌套转发，内含图片）"""
    img = ImageSegment(url="https://example.com/deep.png", created_at=50)
    inner_node = Node(
        sender="user_B",
        content=Message.create([img]),
    )
    inner_fwd = ForwardSegment(children=[inner_node], created_at=60)
    outer_node = Node(
        sender="user_A",
        content=Message.create([TextSegment(content="转发内容"), inner_fwd]),
    )
    outer_fwd = ForwardSegment(children=[outer_node], created_at=100)
    return outer_fwd, img


# ─── ForwardSegment 基础测试 ─────────────────────────────────────────────────


class TestForwardSegment:
    """ForwardSegment 类型基础测试"""

    def test_create_empty(self):
        fwd = ForwardSegment(children=[])
        assert fwd.type == "forward"
        assert fwd.children == []

    def test_create_with_nodes(self):
        node = Node(
            sender="user_1",
            content=Message.create([TextSegment(content="hello")]),
        )
        fwd = ForwardSegment(children=[node])
        assert len(fwd.children) == 1
        assert fwd.children[0].sender == "user_1"

    def test_unique_key(self):
        node = Node(
            sender="user_1",
            content=Message.create([TextSegment(content="a", created_at=10)]),
        )
        fwd = ForwardSegment(children=[node], created_at=100)
        key = fwd.unique_key
        assert "forward" in key
        assert "user_1" in key

    def test_unique_key_differs(self):
        node_a = Node(
            sender="user_1",
            content=Message.create([TextSegment(content="a", created_at=10)]),
        )
        node_b = Node(
            sender="user_2",
            content=Message.create([TextSegment(content="b", created_at=20)]),
        )
        fwd_a = ForwardSegment(children=[node_a], created_at=100)
        fwd_b = ForwardSegment(children=[node_b], created_at=100)
        assert fwd_a.unique_key != fwd_b.unique_key

    def test_format_content_empty(self):
        fwd = ForwardSegment(children=[])
        ctx = _make_ctx()
        result = fwd._format_content(ctx)
        assert "共0条" in result

    def test_format_content_with_text_nodes(self):
        node1 = Node(
            sender="user_1",
            content=Message.create([TextSegment(content="你好")]),
        )
        node2 = Node(
            sender="user_2",
            content=Message.create([TextSegment(content="世界")]),
        )
        fwd = ForwardSegment(children=[node1, node2], created_at=100)
        fwd._message_id = "parent_msg"
        ctx = _make_ctx()
        result = fwd._format_content(ctx)

        assert "共2条" in result
        assert "你好" in result
        assert "世界" in result
        assert "合并转发结束" in result

    def test_format_content_with_image(self):
        """转发中的图片应正确获得 media ref"""
        fwd, img = _make_forward_with_image()
        fwd._message_id = "parent_msg"
        ctx = _make_ctx()
        result = fwd._format_content(ctx)
        assert "IMG:" in result

    def test_to_llm_context_sets_message_id(self):
        """to_llm_context 应将 message_id 传播到内嵌 segment"""
        fwd, img = _make_forward_with_image()
        ctx = _make_ctx()
        fwd.to_llm_context(ctx, "parent_msg_123", memory_ref=1)

        # 验证内嵌 ImageSegment 的 message_id 指向父 MemoryItem
        assert img.message_id == "parent_msg_123"

    def test_to_llm_context_nested_forwards(self):
        """嵌套转发中的深层 segment 也应获得正确的 message_id"""
        outer_fwd, deep_img = _make_nested_forward()
        ctx = _make_ctx()
        outer_fwd.to_llm_context(ctx, "parent_msg_456", memory_ref=1)

        assert deep_img.message_id == "parent_msg_456"


# ─── 序列化往返测试 ───────────────────────────────────────────────────────────


class TestForwardSerialization:
    """ForwardSegment 序列化/反序列化往返测试"""

    def test_message_with_forward_roundtrip(self):
        """包含 ForwardSegment 的 Message 可正确序列化和反序列化"""
        img = ImageSegment(url="https://example.com/a.png", created_at=10)
        node = Node(
            sender="user_1",
            content=Message.create([TextSegment(content="hi", created_at=5), img]),
        )
        fwd = ForwardSegment(children=[node], created_at=20)
        msg = Message.create([TextSegment(content="前文", created_at=1), fwd])

        json_str = msg.model_dump_json()
        restored = Message.model_validate_json(json_str)

        assert len(restored.segments) == 2
        assert restored.segments[0].type == "text"
        assert restored.segments[1].type == "forward"

        restored_fwd = restored.segments[1]
        assert isinstance(restored_fwd, ForwardSegment)
        assert len(restored_fwd.children) == 1
        assert restored_fwd.children[0].sender == "user_1"

        inner_segs = restored_fwd.children[0].content.segments
        assert len(inner_segs) == 2
        assert inner_segs[0].type == "text"
        assert inner_segs[1].type == "image"

    def test_nested_forward_roundtrip(self):
        """嵌套转发的序列化/反序列化"""
        outer_fwd, _ = _make_nested_forward()
        msg = Message.create([outer_fwd])

        json_str = msg.model_dump_json()
        restored = Message.model_validate_json(json_str)

        restored_fwd = restored.segments[0]
        assert isinstance(restored_fwd, ForwardSegment)
        inner_content = restored_fwd.children[0].content
        inner_fwd = inner_content.segments[1]
        assert isinstance(inner_fwd, ForwardSegment)
        deep_img = inner_fwd.children[0].content.segments[0]
        assert isinstance(deep_img, ImageSegment)
        assert deep_img.url == "https://example.com/deep.png"


# ─── deep_find_and_update_segment 测试 ───────────────────────────────────────


class TestDeepFindAndUpdateSegment:
    """递归 segment 查找并更新测试"""

    def test_update_top_level(self):
        """顶层 segment 更新"""
        img = ImageSegment(url="https://example.com/a.png", created_at=10)
        msg = Message.create([img])

        result = msg.deep_find_and_update(img, {"desc": "一张猫图"})
        assert result is True
        assert msg.segments[0].type == "image"
        updated_img = msg.segments[0]
        assert isinstance(updated_img, ImageSegment)
        assert updated_img.desc == "一张猫图"

    def test_update_in_forward(self):
        """ForwardSegment 内部的 segment 更新"""
        fwd, img = _make_forward_with_image()
        msg = Message.create([fwd])

        result = msg.deep_find_and_update(img, {"desc": "风景照"})
        assert result is True

        # 验证更新生效
        inner_segs = msg.segments[0]
        assert isinstance(inner_segs, ForwardSegment)
        updated = inner_segs.children[0].content.segments[1]
        assert isinstance(updated, ImageSegment)
        assert updated.desc == "风景照"

    def test_update_in_nested_forward(self):
        """嵌套转发内部的 segment 更新"""
        outer_fwd, deep_img = _make_nested_forward()
        msg = Message.create([outer_fwd])

        result = msg.deep_find_and_update(deep_img, {"desc": "深层图片"})
        assert result is True

        # 验证深层更新生效
        inner_fwd = outer_fwd.children[0].content.segments[1]
        assert isinstance(inner_fwd, ForwardSegment)
        updated = inner_fwd.children[0].content.segments[0]
        assert isinstance(updated, ImageSegment)
        assert updated.desc == "深层图片"

    def test_not_found(self):
        """找不到目标 segment 时返回 False"""
        msg = Message.create([TextSegment(content="hello")])
        target = ImageSegment(url="https://nonexistent.com/x.png", created_at=999)

        result = msg.deep_find_and_update(target, {"desc": "test"})
        assert result is False


# ─── deep_find_segment_node 测试 ─────────────────────────────────────────────


class TestDeepFindSegmentNode:
    """Message.deep_find_node 实例方法测试"""

    def test_top_level_returns_none(self):
        """顶层 segment 不在任何 Node 内，返回 None"""
        img = ImageSegment(url="https://example.com/a.png", created_at=10)
        msg = Message.create([img])
        assert msg.deep_find_node(img) is None

    def test_find_in_forward(self):
        """找到 ForwardSegment 内 segment 所在的 Node"""
        fwd, img = _make_forward_with_image()
        msg = Message.create([fwd])

        node = msg.deep_find_node(img)
        assert node is not None
        assert node.sender == "user_A"

    def test_find_in_nested_forward(self):
        """找到嵌套转发内 segment 所在的 Node"""
        outer_fwd, deep_img = _make_nested_forward()
        msg = Message.create([outer_fwd])

        node = msg.deep_find_node(deep_img)
        assert node is not None
        assert node.sender == "user_B"

    def test_not_found(self):
        """找不到 segment 返回 None"""
        fwd, _ = _make_forward_with_image()
        msg = Message.create([fwd])
        target = ImageSegment(url="https://nonexistent.com/x.png", created_at=999)
        assert msg.deep_find_node(target) is None


# ─── deep_get_segments 测试 ──────────────────────────────────────────────────


class TestDeepGetSegments:
    """Message.deep_get 实例方法 (list 收集) 测试"""

    def test_collect_top_level_only(self):
        """只有顶层时收集正确"""
        img = ImageSegment(url="https://example.com/a.png", created_at=10)
        msg = Message.create([TextSegment(content="hi"), img])
        result = msg.deep_get(ImageSegment)
        assert len(result) == 1
        assert result[0] is img

    def test_collect_from_forward(self):
        """从 ForwardSegment 内部收集"""
        fwd, img = _make_forward_with_image()
        msg = Message.create([fwd])

        result = msg.deep_get(ImageSegment)
        assert len(result) == 1
        assert result[0] is img

    def test_collect_mixed_levels(self):
        """同时收集顶层和转发内的 segment"""
        top_img = ImageSegment(url="https://example.com/top.png", created_at=1)
        fwd, inner_img = _make_forward_with_image()
        msg = Message.create([top_img, fwd])

        result = msg.deep_get(ImageSegment)
        assert len(result) == 2

    def test_collect_from_nested_forward(self):
        """从嵌套转发内收集"""
        outer_fwd, deep_img = _make_nested_forward()
        msg = Message.create([outer_fwd])

        result = msg.deep_get(ImageSegment)
        assert len(result) == 1
        assert result[0] is deep_img

    def test_collect_text_across_levels(self):
        """跨层收集 TextSegment"""
        outer_fwd, _ = _make_nested_forward()
        msg = Message.create([TextSegment(content="顶层"), outer_fwd])

        result = msg.deep_get(TextSegment)
        # "顶层" + outer_node 的 "转发内容" = 2 个 TextSegment
        assert len(result) == 2

    def test_empty_message(self):
        msg = Message.create()
        assert msg.deep_get(ImageSegment) == []

    def test_collect_mentions_in_forward(self):
        """收集转发内的 MentionSegment"""
        mention = MentionSegment(user_id="user_123")
        node = Node(sender="user_A", content=Message.create([mention]))
        fwd = ForwardSegment(children=[node])
        msg = Message.create([fwd])

        result = msg.deep_get(MentionSegment)
        assert len(result) == 1
        assert result[0].user_id == "user_123"


# ─── Message.deep_has / deep_get 测试 ────────────────────────────────────────


class TestMessageDeepMethods:
    """Message 的 deep_has 和 deep_get 方法测试"""

    def test_deep_has_top_level(self):
        msg = Message.create([ImageSegment(url="https://a.com/b.png")])
        assert msg.deep_has(ImageSegment) is True
        assert msg.deep_has(MentionSegment) is False

    def test_deep_has_in_forward(self):
        fwd, _ = _make_forward_with_image()
        msg = Message.create([fwd])
        assert msg.deep_has(ImageSegment) is True
        assert msg.deep_has("image") is True

    def test_deep_has_in_nested(self):
        outer_fwd, _ = _make_nested_forward()
        msg = Message.create([outer_fwd])
        assert msg.deep_has(ImageSegment) is True

    def test_deep_get_from_forward(self):
        fwd, img = _make_forward_with_image()
        msg = Message.create([fwd])
        result = msg.deep_get(ImageSegment)
        assert len(result) == 1
        assert result[0] is img

    def test_shallow_has_does_not_find_nested(self):
        """普通 has 不应找到嵌套在 ForwardSegment 内的 segment"""
        fwd, _ = _make_forward_with_image()
        msg = Message.create([fwd])
        # 浅层 has 只检查顶层，ForwardSegment 的 type 是 "forward" 不是 "image"
        assert msg.has("image") is False
        # 但 deep_has 可以找到
        assert msg.deep_has("image") is True


# ─── 集成场景：渲染 + RefProvider 反查 ───────────────────────────────────────


class TestForwardRenderingIntegration:
    """ForwardSegment 渲染与 RefProvider 反查集成测试"""

    def test_image_in_forward_gets_media_ref_and_message_id(self):
        """转发内的图片渲染后可通过 RefProvider 反查"""
        from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem

        img = ImageSegment(url="https://example.com/test.jpg", created_at=100)
        node = Node(
            sender="user_1",
            content=Message.create([TextSegment(content="图"), img]),
        )
        fwd = ForwardSegment(children=[node])
        msg = Message.create([fwd])

        item = GroupMemoryItem(
            message_id="msg_001",
            sender="user_0",
            group_id="group_1",
            agent_id="agent_1",
            content=msg,
        )

        ctx = _make_ctx()
        # 注册实体
        item.register_entities(ctx)

        # 渲染
        ref = ctx.ref_provider.next_memory_ref(item)
        rendered = item.content.to_llm_context(ctx, item.message_id, ref)

        # 验证渲染输出包含 IMG 引用
        assert "IMG:" in rendered

        # 验证 RefProvider 可反查到 ImageSegment
        resolved_img = ctx.ref_provider.get_media_typed("IMG:1", ImageSegment)
        assert resolved_img is not None
        assert resolved_img.url == "https://example.com/test.jpg"

        # 验证 ImageSegment.message_id 指向父 MemoryItem
        assert resolved_img.message_id == "msg_001"

    def test_nested_forward_image_has_correct_message_id(self):
        """嵌套转发中深层图片的 message_id 指向最外层 MemoryItem"""
        from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem

        outer_fwd, deep_img = _make_nested_forward()
        msg = Message.create([outer_fwd])

        item = GroupMemoryItem(
            message_id="msg_outer",
            sender="user_0",
            group_id="group_1",
            agent_id="agent_1",
            content=msg,
        )

        ctx = _make_ctx()
        item.register_entities(ctx)
        ref = ctx.ref_provider.next_memory_ref(item)
        item.content.to_llm_context(ctx, item.message_id, ref)

        # 嵌套最深层的图片也应指向 msg_outer
        assert deep_img.message_id == "msg_outer"

    def test_register_entities_covers_forward_nodes(self):
        """register_entities 应注册转发中所有 node 的 sender"""
        from nonebot_plugin_wtfllm.memory.items.base_items import GroupMemoryItem

        mention = MentionSegment(user_id="user_mentioned")
        node1 = Node(
            sender="fwd_user_A", content=Message.create([TextSegment(content="hi")])
        )
        node2 = Node(
            sender="fwd_user_B",
            group_id="fwd_group",
            content=Message.create([mention]),
        )
        fwd = ForwardSegment(children=[node1, node2])

        item = GroupMemoryItem(
            message_id="msg_1",
            sender="user_0",
            group_id="group_1",
            agent_id="agent_1",
            content=Message.create([fwd]),
        )

        ctx = _make_ctx()
        item.register_entities(ctx)

        # 验证 node senders 已注册
        assert ctx.alias_provider.get_alias("fwd_user_A") is not None
        assert ctx.alias_provider.get_alias("fwd_user_B") is not None
        # 验证 node.group_id 已注册
        assert ctx.alias_provider.get_alias("fwd_group") is not None
        # 验证嵌套内的 mention 已注册
        assert ctx.alias_provider.get_alias("user_mentioned") is not None


# ─── backward-compat: utils 委托函数 ─────────────────────────────────────────


class TestUtilsBackwardCompat:
    """utils.py 中的独立函数仍应正确委托到 Message 实例方法"""

    def test_deep_find_and_update_segment_delegates(self):
        img = ImageSegment(url="https://example.com/a.png", created_at=10)
        msg = Message.create([img])
        result = msg.deep_find_and_update(img, {"desc": "ok"})
        assert result is True
        assert msg.segments[0].desc == "ok"

    def test_deep_find_segment_node_delegates(self):
        fwd, img = _make_forward_with_image()
        msg = Message.create([fwd])
        node = msg.deep_find_node(img)
        assert node is not None
        assert node.sender == "user_A"

    def test_deep_get_segments_delegates(self):
        fwd, img = _make_forward_with_image()
        msg = Message.create([fwd])
        result = msg.deep_get(ImageSegment)
        assert len(result) == 1
        assert result[0] is img
