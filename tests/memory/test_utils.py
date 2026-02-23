# tests/memory/test_utils.py
"""memory/utils.py 单元测试"""

import pytest

from nonebot_plugin_wtfllm.memory.utils import (
    generate_alpha_index,
    parse_alpha_index,
    DirtyStateMarker,
)


class TestGenerateAlphaIndex:
    """generate_alpha_index 函数测试"""

    def test_single_letter_start(self):
        """测试单字母起始 (1 -> A)"""
        assert generate_alpha_index(1) == "A"

    def test_single_letter_end(self):
        """测试单字母结束 (26 -> Z)"""
        assert generate_alpha_index(26) == "Z"

    def test_single_letter_middle(self):
        """测试单字母中间值"""
        assert generate_alpha_index(2) == "B"
        assert generate_alpha_index(13) == "M"
        assert generate_alpha_index(25) == "Y"

    def test_double_letter_start(self):
        """测试双字母起始 (27 -> AA)"""
        assert generate_alpha_index(27) == "AA"

    def test_double_letter_az(self):
        """测试双字母 AZ (52 -> AZ)"""
        assert generate_alpha_index(52) == "AZ"

    def test_double_letter_ba(self):
        """测试双字母 BA (53 -> BA)"""
        assert generate_alpha_index(53) == "BA"

    def test_double_letter_zz(self):
        """测试双字母 ZZ (702 -> ZZ)"""
        # 26 + 26*26 = 26 + 676 = 702
        assert generate_alpha_index(702) == "ZZ"

    def test_triple_letter_start(self):
        """测试三字母起始 (703 -> AAA)"""
        assert generate_alpha_index(703) == "AAA"

    def test_triple_letter_various(self):
        """测试三字母各种情况"""
        assert generate_alpha_index(704) == "AAB"
        assert generate_alpha_index(728) == "AAZ"
        assert generate_alpha_index(729) == "ABA"

    def test_invalid_zero(self):
        """测试无效输入: 0"""
        with pytest.raises(ValueError, match="Index must be positive"):
            generate_alpha_index(0)

    def test_invalid_negative(self):
        """测试无效输入: 负数"""
        with pytest.raises(ValueError, match="Index must be positive"):
            generate_alpha_index(-1)
        with pytest.raises(ValueError, match="Index must be positive"):
            generate_alpha_index(-100)


class TestParseAlphaIndex:
    """parse_alpha_index 函数测试"""

    def test_single_letter_start(self):
        """测试单字母起始 (A -> 1)"""
        assert parse_alpha_index("A") == 1

    def test_single_letter_end(self):
        """测试单字母结束 (Z -> 26)"""
        assert parse_alpha_index("Z") == 26

    def test_single_letter_middle(self):
        """测试单字母中间值"""
        assert parse_alpha_index("B") == 2
        assert parse_alpha_index("M") == 13
        assert parse_alpha_index("Y") == 25

    def test_double_letter_start(self):
        """测试双字母起始 (AA -> 27)"""
        assert parse_alpha_index("AA") == 27

    def test_double_letter_az(self):
        """测试双字母 AZ (AZ -> 52)"""
        assert parse_alpha_index("AZ") == 52

    def test_double_letter_ba(self):
        """测试双字母 BA (BA -> 53)"""
        assert parse_alpha_index("BA") == 53

    def test_double_letter_zz(self):
        """测试双字母 ZZ (ZZ -> 702)"""
        assert parse_alpha_index("ZZ") == 702

    def test_triple_letter_start(self):
        """测试三字母起始 (AAA -> 703)"""
        assert parse_alpha_index("AAA") == 703

    def test_triple_letter_various(self):
        """测试三字母各种情况"""
        assert parse_alpha_index("AAB") == 704
        assert parse_alpha_index("AAZ") == 728
        assert parse_alpha_index("ABA") == 729

    def test_lowercase_input(self):
        """测试小写字母输入（应被转换为大写）"""
        assert parse_alpha_index("a") == 1
        assert parse_alpha_index("z") == 26
        assert parse_alpha_index("aa") == 27
        assert parse_alpha_index("aB") == 28

    def test_invalid_empty_string(self):
        """测试无效输入: 空字符串"""
        with pytest.raises(ValueError, match="Invalid alpha index"):
            parse_alpha_index("")

    def test_invalid_non_alpha(self):
        """测试无效输入: 非字母字符"""
        with pytest.raises(ValueError, match="Invalid alpha index"):
            parse_alpha_index("1")
        with pytest.raises(ValueError, match="Invalid alpha index"):
            parse_alpha_index("A1")
        with pytest.raises(ValueError, match="Invalid alpha index"):
            parse_alpha_index("A B")
        with pytest.raises(ValueError, match="Invalid alpha index"):
            parse_alpha_index("A-B")


class TestAlphaIndexRoundtrip:
    """generate_alpha_index 和 parse_alpha_index 往返一致性测试"""

    @pytest.mark.parametrize("n", [1, 13, 26, 27, 52, 53, 100, 500, 702, 703, 1000])
    def test_roundtrip_generate_then_parse(self, n: int):
        """测试 generate -> parse 往返"""
        alpha = generate_alpha_index(n)
        result = parse_alpha_index(alpha)
        assert result == n

    @pytest.mark.parametrize(
        "alpha",
        ["A", "M", "Z", "AA", "AZ", "BA", "ZZ", "AAA", "ABC", "XYZ"],
    )
    def test_roundtrip_parse_then_generate(self, alpha: str):
        """测试 parse -> generate 往返"""
        n = parse_alpha_index(alpha)
        result = generate_alpha_index(n)
        assert result == alpha


class TestDirtyStateMarker:
    """DirtyStateMarker 装饰器测试"""

    def test_marks_dirty_sets_dirty_flag(self):
        """测试 marks_dirty 执行后设置 _dirty = True"""

        class TestClass:
            def __init__(self):
                self._dirty = False

            @DirtyStateMarker.marks_dirty
            def modify(self, value: int) -> int:
                return value * 2

            def _ensure_clean(self) -> None:
                pass

        obj = TestClass()
        assert obj._dirty is False
        result = obj.modify(5)
        assert result == 10
        assert obj._dirty is True

    def test_marks_dirty_already_dirty(self):
        """测试 marks_dirty 对已脏对象的行为"""

        class TestClass:
            def __init__(self):
                self._dirty = True

            @DirtyStateMarker.marks_dirty
            def modify(self) -> str:
                return "done"

            def _ensure_clean(self) -> None:
                pass

        obj = TestClass()
        result = obj.modify()
        assert result == "done"
        assert obj._dirty is True

    def test_marks_clean_sets_clean_flag(self):
        """测试 marks_clean 执行后设置 _dirty = False"""

        class TestClass:
            def __init__(self):
                self._dirty = True

            @DirtyStateMarker.marks_clean
            def save(self) -> str:
                return "saved"

            def _ensure_clean(self) -> None:
                pass

        obj = TestClass()
        assert obj._dirty is True
        result = obj.save()
        assert result == "saved"
        assert obj._dirty is False

    def test_marks_clean_already_clean(self):
        """测试 marks_clean 对已净对象的行为"""

        class TestClass:
            def __init__(self):
                self._dirty = False

            @DirtyStateMarker.marks_clean
            def save(self) -> str:
                return "saved"

            def _ensure_clean(self) -> None:
                pass

        obj = TestClass()
        result = obj.save()
        assert result == "saved"
        assert obj._dirty is False

    def test_needs_clean_calls_ensure_clean_when_dirty(self):
        """测试 needs_clean 脏时调用 _ensure_clean"""

        class TestClass:
            def __init__(self):
                self._dirty = True
                self.ensure_clean_called = False

            @DirtyStateMarker.needs_clean
            def read(self) -> str:
                return "data"

            def _ensure_clean(self) -> None:
                self.ensure_clean_called = True

        obj = TestClass()
        assert obj.ensure_clean_called is False
        result = obj.read()
        assert result == "data"
        assert obj.ensure_clean_called is True
        assert obj._dirty is False

    def test_needs_clean_skips_ensure_clean_when_clean(self):
        """测试 needs_clean 净时不调用 _ensure_clean"""

        class TestClass:
            def __init__(self):
                self._dirty = False
                self.ensure_clean_called = False

            @DirtyStateMarker.needs_clean
            def read(self) -> str:
                return "data"

            def _ensure_clean(self) -> None:
                self.ensure_clean_called = True

        obj = TestClass()
        result = obj.read()
        assert result == "data"
        assert obj.ensure_clean_called is False
        assert obj._dirty is False

    def test_decorator_preserves_function_metadata(self):
        """测试装饰器保留函数元数据"""

        class TestClass:
            def __init__(self):
                self._dirty = False

            @DirtyStateMarker.marks_dirty
            def my_method(self, x: int) -> int:
                """这是一个测试方法"""
                return x

            def _ensure_clean(self) -> None:
                pass

        obj = TestClass()
        assert obj.my_method.__name__ == "my_method"
        assert obj.my_method.__doc__ == "这是一个测试方法"

    def test_combined_workflow(self):
        """测试完整工作流: modify -> read -> save"""

        class DataStore:
            def __init__(self):
                self._dirty = False
                self._data = []
                self._sync_count = 0

            @DirtyStateMarker.marks_dirty
            def add(self, item: str) -> None:
                self._data.append(item)

            @DirtyStateMarker.needs_clean
            def get_all(self) -> list:
                return self._data.copy()

            @DirtyStateMarker.marks_clean
            def sync(self) -> None:
                self._sync_count += 1

            def _ensure_clean(self) -> None:
                self.sync()

        store = DataStore()
        assert store._dirty is False

        # 添加数据，标记为脏
        store.add("item1")
        assert store._dirty is True
        assert store._sync_count == 0

        # 读取时需要先清理
        result = store.get_all()
        assert result == ["item1"]
        assert store._dirty is False
        assert store._sync_count == 1  # _ensure_clean 被调用

        # 再次读取不需要清理
        store.add("item2")
        store.add("item3")
        assert store._dirty is True

        result = store.get_all()
        assert result == ["item1", "item2", "item3"]
        assert store._sync_count == 2  # 再次调用 _ensure_clean
