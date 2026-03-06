"""记忆系统工具函数"""

from functools import wraps
from typing import Callable, Protocol, TypeVar, ParamSpec, Concatenate

R = TypeVar("R")
P = ParamSpec("P")
T = TypeVar("T", bound="Dirtyable")


class Dirtyable(Protocol):
    """支持脏状态标记的协议"""

    _dirty: bool

    def _ensure_clean(self) -> None: ...


class DirtyStateMarker:
    """
    脏状态标记装饰器集合。

    用于标记方法对对象脏状态的影响：
    - marks_dirty: 方法执行后标记为脏
    - marks_clean: 方法执行后标记为干净
    - needs_clean: 方法执行前确保状态干净
    """

    @staticmethod
    def marks_dirty(
        method: Callable[Concatenate[T, P], R],
    ) -> Callable[Concatenate[T, P], R]:
        @wraps(method)
        def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
            result = method(self, *args, **kwargs)
            self._dirty = True
            return result

        return wrapper

    @staticmethod
    def marks_clean(
        method: Callable[Concatenate[T, P], R],
    ) -> Callable[Concatenate[T, P], R]:
        """标记方法会净化状态（执行后 _dirty = False）"""

        @wraps(method)
        def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
            result = method(self, *args, **kwargs)
            self._dirty = False
            return result

        return wrapper

    @staticmethod
    def needs_clean(
        method: Callable[Concatenate[T, P], R],
    ) -> Callable[Concatenate[T, P], R]:
        """要求状态为净化后才能调用，否则自动调用 _ensure_clean()"""

        @wraps(method)
        def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
            if self._dirty:
                self._ensure_clean()
                self._dirty = False
            return method(self, *args, **kwargs)

        return wrapper


def generate_alpha_index(n: int) -> str:
    """
    生成字母索引

    Args:
        n: 1-based 索引值

    Returns:
        字母索引字符串: 1->A, 2->B, ..., 26->Z, 27->AA, 28->AB...

    Examples:
        >>> generate_alpha_index(1)
        'A'
        >>> generate_alpha_index(26)
        'Z'
        >>> generate_alpha_index(27)
        'AA'
        >>> generate_alpha_index(28)
        'AB'
    """
    if n <= 0:
        raise ValueError("Index must be positive")

    result = []
    while n > 0:
        n -= 1
        result.append(chr(ord("A") + (n % 26)))
        n //= 26
    return "".join(reversed(result))


def parse_alpha_index(s: str) -> int:
    """
    解析字母索引为数字

    Args:
        s: 字母索引字符串（如 'A', 'Z', 'AA', 'AB'）

    Returns:
        1-based 索引值

    Examples:
        >>> parse_alpha_index('A')
        1
        >>> parse_alpha_index('Z')
        26
        >>> parse_alpha_index('AA')
        27
        >>> parse_alpha_index('AB')
        28
    """
    if not s or not s.isalpha():
        raise ValueError("Invalid alpha index")

    s = s.upper()
    result = 0
    for char in s:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result
