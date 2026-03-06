import asyncio
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    List,
    Optional,
    Self,
    Type,
    TypeVar,
    overload,
    Sequence,
)

import httpx
from pydantic import BaseModel, Field

from .segments import (
    MEDIA_TYPES,
    AudioSegment,
    FileSegment,
    ForwardSegment,
    ImageSegment,
    MentionSegment,
    BaseSegment,
    MediaBaseSegment,
    Node,
    Segment,
    TextSegment,
)
from ...utils import get_http_client

if TYPE_CHECKING:
    from ..context import LLMContext

_T = TypeVar("_T", bound=BaseSegment)


class Message(BaseModel):
    """消息模型"""

    segments: List[Segment] = Field(default_factory=list)

    @classmethod
    def create(cls, segments: List[Segment] | Segment | None = None) -> Self:
        """创建消息实例"""
        if segments is None:
            return cls(segments=[])
        if isinstance(segments, list):
            return cls(segments=segments)
        return cls(segments=[segments])

    @property
    def created_at(self) -> int:
        """获取消息中最早片段的创建时间戳"""
        if self.segments:
            return min(seg.created_at for seg in self.segments)
        return 0

    async def ensure_local(
        self,
        client: httpx.AsyncClient | None = None,
        base_path_dir: Path | None = None,
    ) -> None:
        """确保消息中所有媒体片段的本地文件已下载（含嵌套 ForwardSegment）"""
        if client is None:
            client = get_http_client()

        coros = []
        self._collect_ensure_local_tasks(client, base_path_dir, coros)
        if coros:
            async with asyncio.TaskGroup() as tg:
                for coro in coros:
                    tg.create_task(coro)

    def _collect_ensure_local_tasks(
        self,
        client: httpx.AsyncClient,
        base_path_dir: Path | None,
        coros: list,
        prefix: str = "",
    ) -> None:
        """递归收集所有需要 ensure_local 的协程（含 ForwardSegment 嵌套）"""
        for i, seg in enumerate(self.segments):
            if isinstance(seg, MediaBaseSegment):
                coros.append(seg.ensure_local(client))
            elif isinstance(seg, ForwardSegment):
                for j, node in enumerate(seg.children):
                    node.content._collect_ensure_local_tasks(
                        client, base_path_dir, coros, prefix=f"{prefix}{i}_n{j}_"
                    )

    def to_llm_context(
        self, ctx: "LLMContext", message_id: str, memory_ref: int | None = None
    ) -> str:
        """转换为 LLM 上下文格式"""
        parts = [
            seg.to_llm_context(ctx, message_id, memory_ref) for seg in self.segments
        ]
        return " ".join(parts)

    def has(self, segment_type: MEDIA_TYPES | type) -> bool:
        """检查消息中是否包含指定类型的片段"""
        if isinstance(segment_type, str):
            return any(seg.type == segment_type for seg in self.segments)
        return any(isinstance(seg, segment_type) for seg in self.segments)

    @overload
    def get(self, segment_type: Type[_T]) -> Sequence[_T]: ...

    @overload
    def get(self, segment_type: MEDIA_TYPES) -> Sequence[BaseSegment]: ...

    def get(self, segment_type: MEDIA_TYPES | type) -> Sequence[BaseSegment]:
        """获取消息中所有指定类型的片段"""
        if isinstance(segment_type, str):
            return [seg for seg in self.segments if seg.type == segment_type]
        return [seg for seg in self.segments if isinstance(seg, segment_type)]

    def deep_has(self, segment_type: MEDIA_TYPES | type) -> bool:
        """递归检查消息中是否包含指定类型的片段（含嵌套 ForwardSegment 内部）"""
        for seg in self.segments:
            if isinstance(segment_type, str):
                if seg.type == segment_type:
                    return True
            elif isinstance(seg, segment_type):
                return True

            if isinstance(seg, ForwardSegment):
                for node in seg.children:
                    if node.content.deep_has(segment_type):
                        return True
        return False

    @overload
    def deep_get(self, segment_type: Type[_T]) -> Sequence[_T]: ...

    @overload
    def deep_get(self, segment_type: MEDIA_TYPES) -> Sequence[BaseSegment]: ...

    def deep_get(self, segment_type: MEDIA_TYPES | type) -> Sequence[BaseSegment]:
        """递归获取消息中所有指定类型的片段（含嵌套 ForwardSegment 内部）"""
        result: List[BaseSegment] = []
        for seg in self.segments:
            if isinstance(segment_type, str):
                if seg.type == segment_type:
                    result.append(seg)
            elif isinstance(seg, segment_type):
                result.append(seg)

            if isinstance(seg, ForwardSegment):
                for node in seg.children:
                    result.extend(node.content.deep_get(segment_type))
        return result

    def deep_find_and_update(self, target: BaseSegment, update: Dict) -> bool:
        """递归遍历（含嵌套 ForwardSegment），找到与 target 匹配的 segment 并更新。

        通过 BaseSegment.__eq__ (unique_key) 进行匹配。

        Args:
            target: 要查找的目标 segment
            update: model_copy(update=...) 的参数

        Returns:
            是否成功找到并更新
        """
        for i, seg in enumerate(self.segments):
            if seg == target:
                self.segments[i] = seg.model_copy(update=update)
                return True

            if isinstance(seg, ForwardSegment):
                for node in seg.children:
                    if node.content.deep_find_and_update(target, update):
                        return True
        return False

    def deep_find_node(self, target: BaseSegment) -> Optional[Node]:
        """递归查找包含目标 segment 的 Node。

        用于获取转发消息中某个 segment 的原始发送者等信息。
        如果 segment 不在任何 ForwardSegment 内部（即在顶层），返回 None。

        Args:
            target: 要查找的目标 segment

        Returns:
            包含该 segment 的 Node，如果在顶层则返回 None
        """
        for seg in self.segments:
            if isinstance(seg, ForwardSegment):
                for node in seg.children:
                    for child_seg in node.content.segments:
                        if child_seg == target:
                            return node

                    found = node.content.deep_find_node(target)
                    if found is not None:
                        return found
        return None

    def get_plain_text(self) -> str:
        """获取纯文本内容"""
        parts = []
        for seg in self.segments:
            if isinstance(seg, TextSegment):
                parts.append(seg.content)
        return "".join(parts)

    def append(self, segment: Segment) -> Self:
        """向消息中添加片段"""
        self.segments.append(segment)
        return self

    def text(self, content: str) -> Self:
        """添加文本片段"""
        return self.append(TextSegment(content=content))

    def mention(self, user_id: str) -> Self:
        """添加提及片段"""
        return self.append(MentionSegment(user_id=user_id))

    def image(self, url: str | None = None, local_path: Path | None = None) -> Self:
        """添加图片片段"""
        return self.append(ImageSegment(url=url, local_path=local_path))

    def file(
        self, filename: str, url: str | None = None, local_path: Path | None = None
    ) -> Self:
        """添加文件片段"""
        return self.append(
            FileSegment(filename=filename, url=url, local_path=local_path)
        )

    def audio(self, url: str | None = None, local_path: Path | None = None) -> Self:
        """添加音频片段"""
        return self.append(AudioSegment(url=url, local_path=local_path))

    def iter(self) -> Generator[BaseSegment, None, None]:
        """迭代消息片段"""
        yield from self.segments

    def __add__(self, other: "Message" | List[Segment] | Segment) -> "Message":
        new_message = Message.create(self.segments.copy())
        if isinstance(other, Message):
            new_message.segments.extend(other.segments)
        elif isinstance(other, list):
            new_message.segments.extend(other)
        else:
            new_message.segments.append(other)
        return new_message

    def __iadd__(self, other: "Message" | List[Segment] | Segment) -> Self:
        if isinstance(other, Message):
            self.segments.extend(other.segments)
        elif isinstance(other, list):
            self.segments.extend(other)
        else:
            self.segments.append(other)
        return self

    def __len__(self) -> int:
        """返回消息中叶子片段的总数（不计ForwardSegment 自身）"""
        count = 0
        for seg in self.segments:
            if isinstance(seg, ForwardSegment):
                for node in seg.children:
                    count += len(node.content)
            else:
                count += 1
        return count

    @property
    def message_count(self) -> int:
        """返回消息树中 Message 的总数"""
        count = 1
        for seg in self.segments:
            if isinstance(seg, ForwardSegment):
                for node in seg.children:
                    count += node.content.message_count
        return count
