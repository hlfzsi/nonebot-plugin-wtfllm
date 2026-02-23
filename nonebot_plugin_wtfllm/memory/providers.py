from collections import defaultdict
from itertools import count
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar

from ._types import EntityType

if TYPE_CHECKING:
    from .content.segments import BaseSegment
    from .items import MemoryItem
    from .items.core_memory import CoreMemory
    from .items.knowledge_base import KnowledgeEntry

T = TypeVar("T", bound="BaseSegment")


class RefProvider:
    """引用ID提供器

    提供唯一引用ID:
    - 记忆项: [1], [2], [3], ...
    - 核心记忆: CM:1, CM:2, ...
    - 知识库: KB:1, KB:2, ...
    - 图片: IMG:1, IMG:2, ...
    - 文件: FILE:1, FILE:2, ...
    - 音频: AUDIO:1, AUDIO:2, ...
    """

    MEDIA_MAPPING = {
        "image": "IMG",
        "video": "VIDEO",
        "file": "FILE",
        "audio": "AUDIO",
    }
    REVERSED_MEDIA_MAPPING = {v: k for k, v in MEDIA_MAPPING.items()}

    def __init__(self):
        self._media_counter: Dict[str, count] = defaultdict(
            lambda: count(1)
        )  # Dict[BaseSegment.type , count]
        self._media_registry: Dict[str, Dict[str, "BaseSegment"]] = defaultdict(
            dict
        )  # Dict[BaseSegment.type, Dict[ref_str, BaseSegment]]

        # 按记忆项引用ID索引媒体
        self._media_by_memory_ref: Dict[int, List["BaseSegment"]] = defaultdict(list)

        # 媒体唯一键到引用号的映射
        self._media_key_to_ref: Dict[str, str] = {}

        # 记忆项计数器
        self._memory_counter = count(1)
        # 记忆项注册表
        self._item_id_to_ref: Dict[str, int] = {}  # item_id -> ref (数字)
        self._ref_to_item: Dict[int, "MemoryItem"] = {}  # ref -> item

        # 核心记忆计数器
        self._core_memory_counter = count(1)
        # 核心记忆注册表
        self._core_memory_id_to_ref: Dict[str, str] = {}  # storage_id -> ref (如 'CM:1')
        self._ref_to_core_memory: Dict[str, "CoreMemory"] = {}  # ref -> CoreMemory

        # 知识库计数器
        self._knowledge_counter = count(1)
        # 知识库注册表
        self._knowledge_id_to_ref: Dict[str, str] = {}  # storage_id -> ref (如 'KB:1')
        self._ref_to_knowledge: Dict[str, "KnowledgeEntry"] = {}  # ref -> KnowledgeEntry

    def reset(self) -> None:
        """重置所有引用计数器和注册表"""
        self.__init__()

    def next_memory_ref(self, item: "MemoryItem") -> int:
        """分配下一个记忆项引用号"""
        if item.message_id in self._item_id_to_ref:
            return self._item_id_to_ref[item.message_id]

        ref = next(self._memory_counter)
        self._item_id_to_ref[item.message_id] = ref
        self._ref_to_item[ref] = item
        return ref

    def get_ref_by_item_id(self, item_id: str) -> Optional[int]:
        """通过记忆项ID获取引用号"""
        return self._item_id_to_ref.get(item_id)

    def get_item_by_ref(self, ref: int) -> Optional["MemoryItem"]:
        """通过引用号获取记忆项ID"""
        return self._ref_to_item.get(ref)

    def next_core_memory_ref(self, memory: "CoreMemory") -> str:
        """分配下一个核心记忆引用号（返回如 'CM:1', 'CM:2'）"""
        if memory.storage_id in self._core_memory_id_to_ref:
            return self._core_memory_id_to_ref[memory.storage_id]

        ref_num = next(self._core_memory_counter)
        ref = f"CM:{ref_num}"
        self._core_memory_id_to_ref[memory.storage_id] = ref
        self._ref_to_core_memory[ref] = memory
        return ref

    def get_core_memory_ref_by_id(self, storage_id: str) -> Optional[str]:
        """通过核心记忆ID获取引用号"""
        return self._core_memory_id_to_ref.get(storage_id)

    def get_core_memory_by_ref(self, ref: str) -> Optional["CoreMemory"]:
        """通过引用号获取核心记忆"""
        return self._ref_to_core_memory.get(ref)

    def next_knowledge_ref(self, entry: "KnowledgeEntry") -> str:
        """分配下一个知识库引用号（返回如 'KB:1', 'KB:2'）"""
        if entry.storage_id in self._knowledge_id_to_ref:
            return self._knowledge_id_to_ref[entry.storage_id]

        ref_num = next(self._knowledge_counter)
        ref = f"KB:{ref_num}"
        self._knowledge_id_to_ref[entry.storage_id] = ref
        self._ref_to_knowledge[ref] = entry
        return ref

    def get_knowledge_ref_by_id(self, storage_id: str) -> Optional[str]:
        """通过知识条目ID获取引用号"""
        return self._knowledge_id_to_ref.get(storage_id)

    def get_knowledge_by_ref(self, ref: str) -> Optional["KnowledgeEntry"]:
        """通过引用号获取知识条目"""
        return self._ref_to_knowledge.get(ref)

    def next_media_ref(
        self, segment: "BaseSegment", memory_ref: int | None = None
    ) -> str:
        """分配下一个媒体引用号

        Args:
            segment: 媒体片段
            memory_ref: 所属记忆项的引用ID，传入时会同时注册到记忆项索引
        """
        if segment.unique_key in self._media_key_to_ref:
            return self._media_key_to_ref[segment.unique_key]

        media_type = segment.type
        ref = next(self._media_counter[media_type])
        ref_str = f"{self.MEDIA_MAPPING.get(media_type, media_type)}:{ref}"
        self._media_registry[media_type][ref_str] = segment
        self._media_key_to_ref[segment.unique_key] = ref_str

        # 注册到记忆项索引
        if memory_ref is not None:
            self._media_by_memory_ref[memory_ref].append(segment)

        return ref_str

    def get_media_by_ref(self, ref: str) -> Optional["BaseSegment"]:
        if ":" not in ref:
            raise ValueError("Invalid media ref format")
        media_type_prefix = ref.split(":")[0]
        media_type = self.REVERSED_MEDIA_MAPPING.get(
            media_type_prefix, media_type_prefix
        )
        return self._media_registry[media_type].get(ref)

    def get_media_typed(self, ref: str, expected_type: Type[T]) -> Optional[T]:
        segment = self.get_media_by_ref(ref)
        if isinstance(segment, expected_type):
            return segment

    @property
    def total_memories(self) -> int:
        """记忆项总数"""
        return len(self._item_id_to_ref)

    @property
    def total_images(self) -> int:
        """图片总数"""
        return len(self._media_registry["image"])

    def get_media_by_memory_ref(self, memory_ref: int) -> List["BaseSegment"]:
        """通过记忆项引用ID获取所有媒体片段

        Args:
            memory_ref: 记忆项引用ID (如 1, 2, 3)

        Returns:
            该记忆项包含的所有媒体片段列表，无媒体返回空列表
        """
        return self._media_by_memory_ref.get(memory_ref, [])

    def get_media_by_memory_ref_typed(
        self, memory_ref: int, expected_type: Type[T]
    ) -> List[T]:
        """通过记忆项引用ID获取指定类型的媒体片段

        Args:
            memory_ref: 记忆项引用ID
            expected_type: 期望的媒体类型 (如 ImageSegment)

        Returns:
            符合类型的媒体片段列表
        """
        return [
            seg
            for seg in self._media_by_memory_ref.get(memory_ref, [])
            if isinstance(seg, expected_type)
        ]


class AliasProvider:
    """实体别名提供器

    为实体生成可读别名:
    - 用户: User_1, User_2, ...
    - 群组: Group_1, Group_2, ...
    - 智能体: Agent_1, Agent_2, ...
    """

    def __init__(self):
        self._entity_to_alias: Dict[str, str] = {}
        self._alias_to_entity: Dict[str, str] = {}

        self._counters: Dict[str, count] = defaultdict(lambda: count(1))

    def get_or_create_alias(
        self, entity_id: str, entity_type: EntityType = "User"
    ) -> str:
        """获取或创建实体别名"""
        if entity_id in self._entity_to_alias:
            return self._entity_to_alias[entity_id]

        num = next(self._counters[entity_type])
        alias = f"{entity_type}_{num}"

        self._entity_to_alias[entity_id] = alias
        self._alias_to_entity[alias] = entity_id
        return alias

    def set_alias(self, entity_id: str, alias: str) -> None:
        """
        设置实体别名
        
        应当仅在初始化或风险可控的情况下使用此方法。

        Args:
            entity_id (str): 实体ID
            alias (str): 别名
        """
        if entity_id in self._entity_to_alias:
            old_alias = self._entity_to_alias[entity_id]
            del self._alias_to_entity[old_alias]

        self._entity_to_alias[entity_id] = alias
        self._alias_to_entity[alias] = entity_id

    def update_aliases(self, mapping: Dict[str, str]) -> None:
        """
        批量设置实体别名

        应当仅在初始化或风险可控的情况下使用此方法。

        Args:
            mapping (Dict[str, str]): 实体ID到别名的映射字典
        """
        if not mapping:
            return

        common_ids = self._entity_to_alias.keys() & mapping.keys()

        for eid in common_ids:
            old_alias = self._entity_to_alias[eid]
            if old_alias != mapping[eid]:
                self._alias_to_entity.pop(old_alias, None)

        self._entity_to_alias.update(mapping)
        self._alias_to_entity.update({v: k for k, v in mapping.items()})

    def get_alias(self, entity_id: str) -> Optional[str]:
        """获取实体的别名"""
        return self._entity_to_alias.get(entity_id)

    def resolve_alias(self, alias: str) -> Optional[str]:
        """解析别名为实体ID"""
        return self._alias_to_entity.get(alias)

    # 便捷

    def register_user(self, user_id: str) -> str:
        """注册用户并返回别名"""
        return self.get_or_create_alias(user_id, "User")

    def register_group(self, group_id: str) -> str:
        """注册群组并返回别名"""
        return self.get_or_create_alias(group_id, "Group")

    def register_agent(self, agent_id: str) -> str:
        """注册智能体并返回别名"""
        return self.get_or_create_alias(agent_id, "Agent")

    @property
    def alias_map(self) -> Dict[str, str]:
        """返回实体ID到别名的映射副本"""
        return dict(self._entity_to_alias)

    @property
    def reverse_map(self) -> Dict[str, str]:
        """返回别名到实体ID的映射副本"""
        return dict(self._alias_to_entity)
