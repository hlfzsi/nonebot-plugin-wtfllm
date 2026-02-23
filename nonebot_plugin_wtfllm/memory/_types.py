__all__ = ["EntityType", "ENTITY_TYPES", "ID_PATTERN"]


from typing import Literal, get_args
import re

EntityType = Literal["User", "Group", "Agent"]

# 从 Literal 类型自动提取所有实体类型
ENTITY_TYPES: tuple[str, ...] = get_args(EntityType)


ID_PATTERN = re.compile(r"\{\{(.+?)\}\}")
