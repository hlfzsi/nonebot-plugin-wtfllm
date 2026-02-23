from dataclasses import dataclass, field

from .providers import AliasProvider, RefProvider


@dataclass
class LLMContext:
    """LLM 上下文容器

    持有 Provider 实例
    """

    ref_provider: RefProvider
    alias_provider: AliasProvider
    condense: bool = field(default=True)

    @classmethod
    def create(cls, condense: bool = True) -> "LLMContext":
        return cls(
            ref_provider=RefProvider(),
            alias_provider=AliasProvider(),
            condense=condense,
        )

    def set_condense(self, condense: bool) -> None:
        """设置是否启用压缩"""
        self.condense = condense

    def copy(self, share_providers: bool = True) -> "LLMContext":
        """创建当前上下文的副本

        Args:
            share_providers (bool, optional): 是否共享 Provider 实例. 默认为 True.

        Returns:
            LLMContext: 上下文副本
        """
        return LLMContext(
            ref_provider=self.ref_provider if share_providers else RefProvider(),
            alias_provider=self.alias_provider if share_providers else AliasProvider(),
            condense=self.condense,
        )
