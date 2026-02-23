import os
from typing import List
from nonebot import get_plugin_config
from pydantic import BaseModel, Field, ValidationError


class ModelConfig(BaseModel):
    name: str = Field(..., description="模型名称")
    base_url: str = Field(..., description="模型API基础URL")
    api_key: str = Field(..., description="模型API密钥")
    extra_body: dict = Field(..., description="模型API额外请求参数")


class Config(BaseModel):
    superusers: List[str] = Field(
        default_factory=list,
        description="管理员用户ID列表",
    )

    ignore_reference: bool = Field(
        default=True,
        description="是否忽略合并引用消息",
    )

    media_lifecycle_days: int = Field(
        default=30,
        description="媒体文件生命周期，单位天，超过该时间的媒体文件将被删除",
    )

    media_auto_unbind: bool = Field(
        default=True,
        description="是否自动解绑过期媒体文件，启用后过期媒体文件将被自动管理并删除",
    )

    agent_base_timeout_seconds: int = Field(
        default=45,
        description="LLM基础超时时间，单位秒",
    )

    tool_point_budget: int = Field(
        default=6,
        description="工具点数预算总量，0 表示不启用预算限制",
    )

    tool_point_default_cost: int = Field(
        default=1,
        description="未显式指定 cost 的工具的默认点数消耗",
    )

    web_search_proxy: str | None = Field(
        default=None,
        description="Web搜索工具的代理设置，设置为空则不使用代理",
    )

    message_track_time_minutes: int = Field(
        default=120,
        description="消息追踪时间窗口，单位分钟",
    )

    short_memory_time_minutes: int = Field(
        default=15,
        description="默认注入的短期记忆时间窗口，单位分钟。仅加载此时间窗口内的最近消息",
    )

    short_memory_max_count: int = Field(
        default=10,
        description="默认注入的短期记忆最大条数。与时间窗口取交集（同时满足）",
    )

    core_memory_max_tokens: int = Field(
        default=2048,
        description="单个会话核心记忆的最大 token 数，超过后自动压缩",
    )

    core_memory_compress_ratio: float = Field(
        default=0.6,
        description="压缩目标比例，压缩后核心记忆占最大值的比例",
    )

    tool_call_record_max_count: int = Field(
        default=1,
        description="工具调用记录的注入数量，设置为0表示不注入工具调用记录",
    )

    knowledge_base_max_results: int = Field(
        default=5,
        description="知识库注入时的最大检索结果数量",
    )

    knowledge_base_max_tokens: int = Field(
        default=1024,
        description="知识库在 prompt 中的最大 token 数",
    )

    memory_item_max_chars: int = Field(
        default=60,
        description="单条记忆中文本片段的最大字符数，超过时取头尾各半压缩展示",
    )

    huggingface_mirror_url: str = Field(
        default="https://hf-mirror.com",
        description="HuggingFace 镜像网址 , 设置为空则不使用镜像",
    )

    bot_name: str = Field(default="小W", description="机器人名称")
    llm_api_key: str = Field(..., description="LLM_API_KEY")
    llm_api_base_url: str = Field(
        ...,
        description="LLM_API_BASE_URL",
    )
    llm_model_name: str = Field(..., description="LLM 模型名称")
    llm_role_setting: str = Field(default="", description="LLM 角色设定")
    llm_extra_body: dict = Field(default_factory=dict, description="LLM 额外请求参数")
    llm_support_vision: bool = Field(
        default=True, description="主agent是否具备视觉理解能力"
    )
    llm_use_responses_api: bool = Field(
        default=False, description="是否使用Response API, 暂无证据显示更节约token"
    )

    compress_api_key: str | None = Field(
        default=None,
        description="核心记忆压缩模型API密钥，设置后启用独立模型压缩核心记忆",
    )
    compress_api_base_url: str | None = Field(
        default=None,
        description="核心记忆压缩模型API基础URL，设置后启用独立模型压缩核心记忆",
    )
    compress_model_name: str | None = Field(
        default=None, description="核心记忆压缩模型名称，设置后启用独立模型压缩核心记忆"
    )
    compress_extra_body: dict = Field(
        default_factory=dict, description="核心记忆压缩模型额外请求参数"
    )

    vision_model_name: str | None = Field(default=None, description="视觉理解模型名称")
    vision_model_base_url: str | None = Field(
        default=None, description="视觉理解模型基础URL"
    )
    vision_api_key: str | None = Field(default=None, description="视觉理解模型API密钥")
    vision_extra_body: dict = Field(
        default_factory=dict, description="视觉理解模型额外请求参数"
    )

    image_generation_model_name: str | None = Field(
        default=None, description="图像生成模型名称"
    )
    image_generation_model_base_url: str | None = Field(
        default=None, description="图像生成模型基础URL"
    )
    image_generation_api_key: str | None = Field(
        default=None, description="图像生成模型API密钥"
    )
    image_generation_extra_body: dict = Field(
        default_factory=dict, description="图像生成模型额外请求参数"
    )

    embedding_model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5", description="向量嵌入模型名称及前缀"
    )

    sparse_model_name: str = Field(
        default="Qdrant/bm25", description="稀疏向量模型名称及前缀"
    )

    def model_post_init(self, __context):
        if self.huggingface_mirror_url:
            os.environ["HF_ENDPOINT"] = self.huggingface_mirror_url

    @property
    def admin_users(self) -> List[str]:
        return self.superusers

    @property
    def main_agent_model_config(self) -> ModelConfig:
        try:
            return ModelConfig(
                name=self.llm_model_name,
                base_url=self.llm_api_base_url,
                api_key=self.llm_api_key,
                extra_body=self.llm_extra_body,
            )
        except ValidationError as e:
            raise ValueError(f"主代理模型配置无效: {e}")

    @property
    def compress_agent_model_config(self) -> ModelConfig:
        try:
            return ModelConfig(
                name=self.compress_model_name or self.llm_model_name,
                base_url=self.compress_api_base_url or self.llm_api_base_url,
                api_key=self.compress_api_key or self.llm_api_key,
                extra_body=self.compress_extra_body or self.llm_extra_body,
            )
        except ValidationError as e:
            raise ValueError(f"核心记忆压缩模型配置无效: {e}")

    @property
    def vision_model_config(self) -> ModelConfig | None:
        try:
            return ModelConfig(
                name=self.vision_model_name,  # pyright: ignore[reportArgumentType]
                base_url=self.vision_model_base_url,  # pyright: ignore[reportArgumentType]
                api_key=self.vision_api_key,  # pyright: ignore[reportArgumentType]
                extra_body=self.vision_extra_body,  # pyright: ignore[reportArgumentType]
            )
        except ValidationError:
            return None

    @property
    def image_generation_model_config(self) -> ModelConfig | None:
        try:
            return ModelConfig(
                name=self.image_generation_model_name,  # pyright: ignore[reportArgumentType]
                base_url=self.image_generation_model_base_url,  # pyright: ignore[reportArgumentType]
                api_key=self.image_generation_api_key,  # pyright: ignore[reportArgumentType]
                extra_body=self.image_generation_extra_body,  # pyright: ignore[reportArgumentType]
            )
        except ValidationError:
            return None


APP_CONFIG = get_plugin_config(Config)
