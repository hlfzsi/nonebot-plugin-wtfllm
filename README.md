<div align="center">

# nonebot-plugin-wtfllm

_一个十分甚至有九分重量级的 Agent 实现_

![Python](https://img.shields.io/badge/Python-≥3.11-blue)
![NoneBot2](https://img.shields.io/badge/NoneBot2-≥2.4.4-green)
![License](https://img.shields.io/badge/License-AGPL--3.0-orange)

</div>

**nonebot-plugin-wtfllm** 是一个面向 NoneBot2 的 LLM Agent 插件。它为你的 Bot 提供多层记忆、多模态理解、工具自主调度等能力——开箱即用，配置即走。

- **三层记忆架构** —— 短期、核心、知识库，让你的 Bot 真正拥有"记性"
- **30+ 工具自主调度** —— 联网搜索、绘图、表情包管理，一句话搞定
- **类Skills设计**—— 足够省钱，单次请求总额约 5k token
- **开箱即用的多模态** —— 看图识图、文生图，能力随 API 自由扩展
- **定时消息与用户画像** —— 关注每一位用户，不遗漏任何约定
- **OpenAI 格式 API** —— 模型自由切换，不绑定任何供应商
- **完善的生命周期管理** —— 优雅启停，数据持久化，断电无忧

## 为什么是 WtfLLM

整个项目的缘起是一台挂机宝。

2025 年下半年的某天，意外以极低的价格入手了一台 8h16g 的挂机宝。出于不浪费的原则，我在上面部署了 Bot。装什么功能呢？第一个想到的就是 Bot 界原神——Agent！

经过紧张而激烈的插件挑选，我发现——没有一个完美符合预期。这很正常，恰好我也有些闲工夫，于是着手写一个自己的 Agent 插件。

---

WtfLLM 经历过不少阶段。

最初它叫 _smart-llm_，基于 Mem0 和 Instructor 构建。看看现在的依赖列表就知道，和当初已经完全不同了。smart-llm 基本完成时，我意识到那些基于 Mem0 的代码实在难以维护，效果也达不到预期：一方面，Mem0 的 token 消耗相当高——对于一个穷得叮当响、裤兜向里掏只有穿身上内裤的大学生来说，每条消息做实体提取的开销比吃饭还贵；另一方面，Mem0 的记忆检索并不适合聊天 Bot，它会丢失上下文语境。

当然，我尝试过关闭实体提取，至于最终效果……这里就不说了。

在此期间我一直在寻找更合适的方案。Letta（原 MemGPT）给我留下了深刻印象，有一段时间 Mem0 和 Letta 方案是并行开发的。遗憾的是，Letta 简直是 token 燃烧机——它像一辆大运，平等地创飞一切阻碍，只是油耗负担不起。况且，对于一个还没拿到 C2 驾照的大学生来说，开大运实在有些难度（笑）。

---

随后我决定推翻重来。有了 smart-llm 的底层积累，我觉得 WtfLLM 的开发应当迅速而容易。尽管事实并非如此，最终它还是被端上来了。WtfLLM 改用了 Pydantic AI 作为 Agent 框架，这个框架用着真的非常舒服。

开发期间，我陆续阅读了不少或老牌、或新兴、或知名、或默默无闻的 Agent 项目——MaiBot、AstrBot、Nekro Agent、Amrita……它们各有独到之处，让我连连感叹。不过由于设计目标不同，很多思路无法直接借鉴。对我启发最大的仍然是 Mem0 和 Letta，WtfLLM 的记忆架构里明显能看到它们的影子。

所以，_WtfLLM_ 这个名字？大概就是开发过程中说得最多的那句话吧。至于为什么不叫 Agent 而是 LLM——大概是因为还没有勇气比肩那些已经有名气的 Agent 项目，只敢先冠以 LLM 之名。

## 安装

> [!NOTE]
> 本插件尚未发布至 PyPI，目前请通过源码安装。

```bash
git clone https://github.com/hlfzsi/nonebot-plugin-wtfllm.git
cd nonebot-plugin-wtfllm
pip install .
```

在 NoneBot2 的 `pyproject.toml` 中加载插件：

```toml
[tool.nonebot]
plugins = ["nonebot_plugin_wtfllm"]
```

向量数据库使用 Qdrant，已内置**自动部署**，首次启动时会自动下载嵌入模型。

### 环境要求


| 项目 | 要求                                                        |
| ---- | ----------------------------------------------------------- |
| CPU  | 至少双核，推荐四核以上                                      |
| 网络 | 需要稳定连接，首次启动开销大                                |
| 磁盘 | 建议 ≥ 4 GB 可用                                          |
| 内存 | 建议 ≥ 500 MB 可用                                         |
| 系统 | `Windows >= 10.0.19041` / `glibc >= 2.28` / `macOS >= 14.0` |

> [!TIP]
> 国内网络环境下，Qdrant 二进制下载已内置 GitHub 代理加速；嵌入模型默认走 `hf-mirror.com` 镜像，可通过 `huggingface_mirror_url` 配置项自定义。

## 配置

在 NoneBot2 的 `.env` 文件中配置以下项目：

有点多，但真正要关注的只有**必填配置**和可选配置中的**独立模型**。

特别再次提醒，本项目的模型应当全部使用**OpenAI**接口。

### 必填配置


| 配置项             | 类型  | 说明             |
| ------------------ | ----- | ---------------- |
| `llm_api_key`      | `str` | LLM API 密钥     |
| `llm_api_base_url` | `str` | LLM API 基础 URL |
| `llm_model_name`   | `str` | LLM 模型名称     |

### 可选配置 — 基础


| 配置项                  | 类型        | 默认值  | 说明                      |
| ----------------------- | ----------- | ------- | ------------------------- |
| `bot_name`              | `str`       | `"小W"` | 机器人名称                |
| `llm_role_setting`      | `str`       | `""`    | LLM 角色设定              |
| `llm_extra_body`        | `dict`      | `{}`    | LLM 额外请求参数          |
| `llm_support_vision`    | `bool`      | `false` | 主 Agent 是否具备视觉能力 |
| `llm_use_responses_api` | `bool`      | `false` | 是否使用 Response API     |
| `superusers`            | `list[str]` | `[]`    | 管理员用户 ID 列表        |

### 可选配置 — 记忆系统


| 配置项                       | 类型    | 默认值 | 说明                                             |
| ---------------------------- | ------- | ------ | ------------------------------------------------ |
| `short_memory_max_count`     | `int`   | `15`   | 默认注入的短期记忆最大条数，与时间窗口取交集     |
| `core_memory_max_tokens`     | `int`   | `2048` | 单个会话核心记忆最大 token 数，超过后自动压缩    |
| `core_memory_compress_ratio` | `float` | `0.6`  | 核心记忆压缩目标比例                             |
| `memory_item_max_chars`      | `int`   | `60`   | 单条记忆文本最大字符数，超过时取头尾各半压缩展示 |
| `knowledge_base_max_results` | `int`   | `5`    | 知识库最大检索结果数                             |
| `knowledge_base_max_tokens`  | `int`   | `1024` | 知识库在 prompt 中的最大 token 数                |

### 可选配置 — Agent 行为


| 配置项                       | 类型  | 默认值 | 说明                       |
| ---------------------------- | ----- | ------ | -------------------------- |
| `agent_base_timeout_seconds` | `int` | `45`   | Agent 基础超时（秒）       |
| `tool_point_budget`          | `int` | `5`    | 工具点数预算，`0` 为不限制 |
| `message_track_time_minutes` | `int` | `120`  | 消息追踪窗口（分钟）       |
| `tool_call_record_max_count` | `int` | `1`    | 注入的工具调用记录数量     |

### 可选配置 — 独立模型

视觉理解、图像生成、记忆压缩可配置独立模型。未配置时回退到主模型或禁用对应功能。


| 配置项                                                                                                                         | 说明         |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| `vision_model_name` / `vision_model_base_url` / `vision_api_key` / `vision_extra_body`                                         | 视觉理解模型 |
| `image_generation_model_name` / `image_generation_model_base_url` / `image_generation_api_key` / `image_generation_extra_body` | 图像生成模型 |
| `compress_model_name` / `compress_api_base_url` / `compress_api_key` / `compress_extra_body`                                   | 记忆压缩模型 |

### 可选配置 — 其他


| 配置项                   | 类型            | 默认值                     | 说明                                                      |
| ------------------------ | --------------- | -------------------------- | --------------------------------------------------------- |
| `database_url`           | `str` or `None` | `None`                     | 数据库连接 URL，设置后使用指定数据库，否则使用默认 SQLite |
| `media_lifecycle_days`   | `int`           | `30`                       | 媒体文件生命周期（天）                                    |
| `media_auto_unbind`      | `bool`          | `true`                     | 自动解绑并清理过期媒体文件                                |
| `web_search_proxy`       | `str` or `None` | `None`                     | Web 搜索工具的代理设置，为空则不使用代理                  |
| `ignore_reference`       | `bool`          | `true`                     | 是否忽略合并引用消息                                      |
| `embedding_model_name`   | `str`           | `"BAAI/bge-small-zh-v1.5"` | 向量嵌入模型                                              |
| `sparse_model_name`      | `str`           | `"Qdrant/bm25"`            | 稀疏向量模型                                              |
| `huggingface_mirror_url` | `str`           | `"https://hf-mirror.com"`  | HuggingFace 镜像                                          |

## 使用

@Bot 与 Bot 对话。Bot 会自动管理记忆、调用工具并生成回复。在某些情况下，Bot 也可能响应未 @ 的消息，但这是一类伪主动策略。 我们正在积极探索和引入**不基于**LLM对上下文理解的主动触达能力。

### 管理指令

以下指令需要配置 `superusers` 后由管理员使用，均需添加命令前缀（默认 `/`）。


| 指令                            | 别名                       | 说明                                 |
| ------------------------------- | -------------------------- | ------------------------------------ |
| `/summary [数量] [关键词]`      | `/摘要`、`/记忆`           | 查询核心记忆，支持按关键词检索       |
| `/summary [数量] -g <群号>`     | —                         | 查询指定群的核心记忆                 |
| `/easyban user add <用户ID>`    | `/eb`、`/ban`              | 将用户加入黑名单                     |
| `/easyban user remove <用户ID>` | —                         | 将用户移出黑名单                     |
| `/easyban user list`            | —                         | 查看黑名单用户列表                   |
| `/easyban group add [群号]`     | —                         | 将群加入黑名单（群内使用可省略群号） |
| `/easyban group remove [群号]`  | —                         | 将群移出黑名单                       |
| `/easyban group list`           | —                         | 查看黑名单群列表                     |
| `/delete_media [-d 天数]`       | `/del`、`/delete`、`/删除` | 清理过期媒体文件                     |

### 工具一览

工具由 Agent 自主调度，无需用户手动触发。


| 工具组              | 说明                                                 | 工具数 | 单次消耗 | 激活方式           |
| ------------------- | ---------------------------------------------------- | -----: | -------- | ------------------ |
| **Core**            | 人设锚定、图片理解、记忆查询、历史消息加载等基础能力 |     10 | 0–2 pt  | 常驻               |
| **Chat**            | 发送中间消息、主动提问并等待回复                     |      2 | -1 pt    | 常驻               |
| **CoreMemory**      | 新增、更新、删除核心记忆                             |      3 | 0–1 pt  | 常驻               |
| **KnowledgeBase**   | 新增、更新、删除全局知识库条目                       |      3 | 0–1 pt  | 常驻               |
| **Memes**           | 保存、搜索、列出表情包                               |      3 | 0 pt     | 常驻               |
| **UserPersona**     | 获取与更新用户画像                                   |      2 | 0 pt     | 常驻               |
| **WebSearch**       | 关键词搜索、网页正文提取                             |      2 | 2 pt     | 常驻               |
| **ImageGeneration** | 文生图、图生图、多图合成                             |      3 | 4 pt     | 需配置图像生成模型 |
| **ScheduleMessage** | 定时消息、定时 Agent 任务创建与管理                  |      4 | 0-3 pt   | 常驻               |

> **工具点数预算**：每轮对话有 `tool_point_budget`（默认 5）个点数。Agent 每调用一个工具就扣除对应 pt；当预算不足时会收到警告并尽快给出最终回复。设为 `0` 可关闭预算限制。

由于不确定有没有人会基于本插件二次开发，当前尚未开放自定义工具的接口，如有需要欢迎在 Issue 中提出。

## 关于测试

> [!WARNING]
> `tests/` 目录下的全部测试用例由 **Claude Opus 4.5 / 4.6** 编写，未经人工逐行审阅。
> 它们对开发过程帮助巨大，但如果你发现了测试本身的问题，欢迎提 Issue 或 PR。
>
> ~~虽然读这些代码会折寿~~

## 鸣谢

- [Letta](https://github.com/letta-ai/letta) & [Mem0](https://github.com/mem0ai/mem0) — 记忆系统设计的灵感来源
- [NoneBot2](https://github.com/nonebot/nonebot2) — 跨平台 Bot 框架
- [nonebot-plugin-alconna](https://github.com/nonebot/plugin-alconna) — 消息处理与指令解析的基石
- [pillowmd](https://github.com/Monody-S/CustomMarkdownImage) — Markdown 渲染为图片的核心依赖
- [Pydantic AI](https://github.com/pydantic/pydantic-ai) — Agent 框架
- [Qdrant](https://github.com/qdrant/qdrant) — 向量检索引擎

## 许可证

[AGPL-3.0](LICENSE)

---

<div align="center">

_如果这个项目对你有帮助，欢迎点个 Star_

</div>
