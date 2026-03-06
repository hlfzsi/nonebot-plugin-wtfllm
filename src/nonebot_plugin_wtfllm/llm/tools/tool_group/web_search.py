import copy
import asyncio
from urllib.parse import urlparse

import orjson
from cachetools import TTLCache
from ddgs.ddgs import DDGS
from ddgs.exceptions import DDGSException
from trafilatura import fetch_url, extract
from trafilatura.settings import DEFAULT_CONFIG
from fake_useragent import UserAgent

from .base import ToolGroupMeta
from .utils import reschedule_deadline
from ...deps import Context
from ....utils import logger, APP_CONFIG

web_search_tool_group = ToolGroupMeta(
    name="WebSearch",
    description="用于网络搜索的工具箱，支持网页读取和搜索",
)

_ddgs = (
    DDGS(proxy=APP_CONFIG.web_search_proxy) if APP_CONFIG.web_search_proxy else DDGS()
)
_ua = UserAgent()
_region = "cn-zh"

_search_cache: TTLCache = TTLCache(maxsize=1024, ttl=24 * 3600)


@web_search_tool_group.tool(cost=2)
async def web_search(ctx: Context, query: str) -> str:
    """
    通过关键词搜索互联网新闻。用于获取实时信息、新闻或基础事实。
    返回包含标题、链接和摘要的 JSON 列表。

    Args:
        query (str): 搜索关键词, 应当使用英文
    """
    logger.debug(f"Performing web search for query: {query}")
    if query in _search_cache:
        logger.debug("Search result found in cache.")
        return _search_cache[query]
    reschedule_deadline(ctx, 40)

    search_task = asyncio.create_task(
        asyncio.to_thread(
            _ddgs.text,
            query,
            region=_region,
            max_results=5,
        )
    )

    try:
        results = await asyncio.wait_for(search_task, timeout=25)
    except (DDGSException, asyncio.TimeoutError) as e:
        logger.error(f"Web search failed: {e}")
        return f"搜索时出错 {e}，无法获取结果。"
    if not results:
        return "未找到结果"

    results_json = orjson.dumps(results, option=orjson.OPT_INDENT_2).decode("utf-8")
    _search_cache[query] = results_json
    logger.debug(f"Search results: {results_json}")

    return results_json


@web_search_tool_group.tool(cost=2)
async def web_fetch(ctx: Context, url_or_href: str) -> str:
    """
    访问指定 URL 并提取网页的正文内容
    当搜索结果的摘要不足以回答问题，需要阅读详情时使用。

    Args:
        url_or_href (str): 目标网址或链接。
    """
    reschedule_deadline(ctx, 20)
    result = urlparse(url_or_href)
    # 防止 file:// 等协议的 URL 注入
    if not all([result.scheme in ("http", "https"), result.netloc]):
        return "无效的 URL，请提供以 http:// 或 https:// 开头的有效链接。"

    _config = copy.deepcopy(DEFAULT_CONFIG)
    _config.set("DEFAULT", "USER_AGENTS", _ua.random)
    logger.debug(f"Fetching and extracting content from URL: {url_or_href}")
    download_task = asyncio.create_task(
        asyncio.to_thread(fetch_url, url_or_href, config=_config)
    )
    try:
        downloaded = await asyncio.wait_for(download_task, timeout=15)
    except asyncio.TimeoutError:
        return "抓取该网址内容超时。"

    if not downloaded:
        return "无法抓取该网址的内容，可能该网址不可访问。"
    markdown_content = await asyncio.to_thread(
        extract,
        downloaded,
        url=url_or_href,
        output_format="markdown",
        fast=True,
        deduplicate=True,
        with_metadata=True,
        favor_precision=True,
        include_links=False,
    )
    return markdown_content or "无法提取该网页的内容。"
