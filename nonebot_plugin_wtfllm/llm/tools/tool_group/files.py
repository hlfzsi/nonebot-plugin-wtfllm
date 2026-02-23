import os
import aiofiles

from .base import ToolGroupMeta
from ...deps import Context
from ....utils import APP_CONFIG


async def _perm(ctx: Context) -> bool:
    return ctx.deps.ids.user_id in APP_CONFIG.admin_users


file_tool_group = ToolGroupMeta(
    name="files", description="文件处理工具组，允许阅读本地文件内容", prem=_perm
)


@file_tool_group.tool
def list_files_in_directory(ctx: Context, directory_path: str = "./") -> str:
    """
    列出指定目录下的所有文件和子目录, 默认为当前目录

    Args:
        directory_path (str): 目录路径
    """
    try:
        items = os.listdir(directory_path)
        return "\n".join(items) if items else "目录为空。"
    except (OSError, PermissionError) as e:
        return f"无法访问目录 '{directory_path}'，错误信息: {str(e)}"


@file_tool_group.tool
async def read_local_file(ctx: Context, file_path: str) -> str:
    """
    读取本地文件内容

    Args:
        file_path (str): 本地文件路径
    """
    try:
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
            content = await f.read()
        return content
    except (OSError, UnicodeDecodeError) as e:
        return f"无法读取文件 '{file_path}'，错误信息: {str(e)}"


@file_tool_group.tool
async def search_file(ctx: Context, dir_path: str, keyword: str) -> str:
    """
    在指定目录下搜索包含关键词的文件

    Args:
        dir_path (str): 目录路径
        keyword (str): 搜索关键词
    """
    matched_files = []
    try:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    async with aiofiles.open(
                        file_path, mode="r", encoding="utf-8"
                    ) as f:
                        content = await f.read()
                        if keyword in content:
                            matched_files.append(file_path)
                except (OSError, UnicodeDecodeError):
                    continue
        return "\n".join(matched_files) if matched_files else "未找到匹配的文件。"
    except OSError as e:
        return f"无法访问目录 '{dir_path}'，错误信息: {str(e)}"
