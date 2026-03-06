import os
import socket
import asyncio
import shutil
import httpx
from threading import Lock
from pathlib import Path
from typing import List, Set, Tuple, Optional, Final
from abc import ABC, abstractmethod

from qdrant_client import AsyncQdrantClient
from ...utils import logger, APP_CONFIG

LOG_ID: Final[str] = "[QdrantDeployer]"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])


class BaseQdrantDeployer(ABC):
    def __init__(
        self,
        base_dir: str | Path = "./qdrant_env",
        model_dir: str | Path | None = None,
        http_port: Optional[int] = None,
        grpc_port: Optional[int] = None,
    ) -> None:
        self._is_running: bool = False
        self.base_dir: Path = Path(base_dir).resolve()
        self.bin_dir: Path = self.base_dir / "bin"
        self.data_dir: Path = self.base_dir / "data"
        self.http_port: int = http_port or _get_free_port()
        self.grpc_port: int = grpc_port or _get_free_port()
        self.process: Optional[asyncio.subprocess.Process] = None

        self.exe_path: Path = self.bin_dir / self.executable_name
        if model_dir is not None:
            self.model_cache_path = Path(model_dir).resolve()
            self.model_cache_path.mkdir(parents=True, exist_ok=True)
        else:
            self.model_cache_path = None

        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._client: Optional[AsyncQdrantClient] = None
        self._lock = Lock()

    @property
    @abstractmethod
    def executable_name(self) -> str:
        """返回平台对应的可执行文件名"""
        pass

    @property
    @abstractmethod
    def asset_search_pattern(self) -> str:
        """返回 GitHub Release 资产文件的匹配字符串"""
        pass

    @abstractmethod
    def _extract_archive(self, archive_path: Path, extract_path: Path) -> None:
        """处理不同格式的压缩包解压"""
        pass

    def _set_permissions(self, exe_path: Path) -> None:
        """针对 Unix 系统设置执行权限"""
        pass

    @property
    def http_url(self) -> str:
        return f"http://localhost:{self.http_port}"

    @property
    def client(self) -> AsyncQdrantClient:
        if not self._is_running:
            raise RuntimeError(f"{LOG_ID} Qdrant 服务未运行，无法获取客户端实例。")

        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = AsyncQdrantClient(
                        url=self.http_url,
                        port=self.http_port,
                        grpc_port=self.grpc_port,
                        prefer_grpc=True,
                    )
        return self._client

    async def _get_latest_release_info(self) -> Tuple[str, str, str]:
        api_url = "https://api.github.com/repos/qdrant/qdrant/releases/latest"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
            tag: str = data["tag_name"]

            for asset in data["assets"]:
                if self.asset_search_pattern in asset["name"]:
                    return tag, asset["browser_download_url"], asset["name"]

        raise RuntimeError(
            f"{LOG_ID} 未能找到适合当前平台的二进制包 ({self.asset_search_pattern})"
        )

    async def _download_worker(
        self, client: httpx.AsyncClient, url: str, temp_path: Path
    ) -> Path:
        """具体的下载执行流"""
        try:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
            return temp_path
        except (httpx.HTTPError, OSError) as e:
            if temp_path.exists():
                os.remove(temp_path)
            raise e

    async def download_if_needed(
        self, proxy_prefix: str = "https://gh.jasonzeng.dev/"
    ) -> None:
        if self.exe_path.exists():
            logger.info(f"{LOG_ID} 已检测到已安装的 Qdrant，跳过下载步骤。")
            return

        tag, download_url, asset_name = await self._get_latest_release_info()

        version_marker = self.bin_dir / f"version_{tag}.txt"

        logger.info(f"{LOG_ID} 发现版本 {tag}，正在准备下载...")

        self.bin_dir.mkdir(parents=True, exist_ok=True)

        archive_path: Path = self.bin_dir / asset_name
        urls: List[str] = [f"{proxy_prefix}{download_url}", download_url]

        tasks: Set[asyncio.Task[Path]] = set()
        temp_paths: List[Path] = []
        winner_path: Optional[Path] = None

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
                for i, url in enumerate(urls):
                    t_path = self.bin_dir / f"{asset_name}.tmp{i}"
                    temp_paths.append(t_path)
                    tasks.add(
                        asyncio.create_task(self._download_worker(client, url, t_path))
                    )

                while tasks:
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        tasks.remove(task)
                        try:
                            winner_path = await task
                            for p_task in pending:
                                p_task.cancel()
                            remaining_tasks = list(pending)
                            if remaining_tasks:
                                await asyncio.gather(
                                    *remaining_tasks, return_exceptions=True
                                )
                            tasks.clear()
                            break
                        except (httpx.HTTPError, OSError) as e:
                            logger.warning(
                                f"{LOG_ID} 下载源故障，尝试备用源... 错误: {e}"
                            )

                if not winner_path or not winner_path.exists():
                    raise RuntimeError(f"{LOG_ID} 所有下载源均请求失败。")

                if archive_path.exists():
                    os.remove(archive_path)
                shutil.move(str(winner_path), str(archive_path))

        finally:
            for tp in temp_paths:
                try:
                    if tp.exists():
                        os.remove(tp)
                except OSError:
                    pass

        logger.info(f"{LOG_ID} 解压资源中...")
        try:
            self._extract_archive(archive_path, self.bin_dir)
            self._set_permissions(self.exe_path)

            for old_marker in self.bin_dir.glob("version_*.txt"):
                os.remove(old_marker)

            version_marker.touch()
            logger.success(f"{LOG_ID} Qdrant {tag} 部署完成")
        finally:
            if archive_path.exists():
                os.remove(archive_path)

    async def start(self, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                if self.http_port is None or attempt > 0:
                    self.http_port = _get_free_port()
                if self.grpc_port is None or attempt > 0:
                    while True:
                        self.grpc_port = _get_free_port()
                        if self.grpc_port != self.http_port:
                            break
                env = os.environ.copy()
                env["QDRANT__SERVICE__HTTP_PORT"] = str(self.http_port)
                env["QDRANT__SERVICE__GRPC_PORT"] = str(self.grpc_port)
                env["QDRANT__STORAGE__STORAGE_PATH"] = str(self.data_dir)

                logger.info(
                    f"{LOG_ID} 正在启动服务 (HTTP端口: {self.http_port}, gRPC端口: {self.grpc_port})..."
                )
                creation_flags = 0
                if os.name == "nt":
                    creation_flags = 0x08000000  # CREATE_NO_WINDOW

                self.process = await asyncio.create_subprocess_exec(
                    str(self.exe_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=creation_flags,
                )

                async with httpx.AsyncClient(timeout=1) as client:
                    for _ in range(120):
                        try:
                            res = await client.get(f"{self.http_url}/healthz")
                            if res.status_code == 200:
                                self._is_running = True
                                logger.success(f"{LOG_ID} 服务已就绪: {self.http_url}")
                                return self.http_url
                        except httpx.RequestError:
                            await asyncio.sleep(1)
            except (OSError, RuntimeError):
                continue

        await self.stop()
        raise RuntimeError(f"{LOG_ID} 启动超时。")

    async def stop(self) -> None:
        self._is_running = False

        if self._client:
            await self._client.close()
            self._client = None

        if self.process:
            logger.info(f"{LOG_ID} 正在关闭进程...")
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except (OSError, ProcessLookupError):
                if self.process:
                    self.process.kill()
            self.process = None
            logger.success(f"{LOG_ID} 服务已安全关闭。")

    async def post_start(self) -> None:
        client = self.client

        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                asyncio.to_thread(
                    client.set_model,
                    APP_CONFIG.embedding_model_name,
                    cache_dir=str(self.model_cache_path)
                    if self.model_cache_path
                    else None,
                )
            )
            tg.create_task(
                asyncio.to_thread(
                    client.set_sparse_model,
                    APP_CONFIG.sparse_model_name,
                    cache_dir=str(self.model_cache_path)
                    if self.model_cache_path
                    else None,
                )
            )
