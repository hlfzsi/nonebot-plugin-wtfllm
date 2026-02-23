import stat
import platform
import tarfile
import zipfile
from pathlib import Path

from ...utils import logger

from .utils import BaseQdrantDeployer


class WindowsQdrantDeployer(BaseQdrantDeployer):
    @property
    def executable_name(self) -> str:
        return "qdrant.exe"

    @property
    def asset_search_pattern(self) -> str:
        return "pc-windows-msvc.zip"

    def _extract_archive(self, archive_path: Path, extract_path: Path):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    def _set_permissions(self, exe_path: Path):
        pass
        
    
class UnixQdrantDeployer(BaseQdrantDeployer):
    @property
    def executable_name(self) -> str:
        return "qdrant"

    @property
    def asset_search_pattern(self) -> str:
        sys_name = platform.system().lower()  # linux 或 darwin (macOS)
        machine = platform.machine().lower()  # x86_64, aarch64, arm64

        if machine in ["arm64", "aarch64"]:
            arch = "aarch64"
        else:
            arch = "x86_64"

        if sys_name == "linux":
            return f"{arch}-unknown-linux-gnu.tar.gz"
        elif sys_name == "darwin":
            return f"{arch}-apple-darwin.tar.gz"
        
        raise RuntimeError(f"暂不支持的 Unix 变体: {sys_name}")

    def _extract_archive(self, archive_path: Path, extract_path: Path):
        logger.info(f"[*] 正在解压 {archive_path.name}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_path)

    def _set_permissions(self, exe_path: Path):
        if exe_path.exists():
            current_mode = exe_path.stat().st_mode
            exe_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            logger.info(f"[*] 已设置执行权限: {exe_path}")
            
    