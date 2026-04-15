"""跨平台路径工具。"""
import os
from pathlib import Path, PureWindowsPath


def normalize_path(path: str, make_absolute: bool = True) -> str:
    """标准化路径，兼容 `~`、环境变量和 Windows/POSIX 分隔符。"""
    if path is None:
        return ''

    p = os.path.expandvars(os.path.expanduser(str(path).strip()))

    # Linux/macOS 下若收到 Windows 风格路径，先按 Windows 语义解析后转 POSIX
    if os.name != 'nt' and '\\' in p:
        p = PureWindowsPath(p).as_posix()

    path_obj = Path(p)
    if make_absolute:
        path_obj = path_obj.resolve(strict=False)
    else:
        path_obj = Path(os.path.normpath(str(path_obj)))
    return str(path_obj)
