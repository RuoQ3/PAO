"""
errors.py — Aspen Plus 驱动层异常层级。

所有异常均继承自 AspenError，调用方可以：
- 捕获 AspenError 处理所有驱动层错误
- 捕获具体子类处理特定场景

异常层级
---------
AspenError
├── AspenConnectionError      COM 连接建立/断开失败
│   └── AspenTypeLibraryError  type library 加载或 HAP 常量不完整
├── AspenFileError            仿真文件操作失败
├── AspenRunError             仿真运行控制失败
│   └── AspenRunTimeoutError   仿真超时
└── AspenNodeError            树节点访问或读写失败
    └── AspenNodeValueError    节点值类型转换失败
"""
from __future__ import annotations


class AspenError(Exception):
    """Aspen Plus 驱动层所有异常的基类。"""


# ---------------------------------------------------------------------------
# 连接层
# ---------------------------------------------------------------------------

class AspenConnectionError(AspenError):
    """COM 连接无法建立或已断开时抛出。"""


class AspenTypeLibraryError(AspenConnectionError):
    """
    Aspen Plus type library（happ.tlb）加载失败，
    或 EnsureDispatch 成功但 HAP 常量不完整时抛出。

    通常在 require_type_library=True 且常量加载失败时由
    AspenDriver.connect() 抛出。
    """


# ---------------------------------------------------------------------------
# 文件层
# ---------------------------------------------------------------------------

class AspenFileError(AspenError):
    """仿真文件无法打开、保存或导出时抛出。"""


# ---------------------------------------------------------------------------
# 运行控制层
# ---------------------------------------------------------------------------

class AspenRunError(AspenError):
    """仿真运行失败时抛出（启动失败、引擎错误、强制停止失败等）。"""


class AspenRunTimeoutError(AspenRunError):
    """
    仿真运行超过指定超时时间时抛出。

    Attributes
    ----------
    timeout:
        超时限制（秒）。
    """

    def __init__(self, timeout: float) -> None:
        super().__init__(f"仿真超时（{timeout}s）。")
        self.timeout = timeout


# ---------------------------------------------------------------------------
# 节点层
# ---------------------------------------------------------------------------

class AspenNodeError(AspenError):
    """树节点不存在、Elements 枚举失败或属性读写失败时抛出。"""


class AspenNodeValueError(AspenNodeError):
    """
    节点值类型转换失败时抛出（如将字符串节点强制转换为 float）。

    Attributes
    ----------
    path:
        出错的节点路径。
    raw:
        读取到的原始值。
    target_type:
        期望转换的目标类型名称（如 "float"、"int"）。
    """

    def __init__(self, path: str, raw: object, target_type: str) -> None:
        super().__init__(
            f"节点 '{path}' 的值 {raw!r} 无法转换为 {target_type}。"
        )
        self.path = path
        self.raw = raw
        self.target_type = target_type
