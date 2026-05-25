"""
driver.py — Aspen Plus COM 底层接口封装。

职责：连接生命周期管理、文件操作、仿真控制、节点读写。
不包含任何业务逻辑，业务逻辑由 runner.py 及上层模块负责。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pythoncom
import win32com.client

from .errors import (
    AspenConnectionError,
    AspenFileError,
    AspenNodeError,
    AspenRunError,
    AspenRunTimeoutError,
    AspenTypeLibraryError,
)
from .node import HAP_NAMES

_log = logging.getLogger(__name__)


class AspenDriver:
    """对单个 Aspen Plus COM 实例的低级封装。"""

    DEFAULT_PROG_ID = "Apwn.Document"
    _RUN_TIMEOUT = 300      # 默认仿真超时（秒）
    _POLL_INTERVAL = 1.0    # 轮询引擎状态的间隔（秒）

    def __init__(
        self,
        visible: bool = False,
        suppress_dialogs: bool = True,
        prog_id: str = DEFAULT_PROG_ID,
        require_type_library: bool = False,
    ) -> None:
        self._app: Any | None = None
        self._visible = visible
        self._suppress_dialogs = suppress_dialogs
        self._prog_id = prog_id
        self._require_type_library = require_type_library
        self._filepath: Path | None = None
        self._com_initialized = False
        # HAPAttributeNumber 常量字典，由 connect() 从 type library 加载后填充。
        # None 表示尚未加载或加载失败；AspenNode.info() 依赖此字段。
        self._hap_constants: dict[str, int] | None = None
        # set_value() 每次调用后递增，用于检测 run_case() 后是否有输入被修改。
        self._mutation_count: int = 0

    # ------------------------------------------------------------------ #
    # 连接生命周期
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """
        创建 Aspen Plus ActiveX Automation Server 实例。

        使用 EnsureDispatch 触发 win32com gencache，填充
        win32com.client.constants（含 HAPAttributeNumber 枚举），
        并将已验证的常量存入 self._hap_constants。

        Parameters（构造时传入）
        -------------------------
        require_type_library:
            True：EnsureDispatch 失败时直接抛出 AspenConnectionError，
                  适用于需要可靠节点元数据（info()）的场景。
            False（默认）：EnsureDispatch 失败时回退到 Dispatch 并记录
                  WARNING，self._hap_constants 保持 None，
                  AspenNode.info() 调用时会抛出 AspenNodeError。
        """
        if self._app is not None:
            return

        try:
            pythoncom.CoInitialize()
            self._com_initialized = True

            gencache_ok = False
            try:
                self._app = win32com.client.gencache.EnsureDispatch(self._prog_id)
                gencache_ok = True
            except Exception as gc_exc:
                if self._require_type_library:
                    self._release_com()
                    raise AspenTypeLibraryError(
                        f"EnsureDispatch 失败，无法加载 Aspen type library：{gc_exc}。"
                        "若不需要节点元数据（info()），可设置 require_type_library=False。"
                    ) from gc_exc
                _log.warning(
                    "EnsureDispatch('%s') 失败（%s），回退到 Dispatch。"
                    "hap_constants 将不可用，AspenNode.info() 调用时会抛出错误。"
                    "运行 scripts/verify_hap_constants.py 诊断。",
                    self._prog_id, gc_exc,
                )
                self._app = win32com.client.Dispatch(self._prog_id)

            if gencache_ok:
                self._hap_constants = self._load_hap_constants_from_cache()
                if self._require_type_library and self._hap_constants is None:
                    raise AspenTypeLibraryError(
                        "EnsureDispatch 成功，但 HAP 常量加载不完整。"
                        "运行 scripts/verify_hap_constants.py 诊断缺失的常量。"
                    )

            self._configure_application()
        except AspenConnectionError:
            raise
        except Exception as exc:
            self._release_com()
            raise AspenConnectionError(f"无法连接到 Aspen Plus：{exc}") from exc

    @staticmethod
    def _load_hap_constants_from_cache() -> dict[str, int] | None:
        """
        从 win32com.client.constants 读取 HAPAttributeNumber 枚举值。
        EnsureDispatch 成功后调用；若仍读不到则返回 None 并记录 WARNING。
        """
        c = win32com.client.constants
        loaded = {name: int(getattr(c, name)) for name in HAP_NAMES if hasattr(c, name)}
        if len(loaded) == len(HAP_NAMES):
            _log.debug("HAPAttributeNumber：从 type library 加载了全部 %d 个常量。", len(loaded))
            return loaded
        missing = set(HAP_NAMES) - set(loaded)
        _log.warning(
            "HAPAttributeNumber：EnsureDispatch 成功但仍有 %d 个常量缺失：%s。"
            "hap_constants 将不可用。运行 scripts/verify_hap_constants.py 诊断。",
            len(missing), missing,
        )
        return None

    def disconnect(self) -> None:
        """释放 COM 对象并关闭 Aspen Plus 实例。"""
        if self._app is not None:
            try:
                self._close_application()
            except Exception:
                pass
            finally:
                self._app = None
                self._filepath = None
                self._hap_constants = None
                self._mutation_count = 0
        self._release_com()

    def __enter__(self) -> AspenDriver:
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------ #
    # 文件操作
    # ------------------------------------------------------------------ #

    def open(self, filepath: str | Path, host_type: int = 0) -> None:
        """
        打开 Aspen Plus 仿真文件（.bkp / .apw / .apwz）。
        """
        self._require_connection()
        path = Path(filepath).resolve()
        if not path.exists():
            raise AspenFileError(f"仿真文件不存在：{path}")

        try:
            self._init_from_file(path, host_type)
            self._configure_application()
            self._filepath = path
        except Exception as exc:
            raise AspenFileError(f"无法打开文件 {path}：{exc}") from exc

    def save(self, filepath: str | Path | None = None, overwrite: bool = True) -> None:
        """保存仿真文件。若不指定路径则保存到当前打开的文件。"""
        self._require_connection()
        target = Path(filepath).resolve() if filepath else self._filepath
        if target is None:
            raise AspenFileError("未指定保存路径，且当前没有打开的文件。")

        try:
            if filepath is None or target == self._filepath:
                self._app.Save()
            else:
                self._app.SaveAs(str(target), overwrite)
                self._filepath = target
        except Exception as exc:
            raise AspenFileError(f"保存失败：{exc}") from exc

    def write_archive(self, filepath: str | Path, save_children: bool = True) -> None:
        """导出 Aspen Plus backup/archive 文件。"""
        self._require_connection()
        target = Path(filepath).resolve()
        try:
            self._app.WriteArchive2(str(target), int(save_children))
        except Exception as exc:
            raise AspenFileError(f"导出 archive 文件失败：{exc}") from exc

    # ------------------------------------------------------------------ #
    # 仿真控制
    # ------------------------------------------------------------------ #

    def reinit(self) -> None:
        """重新初始化仿真（清除结果，保留输入）。"""
        self._require_connection()
        try:
            self._app.Reinit()
        except Exception as exc:
            raise AspenRunError(f"重新初始化失败：{exc}") from exc

    def run(self, timeout: float = _RUN_TIMEOUT) -> None:
        """运行仿真，阻塞直到完成或超时。"""
        self._require_connection()
        engine = self._app.Engine

        try:
            engine.Run2(True)
        except Exception as exc:
            raise AspenRunError(f"无法启动仿真：{exc}") from exc

        deadline = time.monotonic() + timeout
        while self._engine_is_running(engine):
            pythoncom.PumpWaitingMessages()
            if time.monotonic() >= deadline:
                try:
                    engine.Stop()
                except Exception:
                    pass
                raise AspenRunTimeoutError(timeout)
            time.sleep(self._POLL_INTERVAL)

    def stop(self) -> None:
        """强制停止正在运行的仿真。"""
        self._require_connection()
        try:
            self._app.Engine.Stop()
        except Exception as exc:
            raise AspenRunError(f"停止仿真失败：{exc}") from exc

    @property
    def is_running(self) -> bool:
        """仿真引擎是否正在运行。"""
        if self._app is None:
            return False
        return self._engine_is_running(self._app.Engine)

    # ------------------------------------------------------------------ #
    # 节点读写
    # ------------------------------------------------------------------ #

    def get_node(self, path: str) -> Any:
        """返回指定树路径的 COM 节点对象。"""
        self._require_connection()
        try:
            node = self._app.Tree.FindNode(path)
        except Exception as exc:
            raise AspenNodeError(f"访问节点 '{path}' 时出错：{exc}") from exc
        if node is None:
            raise AspenNodeError(f"节点不存在：'{path}'")
        return node

    def get_value(self, path: str) -> Any:
        """读取指定树路径的值。"""
        return self.get_node(path).Value

    def set_value(self, path: str, value: Any) -> None:
        """向指定树路径写入值。"""
        node = self.get_node(path)
        try:
            node.Value = value
        except Exception as exc:
            raise AspenNodeError(f"设置 '{path}' = {value!r} 失败：{exc}") from exc
        self._mutation_count += 1

    def node_exists(self, path: str) -> bool:
        """判断指定路径的节点是否存在。"""
        self._require_connection()
        try:
            return self._app.Tree.FindNode(path) is not None
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # 属性与内部工具
    # ------------------------------------------------------------------ #

    @property
    def filepath(self) -> Path | None:
        """当前打开的仿真文件路径。"""
        return self._filepath

    @property
    def mutation_count(self) -> int:
        """
        set_value() 的累计调用次数。

        SimulationResult 记录 run_case() 完成时的快照值；
        TreeExporter 比对当前值与快照值，若不一致说明 run 后有输入被修改。
        """
        return self._mutation_count

    @property
    def hap_constants(self) -> dict[str, int] | None:
        """
        HAPAttributeNumber 常量字典（{名称: 整数值}）。

        connect() 成功且 EnsureDispatch 填充了 gencache 后可用；
        否则为 None。AspenNode.info() 依赖此属性。
        """
        return self._hap_constants

    @property
    def app(self) -> Any:
        """
        直接访问底层 COM 对象（调试/底层逃生口，谨慎使用）。

        警告：通过此属性直接修改 Aspen 树节点（如 app.Tree.FindNode(...).Value = ...）
        不会递增 mutation_count，TreeExporter 的一致性检查将无法感知输入已改变。
        如需在 aspen_driver 外部修改节点，请改用 driver.set_value()，
        或在修改后调用 driver.mark_mutated() 手动标记。
        """
        self._require_connection()
        return self._app

    def mark_mutated(self, count: int = 1) -> None:
        """
        手动递增 mutation_count，用于通过 driver.app 直接修改 Aspen 树后的标记。

        Parameters
        ----------
        count:
            递增量，默认 1。若一次性修改了多个节点，传入实际修改数量。
            必须为正整数，否则抛出 ValueError。
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"mark_mutated() 的 count 必须为正整数，收到：{count!r}")
        self._mutation_count += count

    def _require_connection(self) -> None:
        if self._app is None:
            raise AspenConnectionError("未连接，请先调用 connect()。")

    def _configure_application(self) -> None:
        if self._app is None:
            return
        try:
            self._app.Visible = self._visible
        except Exception:
            pass
        try:
            self._app.SuppressDialogs = self._suppress_dialogs
        except Exception:
            pass

    def _init_from_file(self, path: Path, host_type: int) -> None:
        if self._app is None:
            raise AspenConnectionError("未连接，请先调用 connect()。")

        suffix = path.suffix.lower()
        if suffix == ".bkp":
            init_methods = ("InitFromArchive2", "InitFromFile2")
        else:
            init_methods = ("InitFromFile2", "InitFromArchive2")

        last_error: Exception | None = None
        for method_name in init_methods:
            try:
                getattr(self._app, method_name)(str(path), host_type)
                return
            except Exception as exc:
                last_error = exc

        raise AspenFileError(f"InitFromArchive2/InitFromFile2 均失败：{last_error}")

    def _close_application(self) -> None:
        if self._app is None:
            return

        self._configure_application()
        for call in (
            lambda: self._app.Close(False),
            lambda: self._app.Close(),
            lambda: self._app.Quit(),
        ):
            try:
                call()
                return
            except Exception:
                continue

    def _release_com(self) -> None:
        if self._com_initialized:
            try:
                pythoncom.CoUninitialize()
            finally:
                self._com_initialized = False

    @staticmethod
    def _engine_is_running(engine: Any) -> bool:
        try:
            return bool(engine.IsRunning)
        except Exception as exc:
            raise AspenRunError(f"读取仿真运行状态失败：{exc}") from exc
