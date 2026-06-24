"""
driver.py — Aspen Plus COM 底层接口封装。

职责：连接生命周期管理、文件操作、仿真控制、节点读写。
不包含任何业务逻辑，业务逻辑由 runner.py 及上层模块负责。

STA 线程模型
------------
所有 COM 操作通过 ComApartment（com_thread.py）在专用 STA 线程上执行，
避免 RPC_E_WRONG_THREAD 错误和多 agent 并发时的 COM 竞争。

  connect()    → 启动 STA 线程，在线程内完成 EnsureDispatch + HAP 常量加载
  run()        → 整个 Run2 + IsRunning 轮询 + PumpWaitingMessages 在 STA 线程内
  get_value()  → FindNode + .Value 在 STA 线程内原子执行
  set_value()  → FindNode + .Value 赋值在 STA 线程内原子执行
  disconnect() → 在 STA 线程关闭应用，然后 shutdown() 停止线程
  recover()    → 重建 ComApartment 实例后重新 connect()
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import win32com.client

from .com_thread import ComApartment
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
        # HAPAttributeNumber 常量字典，由 connect() 在 STA 线程内加载后填充。
        # None 表示尚未加载或加载失败；AspenNode.info() 依赖此字段。
        self._hap_constants: dict[str, int] | None = None
        # set_value() 每次调用后递增，用于检测 run_case() 后是否有输入被修改。
        self._mutation_count: int = 0
        # COM 自愈标志：仿真超时或 Run2 抛 COM 异常后置为 True，
        # 表示底层 COM/引擎可能已处于不一致状态，下一次运行前应先 recover()。
        self._needs_recovery: bool = False
        # STA 专用后台线程。connect() 时 start()，disconnect() 时 shutdown()。
        # recover() 在重新 connect() 前创建新实例。
        self._com_apt: ComApartment = ComApartment()

    # ------------------------------------------------------------------ #
    # 连接生命周期
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """
        创建 Aspen Plus ActiveX Automation Server 实例。

        在 STA 专用后台线程中执行 EnsureDispatch，触发 win32com gencache，
        填充 win32com.client.constants（含 HAPAttributeNumber 枚举），
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

        # 启动 STA 线程（CoInitialize 在线程内执行）
        self._com_apt.start()

        # 将全部 COM 初始化逻辑提交到 STA 线程
        try:
            self._com_apt.submit(self._connect_on_com_thread)
        except AspenConnectionError:
            raise
        except Exception as exc:
            raise AspenConnectionError(f"无法连接到 Aspen Plus：{exc}") from exc

    def _connect_on_com_thread(self) -> None:
        """
        在 STA 线程上执行 EnsureDispatch + HAP 常量加载 + 应用配置。

        注意：CoInitialize() 已由 ComApartment._run_loop() 处理，此处不调用。
        此方法只能通过 _com_apt.submit() 在 STA 线程内调用，不可直接调用。
        """
        gencache_ok = False
        try:
            self._app = win32com.client.gencache.EnsureDispatch(self._prog_id)
            gencache_ok = True
        except AttributeError as gc_exc:
            # win32com gen_py 缓存损坏（CLSIDToPackageMap 缺失等），清除后重试。
            # 常见于 pywin32 升级后或缓存文件被部分删除的情况。
            _log.warning(
                "EnsureDispatch('%s') 遇到缓存损坏错误（%s），尝试清除 gen_py 缓存后重试。",
                self._prog_id, gc_exc,
            )
            try:
                import shutil, os, sys, tempfile
                gen_py_dir = os.path.join(tempfile.gettempdir(), "gen_py")
                if os.path.isdir(gen_py_dir):
                    shutil.rmtree(gen_py_dir)
                    _log.info("已清除 gen_py 缓存目录：%s", gen_py_dir)
                # 磁盘缓存删除后，Python 内存中仍持有旧模块引用，
                # 必须同步清除，否则重试时仍会拿到损坏的过期模块。
                stale = [k for k in list(sys.modules.keys()) if "gen_py" in k]
                for k in stale:
                    del sys.modules[k]
                if stale:
                    _log.info("已从 sys.modules 清除 %d 个过期 gen_py 条目。", len(stale))
                win32com.client.gencache.is_readonly = False
                self._app = win32com.client.gencache.EnsureDispatch(self._prog_id)
                gencache_ok = True
                _log.info("清除缓存后 EnsureDispatch 成功。")
            except Exception as retry_exc:
                gc_exc = retry_exc
                if self._require_type_library:
                    raise AspenTypeLibraryError(
                        f"EnsureDispatch 失败，无法加载 Aspen type library：{gc_exc}。"
                        "若不需要节点元数据（info()），可设置 require_type_library=False。"
                    ) from gc_exc
                _log.warning(
                    "清除缓存后 EnsureDispatch 仍失败（%s），回退到 Dispatch。"
                    "hap_constants 将不可用，AspenNode.info() 调用时会抛出错误。",
                    gc_exc,
                )
                self._app = win32com.client.Dispatch(self._prog_id)
        except Exception as gc_exc:
            if self._require_type_library:
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
                self._com_apt.submit(self._close_application)
            except Exception:
                pass
            finally:
                self._app = None
                self._filepath = None
                self._hap_constants = None
                self._mutation_count = 0
        # 停止 STA 线程（CoUninitialize 在线程内执行）
        self._com_apt.shutdown()

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
            def _do() -> None:
                self._init_from_file(path, host_type)
                self._configure_application()
            self._com_apt.submit(_do)
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
                self._com_apt.submit(lambda: self._app.Save())
            else:
                self._com_apt.submit(lambda: self._app.SaveAs(str(target), overwrite))
                self._filepath = target
        except Exception as exc:
            raise AspenFileError(f"保存失败：{exc}") from exc

    def write_archive(self, filepath: str | Path, save_children: bool = True) -> None:
        """导出 Aspen Plus backup/archive 文件。"""
        self._require_connection()
        target = Path(filepath).resolve()
        try:
            self._com_apt.submit(
                lambda: self._app.WriteArchive2(str(target), int(save_children))
            )
        except Exception as exc:
            raise AspenFileError(f"导出 archive 文件失败：{exc}") from exc

    # ------------------------------------------------------------------ #
    # 仿真控制
    # ------------------------------------------------------------------ #

    def reinit(self) -> None:
        """重新初始化仿真（清除结果，保留输入）。"""
        self._require_connection()
        try:
            self._com_apt.submit(lambda: self._app.Reinit())
        except Exception as exc:
            raise AspenRunError(f"重新初始化失败：{exc}") from exc

    def run(self, timeout: float = _RUN_TIMEOUT) -> None:
        """运行仿真，阻塞直到完成或超时。

        整个运行循环（Run2 启动、IsRunning 轮询、PumpWaitingMessages）
        在 STA 线程内完成，调用线程只阻塞等待最终结果。

        com_timeout = timeout + 30s，为 Stop() 和清理留出余量。

        超时或启动失败时，除抛出异常外还会置 _needs_recovery=True，
        提示上层在下一次运行前调用 recover() 重建 COM 连接——
        因为超时后 Aspen 引擎常处于不一致状态，后续 Run2 会连续抛
        COM 异常（DISP_E_EXCEPTION），不重建连接则整轮 DOE 全部报废。
        """
        self._require_connection()
        import pythoncom as _pythoncom  # 在 STA 线程内使用，不在模块顶部导入
        com_timeout = timeout + 30.0

        def _run_on_com_thread() -> None:
            engine = self._app.Engine
            _log.debug("driver.run: 调用 Run2(True)，IsRunning=%s",
                       self._engine_is_running(engine))
            try:
                engine.Run2(True)
            except Exception as exc:
                # Run2 启动即抛异常，通常是上一次超时残留的坏连接
                raise AspenRunError(f"无法启动仿真：{exc}") from exc

            _log.debug("driver.run: Run2(True) 返回，IsRunning=%s",
                       self._engine_is_running(engine))

            deadline = time.monotonic() + timeout
            poll_count = 0
            while self._engine_is_running(engine):
                # 在 STA 线程内泵送 COM 消息，防止 Aspen 内部回调死锁
                try:
                    _pythoncom.PumpWaitingMessages()
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    try:
                        engine.Stop()
                    except Exception:
                        pass
                    raise AspenRunTimeoutError(timeout)
                time.sleep(self._POLL_INTERVAL)
                poll_count += 1

            _log.debug("driver.run: 引擎停止，poll_count=%d（0=Run2返回时已停止）",
                       poll_count)
            # 引擎停止后读取状态信息，帮助诊断 NO_RESULTS 原因
            for attr in ("StatusMessage", "ErrorMessage", "Status", "ErrorCode"):
                try:
                    val = getattr(engine, attr, None)
                    if val is not None and str(val).strip():
                        _log.warning("driver.run: engine.%s = %r", attr, val)
                except Exception:
                    pass

        try:
            self._com_apt.submit(_run_on_com_thread, timeout=com_timeout)
        except AspenRunTimeoutError:
            # 超时：引擎可能卡住，COM 状态不可信 → 标记需恢复
            self._needs_recovery = True
            raise
        except AspenRunError:
            # Run2 启动即失败：连接可能已损坏 → 标记需恢复
            self._needs_recovery = True
            raise
        except Exception as exc:
            self._needs_recovery = True
            raise AspenRunError(f"仿真运行失败：{exc}") from exc

    def stop(self) -> None:
        """强制停止正在运行的仿真。"""
        self._require_connection()
        try:
            self._com_apt.submit(lambda: self._app.Engine.Stop(), timeout=10.0)
        except Exception as exc:
            raise AspenRunError(f"停止仿真失败：{exc}") from exc

    @property
    def is_running(self) -> bool:
        """仿真引擎是否正在运行。"""
        if self._app is None:
            return False
        try:
            return self._com_apt.submit(
                lambda: self._engine_is_running(self._app.Engine),
                timeout=10.0,
            )
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # 节点读写
    # ------------------------------------------------------------------ #

    def get_node(self, path: str) -> Any:
        """返回指定树路径的 COM 节点对象（在 STA 线程上获取）。

        警告：返回的 COM 对象属于 STA 线程，不能在调用线程上直接调用其方法，
        否则会引发 RPC_E_WRONG_THREAD（-2147417842）。
        需要对节点执行操作时，请使用 submit_on_node()，而非此方法。
        """
        self._require_connection()

        def _do() -> Any:
            try:
                node = self._app.Tree.FindNode(path)
            except Exception as exc:
                raise AspenNodeError(f"访问节点 '{path}' 时出错：{exc}") from exc
            if node is None:
                raise AspenNodeError(f"节点不存在：'{path}'")
            return node

        return self._com_apt.submit(_do)

    def submit_on_node(self, path: str, fn: Any) -> Any:
        """
        在 STA 线程上找到指定路径的 COM 节点，并在同一 STA 线程内执行 fn(com_node)。

        STA COM 对象（IHNode）的所有方法（AttributeValue、UnitString、Dimension、
        Elements 等）必须在创建它的 STA 线程上调用，否则会引发 RPC_E_WRONG_THREAD。
        此方法将 FindNode 与后续操作合并为一次原子 STA 提交，确保线程安全。

        Parameters
        ----------
        path:
            Aspen 树节点路径。
        fn:
            接受一个 COM 节点对象（IHNode）并返回任意值的可调用对象。
            在 STA 线程内执行，不得阻塞或调用非线程安全操作。
        """
        self._require_connection()

        def _do() -> Any:
            try:
                node = self._app.Tree.FindNode(path)
            except Exception as exc:
                raise AspenNodeError(f"访问节点 '{path}' 时出错：{exc}") from exc
            if node is None:
                raise AspenNodeError(f"节点不存在：'{path}'")
            return fn(node)

        return self._com_apt.submit(_do)

    def get_value(self, path: str) -> Any:
        """读取指定树路径的值（FindNode + .Value 在 STA 线程内原子执行）。"""
        return self._com_apt.submit(
            lambda: self._app.Tree.FindNode(path).Value
        )

    def set_value(self, path: str, value: Any) -> None:
        """向指定树路径写入值（FindNode + 赋值在 STA 线程内原子执行）。"""
        def _do() -> None:
            node = self._app.Tree.FindNode(path)
            if node is None:
                raise AspenNodeError(f"节点不存在：'{path}'")
            try:
                node.Value = value
            except Exception as exc:
                raise AspenNodeError(f"设置 '{path}' = {value!r} 失败：{exc}") from exc

        self._com_apt.submit(_do)
        # mutation_count 在调用线程递增（优化循环单线程访问，无竞争）
        self._mutation_count += 1

    def node_exists(self, path: str) -> bool:
        """判断指定路径的节点是否存在。"""
        self._require_connection()
        try:
            return self._com_apt.submit(
                lambda: self._app.Tree.FindNode(path) is not None,
                timeout=10.0,
            )
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

    # ------------------------------------------------------------------ #
    # COM 自愈（执行层,与工艺无关）
    # ------------------------------------------------------------------ #

    @property
    def needs_recovery(self) -> bool:
        """
        底层 COM/引擎是否处于需要重建的不一致状态。

        在仿真超时或 Run2 启动异常后置为 True。上层运行循环应在每次
        run 前检查此标志,为 True 时先调用 recover(),避免坏连接导致
        后续所有仿真连锁失败(DISP_E_EXCEPTION)。
        """
        return self._needs_recovery

    def recover(self) -> bool:
        """
        重建 Aspen COM 连接并重新打开当前文件，清除 needs_recovery 标志。

        用于超时/COM 崩溃后的执行层自愈：彻底断开旧连接（释放损坏的 COM
        对象和 STA 线程），重新创建 ComApartment，再 connect() + open()。
        整个过程与具体工艺无关。

        Returns
        -------
        bool
            True  恢复成功，可继续运行后续工况。
            False 恢复失败（连接或打开文件抛异常），调用方应终止本轮优化。
            无论成败，_needs_recovery 都会被清零（失败时由上层据返回值终止）。
        """
        saved_filepath = self._filepath
        _log.warning(
            "driver.recover: 检测到 COM 需恢复，开始重建连接(file=%s)。",
            saved_filepath,
        )
        # 先彻底断开旧连接，释放可能损坏的 COM 对象并停止 STA 线程
        try:
            self.disconnect()
        except Exception as exc:  # noqa: BLE001
            _log.warning("driver.recover: disconnect 阶段异常（已忽略，继续重连）：%s", exc)

        # disconnect() 已 shutdown 旧 _com_apt；创建新实例供下次 connect() 使用
        self._com_apt = ComApartment()
        self._needs_recovery = False
        try:
            self.connect()
            if saved_filepath is not None:
                self.open(saved_filepath)
            _log.info("driver.recover: COM 连接已重建，文件已重新打开。")
            return True
        except Exception as exc:  # noqa: BLE001
            _log.error("driver.recover: 重建连接失败，本轮优化应终止：%s", exc)
            return False

    def clear_recovery_flag(self) -> None:
        """手动清除 needs_recovery 标志（上层已用其他方式处理时使用）。"""
        self._needs_recovery = False

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
        """已废弃：CoInitialize/CoUninitialize 改由 ComApartment 内部管理。保留为空操作确保兼容性。"""
        pass  # noqa: PIE790 — 保留以防外部代码仍有调用

    @staticmethod
    def _engine_is_running(engine: Any) -> bool:
        try:
            return bool(engine.IsRunning)
        except Exception as exc:
            raise AspenRunError(f"读取仿真运行状态失败：{exc}") from exc
