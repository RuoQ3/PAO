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
        """Create a dedicated DispatchEx application on its owned COM STA.

        HAP constants must already be available from an installed type library.
        Missing constants remain explicit; no user instance or shared cache is
        modified to recover them.
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
        # DispatchEx requests a new application object, never the ROT instance
        # potentially owned by the user. Do not delete shared gen_py caches.
        self._app = win32com.client.DispatchEx(self._prog_id)
        self._hap_constants = self._load_hap_constants_from_cache()
        if self._require_type_library and self._hap_constants is None:
            raise AspenTypeLibraryError("HAP constants unavailable for dedicated instance")

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
        """Close only our DispatchEx instance; report every cleanup failure."""
        errors = []
        if self._app is not None:
            try:
                self._com_apt.submit(self._close_application, timeout=2.0, cleanup=True)
            except Exception as exc:
                errors.append(str(exc))
        try:
            self._com_apt.shutdown(timeout=2.0)
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            self._needs_recovery = True
            raise AspenRunError("cleanup incomplete: " + "; ".join(errors))
        self._app = None
        self._filepath = None
        self._hap_constants = None
        self._mutation_count = 0

    def __enter__(self) -> AspenDriver:
        self.connect()
        return self

    def __exit__(self, exc_type, primary, traceback) -> None:
        try:
            self.disconnect()
        except Exception as cleanup:
            if primary is None:
                raise
            primary.cleanup_errors = list(getattr(primary, "cleanup_errors", [])) + [str(cleanup)]
            _log.error("cleanup incomplete after primary failure: %s", cleanup)

    # ------------------------------------------------------------------ #
    # 文件操作
    # ------------------------------------------------------------------ #

    def open(self, filepath: str | Path, host_type: int = 0) -> None:
        """
        打开 Aspen Plus 仿真文件（.bkp / .apw / .apwz）。
        """
        from src.agents.actions.authorization import check_model_open
        check_model_open(filepath)
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
        raise PermissionError("Saving Aspen documents requires a separate approved file-write plan; disabled")

    def write_archive(self, filepath: str | Path, save_children: bool = True) -> None:
        """导出 Aspen Plus backup/archive 文件。"""
        raise PermissionError("Writing Aspen archives requires a separate approved file-write plan; disabled")

    # ------------------------------------------------------------------ #
    # 仿真控制
    # ------------------------------------------------------------------ #

    def reinit(self) -> None:
        """重新初始化仿真（清除结果，保留输入）。"""
        from src.agents.actions.authorization import require_authorization
        require_authorization()
        self._require_connection()
        try:
            self._com_apt.submit(lambda: self._app.Reinit())
        except Exception as exc:
            raise AspenRunError(f"重新初始化失败：{exc}") from exc

    def run(self, timeout: float = _RUN_TIMEOUT) -> None:
        """Run on the owned STA with cooperative Stop at the deadline.

        Timeout/start failure marks the unit unknown. No automatic reconnect
        or retry is allowed. The caller's bounded wait and cleanup status are
        tracked separately by ExecutionUnit; Stop return is not termination.
        """
        from src.agents.actions.authorization import check_run
        grant = check_run(started=False)
        self._require_connection()
        import pythoncom as _pythoncom  # 在 STA 线程内使用，不在模块顶部导入
        com_timeout = timeout + 30.0

        def _run_on_com_thread() -> None:
            request = getattr(self, "stop_requested", None)
            if grant.closed or request is not None and request.is_set() or time.monotonic() >= grant.deadline:
                raise AspenRunError("execution deadline/stop reached before engine start; model not run")
            engine = self._app.Engine
            _log.debug("driver.run: 调用 Run2(True)，IsRunning=%s",
                       self._engine_is_running(engine))
            try:
                # Count the engine invocation itself, even if its outcome is unknown.
                grant.started_runs += 1
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
                if time.monotonic() >= deadline or getattr(self, "stop_requested", None) is not None and self.stop_requested.is_set():
                    try:
                        engine.Stop()
                    except Exception as exc:
                        raise AspenRunError(f"deadline exceeded; stop failed; execution unknown: {exc}") from exc
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
        from src.agents.actions.authorization import check_write
        check_write(path, value)
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
        """Unknown execution/COM state; this driver must not be reused."""
        return self._needs_recovery

    def recover(self) -> bool:
        raise AspenRunError("unknown execution cannot be retried or reconnected automatically; inspect resources and approve a new plan")

    def clear_recovery_flag(self) -> None:
        """Unknown units cannot be made reusable by clearing a boolean."""
        if self._needs_recovery:
            raise AspenRunError("unknown execution cannot be cleared; inspect resources and create a new approved unit")

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
            except Exception as exc:
                last_error = exc
        raise AspenRunError(f"application close failed: {last_error}")

    def _release_com(self) -> None:
        """已废弃：CoInitialize/CoUninitialize 改由 ComApartment 内部管理。保留为空操作确保兼容性。"""
        pass  # noqa: PIE790 — 保留以防外部代码仍有调用

    @staticmethod
    def _engine_is_running(engine: Any) -> bool:
        try:
            return bool(engine.IsRunning)
        except Exception as exc:
            raise AspenRunError(f"读取仿真运行状态失败：{exc}") from exc
