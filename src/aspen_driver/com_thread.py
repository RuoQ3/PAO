"""
com_thread.py — COM STA 专用后台线程。

架构说明
--------
Aspen Plus COM 对象必须在同一个 STA（单线程单元）线程上创建和访问，
否则跨线程调用会触发 RPC_E_WRONG_THREAD 错误，导致 COM 服务器崩溃。

本模块提供 ComApartment 类，将所有 COM 操作隔离到一个持久化后台线程中：
  - 该线程在 start() 时调用 CoInitialize()，在 shutdown() 后调用 CoUninitialize()
  - 调用方通过 submit(fn) 将任意 callable 提交到 COM 线程执行
  - submit() 会阻塞调用线程直到 fn 执行完毕，并返回结果或重新抛出异常
  - 每个 job 执行后自动调用 PumpWaitingMessages()，防止跨线程 RPC 死锁

使用方式（AspenDriver 内部）
----------------------------
    self._com_apt = ComApartment()
    self._com_apt.start()                          # connect() 中调用
    result = self._com_apt.submit(lambda: ...)     # 所有 COM 操作
    self._com_apt.shutdown()                       # disconnect() 中调用

参考实现：aspen-mcp/src/aspen_mcp/com_bridge.py
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, TypeVar

import pythoncom

_log = logging.getLogger(__name__)

T = TypeVar("T")

# submit() 的默认超时（秒）：普通节点读写、open、reinit 等操作
_DEFAULT_TIMEOUT: float = 120.0

# 停止信号：放入队列后 STA 线程退出循环
_STOP_SENTINEL = object()


class _ComJob:
    """
    一个 COM 线程工作单元。

    调用方通过 wait() 阻塞等待 STA 线程执行 fn() 的结果；
    fn() 的返回值或异常通过内部队列传回调用线程。
    """

    __slots__ = ("fn", "_result_q")

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.fn = fn
        # maxsize=1：结果只写入一次
        self._result_q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def run(self) -> None:
        """在 STA 线程上执行 fn()，将结果放入队列。"""
        try:
            value = self.fn()
            self._result_q.put(("ok", value))
        except BaseException as exc:  # noqa: BLE001 — 所有异常都要传回调用线程
            self._result_q.put(("err", exc))

    def wait(self, timeout: float) -> Any:
        """
        阻塞调用线程直到结果就绪。

        Returns
        -------
        fn() 的返回值。

        Raises
        ------
        TimeoutError:
            等待超过 timeout 秒。
        BaseException:
            fn() 中抛出的任何异常（原样重新抛出）。
        """
        try:
            status, value = self._result_q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"COM 操作超时（{timeout}s）。"
                "仿真运行时请通过 run_timeout 参数控制超时，"
                "并确保 com_timeout = run_timeout + 30 留有余量。"
            )
        if status == "err":
            raise value
        return value


class ComApartment:
    """
    STA 后台线程持有者。

    线程生命周期
    ------------
    1. __init__()   — 创建队列和 Event，不启动线程
    2. start()      — 启动 STA 线程，阻塞直到 CoInitialize() 完成
    3. submit(fn)   — 提交任务，阻塞等待结果
    4. shutdown()   — 发送哨兵信号，等待线程退出（CoUninitialize 在线程内执行）

    connect() / disconnect() / recover() 管理此生命周期。

    线程安全性
    ----------
    - submit() 可以从任意线程调用（队列操作是线程安全的）
    - _app、_hap_constants 等 COM 对象属性只在 STA 线程内访问
    - AspenDriver 的 Python 层属性（mutation_count 等）由单一调用线程修改
    """

    # 普通操作默认超时（节点读写、open、save 等）
    DEFAULT_TIMEOUT: float = _DEFAULT_TIMEOUT

    def __init__(self) -> None:
        self._job_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """
        启动 STA 线程，阻塞直到 CoInitialize() 完成（最多等待 10s）。

        幂等：若线程已在运行则直接返回。
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._started.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="aspen-com-apt",
            daemon=True,  # 进程退出时自动终止，不阻塞 Python 解释器关闭
        )
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError(
                "COM apartment 线程未能在 10s 内就绪（CoInitialize 超时）。"
                "请检查 pythoncom 是否正常安装（pip install pywin32）。"
            )
        _log.debug("ComApartment: STA 线程已就绪（thread_id=%d）。",
                   self._thread.ident)

    def shutdown(self) -> None:
        """
        停止 STA 线程，等待退出（最多 10s）。

        CoUninitialize() 由 _run_loop 的 finally 块负责调用。
        安全：若线程未启动或已退出则静默返回。
        """
        if self._thread is not None and self._thread.is_alive():
            self._job_q.put(_STOP_SENTINEL)
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                _log.warning("ComApartment: STA 线程在 10s 内未退出。")
        _log.debug("ComApartment: STA 线程已关闭。")

    # ------------------------------------------------------------------ #
    # 任务提交
    # ------------------------------------------------------------------ #

    def submit(self, fn: Callable[[], T], timeout: float = _DEFAULT_TIMEOUT) -> T:
        """
        将 fn 提交到 STA 线程执行，阻塞等待结果。

        Parameters
        ----------
        fn:
            在 STA 线程上调用的 callable。通常是 lambda 捕获 self._app。
            fn 应是轻量操作；长时间运行（如 run 循环）须自行处理消息泵。
        timeout:
            等待超时秒数。普通操作用默认值（120s）；
            仿真运行用 run_timeout + 30s。

        Returns
        -------
        fn() 的返回值。

        Raises
        ------
        TimeoutError:
            等待超过 timeout 秒。
        任意异常:
            fn() 抛出的异常（原样传回调用线程）。
        """
        job = _ComJob(fn)
        self._job_q.put(job)
        return job.wait(timeout)

    # ------------------------------------------------------------------ #
    # STA 线程主循环（内部）
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """
        STA 线程主循环。

        流程：
          1. CoInitialize() — 将此线程初始化为 STA
          2. 设置 _started Event 通知 start() 返回
          3. 循环取出 job 并执行，每次执行后泵送 COM 消息
          4. 收到哨兵信号 → 退出循环
          5. CoUninitialize() — 释放 COM 资源
        """
        pythoncom.CoInitialize()
        self._started.set()
        _log.debug("ComApartment._run_loop: 进入 STA 消息循环。")
        try:
            while True:
                item = self._job_q.get()
                if item is _STOP_SENTINEL:
                    _log.debug("ComApartment._run_loop: 收到停止信号，退出循环。")
                    break
                # 类型断言（测试环境中不引入 assert 开销）
                if not isinstance(item, _ComJob):
                    _log.error("ComApartment._run_loop: 队列收到非 _ComJob 对象 %r，已跳过。", item)
                    continue
                item.run()
                # 每个 job 后泵送 COM 消息，防止 Aspen 内部回调死锁
                try:
                    pythoncom.PumpWaitingMessages()
                except Exception:
                    pass
        finally:
            pythoncom.CoUninitialize()
            _log.debug("ComApartment._run_loop: CoUninitialize 完成，线程退出。")
