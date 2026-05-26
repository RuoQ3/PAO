"""
logger.py — PAO 统一日志配置与工具集。

功能
----
  setup_logging()     配置全局日志（彩色控制台 + 可选文件轮转）
  get_logger()        logging.getLogger() 的便捷封装
  LogContext          上下文管理器，为当前线程所有日志注入结构化字段
  PerformanceTimer    计时上下文管理器，自动记录耗时
  ProgressLogger      优化迭代进度专用日志器（含 ETA 估算）
  suppress_loggers()  静默第三方噪声日志器

典型用法
--------
    from src.utils.logger import setup_logging, get_logger, PerformanceTimer, LogContext

    setup_logging("INFO", log_file="output/run.log")
    log = get_logger(__name__)

    with PerformanceTimer(log, "高斯过程拟合"):
        gp.fit(X, y)

    with LogContext(iteration=5, total=30, phase="bo"):
        log.info("工况运行完成，目标值 = %.4g", obj_val)

与 main.py 的关系
-----------------
main.py 中的 _setup_logging() 可替换为本模块的 setup_logging()：

    # 旧写法
    logging.basicConfig(level=..., format=..., datefmt=...)

    # 新写法（支持彩色输出和文件轮转）
    from src.utils.logger import setup_logging
    setup_logging(args.log, log_file=db_path.parent / "run.log")
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import time
from pathlib import Path
from typing import Any, Generator
from contextlib import contextmanager


# ---------------------------------------------------------------------------
# ANSI 颜色支持
# ---------------------------------------------------------------------------

def _enable_windows_ansi() -> bool:
    """在 Windows 10+ 上启用 ANSI 虚拟终端处理（ENABLE_VIRTUAL_TERMINAL_PROCESSING）。"""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # ENABLE_PROCESSED_OUTPUT(1) | ENABLE_WRAP_AT_EOL_OUTPUT(2) | ENABLE_VIRTUAL_TERMINAL_PROCESSING(4)
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 0x0007)
        return True
    except Exception:
        return False


def _supports_color(stream: Any = None) -> bool:
    """检测终端是否支持 ANSI 颜色输出。"""
    if stream is None:
        stream = sys.stderr
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if sys.platform == "win32":
        return _enable_windows_ansi()
    return True


# 各级别对应的 ANSI 前景色
_LEVEL_COLORS: dict[str, str] = {
    "DEBUG":    "\033[36m",    # 青色
    "INFO":     "\033[32m",    # 绿色
    "WARNING":  "\033[33m",    # 黄色
    "ERROR":    "\033[31m",    # 红色
    "CRITICAL": "\033[1;35m",  # 粗体洋红
}
_RESET  = "\033[0m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"


# ---------------------------------------------------------------------------
# ColorFormatter
# ---------------------------------------------------------------------------

class ColorFormatter(logging.Formatter):
    """
    带 ANSI 颜色的日志格式化器。

    格式：HH:MM:SS [LEVEL   ] logger.name: message
      - 时间戳：暗色
      - 级别标签：按级别着色，固定宽度 8 字符
      - logger 名称：暗色
      - 消息：正常色

    use_color=False 时退化为纯文本，适合写入日志文件。
    """

    _FMT     = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _DATEFMT = "%H:%M:%S"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # 复制 record，避免修改原始对象影响其他 handler
        r = logging.makeLogRecord(record.__dict__)
        if self._use_color:
            color = _LEVEL_COLORS.get(r.levelname, "")
            r.levelname = f"{color}{r.levelname:<8}{_RESET}"
            r.name      = f"{_DIM}{r.name}{_RESET}"
        else:
            r.levelname = f"{r.levelname:<8}"
        return super().format(r)


# ---------------------------------------------------------------------------
# 线程本地上下文注入
# ---------------------------------------------------------------------------

_context_local = threading.local()


class _ContextFilter(logging.Filter):
    """
    将线程本地上下文字段以 "[k=v ...]" 前缀注入日志消息。

    由 setup_logging() 自动添加到根 logger，无需手动使用。
    """

    _MARKER = "__pao_ctx_applied__"

    def filter(self, record: logging.LogRecord) -> bool:
        # 防止同一 record 被多个 handler 重复注入
        if getattr(record, self._MARKER, False):
            return True
        stack: list[dict[str, Any]] = getattr(_context_local, "stack", [])
        if stack:
            merged: dict[str, Any] = {}
            for frame in stack:
                merged.update(frame)
            if merged:
                ctx_str = " ".join(f"{k}={v}" for k, v in merged.items())
                # 转义 % 防止与 record.args 的格式化冲突
                ctx_str = ctx_str.replace("%", "%%")
                record.msg = f"[{ctx_str}] {record.msg}"
                setattr(record, self._MARKER, True)
        return True


class LogContext:
    """
    上下文管理器，为当前线程的所有日志记录注入结构化字段。

    字段以 "key=value" 形式前缀到消息中，支持嵌套（内层字段覆盖外层同名字段）。
    退出上下文后自动恢复，不影响外层上下文。

    用法
    ----
        with LogContext(iteration=5, total=30, phase="bo"):
            log.info("工况运行完成")
            # 输出：[iteration=5 total=30 phase=bo] 工况运行完成

        # 嵌套
        with LogContext(phase="doe"):
            with LogContext(sample=3):
                log.debug("采样点生成")
                # 输出：[phase=doe sample=3] 采样点生成

    注意
    ----
    需要先调用 setup_logging() 以注册 _ContextFilter，否则字段不会出现在输出中。
    """

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __enter__(self) -> "LogContext":
        if not hasattr(_context_local, "stack"):
            _context_local.stack = []
        _context_local.stack.append(self._fields)
        return self

    def __exit__(self, *_: object) -> None:
        if hasattr(_context_local, "stack") and _context_local.stack:
            _context_local.stack.pop()


# ---------------------------------------------------------------------------
# setup_logging / get_logger
# ---------------------------------------------------------------------------

def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
    use_color: bool | None = None,
    suppress: list[str] | None = None,
) -> None:
    """
    配置 PAO 全局日志系统，替代 main.py 中的 logging.basicConfig()。

    参数
    ----
    level        : 日志级别（"DEBUG"/"INFO"/"WARNING"/"ERROR" 或对应整数）
    log_file     : 日志文件路径（None 表示不写文件）；父目录不存在时自动创建
    max_bytes    : 单个日志文件最大字节数（默认 10 MB），超出后轮转
    backup_count : 保留的历史日志文件数（默认 3，即最多保留 .1/.2/.3）
    use_color    : 是否启用彩色控制台输出（None 表示自动检测 TTY）
    suppress     : 额外需要静默到 WARNING 的第三方 logger 名称列表

    行为
    ----
    - 清除根 logger 上已有的 handler，避免重复输出
    - 注册 _ContextFilter，使 LogContext 的字段注入生效
    - 控制台输出到 stderr（与 Aspen COM 的 stdout 输出分离）
    - 文件输出使用 RotatingFileHandler，UTF-8 编码，不带颜色
    - 默认静默：skopt、sklearn、matplotlib、PIL、urllib3、httpx
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    # 清除已有 handler，防止 basicConfig 或多次调用导致重复输出
    root.handlers.clear()

    # 注册上下文注入 filter（作用于根 logger，所有子 logger 均继承）
    root.addFilter(_ContextFilter())

    # 控制台 handler（stderr，避免与 Aspen COM stdout 混淆）
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    _color = use_color if use_color is not None else _supports_color(sys.stderr)
    console.setFormatter(ColorFormatter(use_color=_color))
    root.addHandler(console)

    # 文件 handler（可选，纯文本，支持轮转）
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(ColorFormatter(use_color=False))
        root.addHandler(fh)

    # 静默第三方噪声 logger（优化库在 INFO 级别输出大量内部信息）
    _noisy = ["skopt", "sklearn", "matplotlib", "PIL", "urllib3", "httpx"]
    for name in _noisy + (suppress or []):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    logging.getLogger() 的便捷封装，提供统一的导入入口。

    与直接调用 logging.getLogger(name) 完全等价。
    推荐在每个模块顶部使用：

        from src.utils.logger import get_logger
        _log = get_logger(__name__)
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# PerformanceTimer
# ---------------------------------------------------------------------------

class PerformanceTimer:
    """
    计时上下文管理器，在退出时自动记录耗时。

    用法
    ----
        # 基本用法（DEBUG 级别）
        with PerformanceTimer(log, "高斯过程拟合"):
            gp.fit(X, y)
        # → DEBUG: 高斯过程拟合 完成，耗时 1.234 s

        # 超时自动升级为 WARNING
        with PerformanceTimer(log, "Aspen 仿真", threshold_warn=120.0):
            runner.run()
        # 若耗时 > 120s → WARNING: Aspen 仿真 耗时过长：135.2 s（阈值 120.0 s）

        # 读取耗时
        with PerformanceTimer(log, "数据库写入") as t:
            db.save(case)
        print(f"写入耗时 {t.elapsed:.3f} s")

        # 异常时也会记录（不抑制异常）
        with PerformanceTimer(log, "危险操作"):
            raise ValueError("出错了")
        # → DEBUG: 危险操作 中断（0.001 s）：出错了
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        level: int = logging.DEBUG,
        threshold_warn: float | None = None,
    ) -> None:
        """
        参数
        ----
        logger         : 日志器
        label          : 操作标签（出现在日志消息中）
        level          : 正常完成时的日志级别（默认 DEBUG）
        threshold_warn : 超过此秒数时升级为 WARNING（None 表示不升级）
        """
        self._log            = logger
        self._label          = label
        self._level          = level
        self._threshold_warn = threshold_warn
        self._start: float   = 0.0
        self.elapsed: float  = 0.0

    def __enter__(self) -> "PerformanceTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        _tb: object,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
        if exc_type is not None:
            self._log.debug(
                "%s 中断（%.3f s）：%s", self._label, self.elapsed, exc_val
            )
            return  # 不抑制异常
        if self._threshold_warn is not None and self.elapsed > self._threshold_warn:
            self._log.warning(
                "%s 耗时过长：%.3f s（阈值 %.1f s）",
                self._label, self.elapsed, self._threshold_warn,
            )
        else:
            self._log.log(
                self._level, "%s 完成，耗时 %.3f s", self._label, self.elapsed
            )


# ---------------------------------------------------------------------------
# ProgressLogger
# ---------------------------------------------------------------------------

class ProgressLogger:
    """
    优化迭代进度专用日志器。

    功能
    ----
    - 格式化输出每次迭代的参数、目标值、状态
    - 追踪历史最优值及其迭代编号，新最优时标注 ★
    - 基于最近 N 次迭代的滑动平均估算剩余时间（ETA）
    - 输出阶段分隔线（begin）和完成摘要（end）

    用法
    ----
        prog = ProgressLogger(log, total=30, objective_name="TAC", minimize=True)
        prog.begin("贝叶斯优化阶段")

        for i, (params, result) in enumerate(iterations):
            prog.log_iteration(
                index=i + 1,
                params=params,
                value=result.objective_value,
                status=result.status,
                elapsed=result.elapsed,
            )

        prog.end()

    与 optimize_case.py 的关系
    --------------------------
    optimize_case.py 中的内联 _log.info() 调用可逐步迁移到 ProgressLogger，
    也可并行使用——ProgressLogger 只是在同一 logger 上输出更结构化的格式。
    """

    _SEP        = "─" * 60
    _ETA_WINDOW = 10  # 用最近 N 次迭代的平均耗时估算 ETA

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        objective_name: str = "objective",
        minimize: bool = True,
    ) -> None:
        """
        参数
        ----
        logger         : 日志器（通常为 logging.getLogger(__name__)）
        total          : 总迭代次数（含初始 DOE）
        objective_name : 目标函数名称，用于日志显示
        minimize       : True 表示最小化，False 表示最大化
        """
        self._log            = logger
        self._total          = total
        self._objective_name = objective_name
        self._minimize       = minimize

        self._best_value: float | None = None
        self._best_index: int | None   = None
        self._n_success                = 0
        self._n_failed                 = 0
        self._start_time: float        = 0.0
        self._last_iter_t: float       = 0.0
        self._iter_times: list[float]  = []

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def begin(self, phase_name: str = "") -> None:
        """
        记录阶段开始，打印分隔线。

        参数
        ----
        phase_name : 阶段名称（如 "初始 DOE"、"贝叶斯优化"），空字符串只打印分隔线
        """
        self._start_time  = time.perf_counter()
        self._last_iter_t = self._start_time
        if phase_name:
            inner = f"  {phase_name}  "
            pad   = max(0, (60 - len(inner)) // 2)
            self._log.info("%s%s%s", "─" * pad, inner, "─" * max(0, 60 - pad - len(inner)))
        else:
            self._log.info(self._SEP)

    def log_iteration(
        self,
        index: int,
        params: dict[str, float],
        value: float | None,
        status: str,
        elapsed: float,
        error: str | None = None,
    ) -> None:
        """
        记录单次迭代结果。

        参数
        ----
        index   : 迭代编号（从 1 开始）
        params  : 设计变量 {Aspen 树路径: 值}，路径自动截取最后一段显示
        value   : 目标函数值（None 表示失败/不可用）
        status  : 工况状态字符串（如 "success"、"sim_failed"、"infeasible"）
        elapsed : 本次工况耗时（秒）
        error   : 底层错误信息（SimulationResult.error），失败时传入以便写入日志
        """
        now = time.perf_counter()
        self._iter_times.append(now - self._last_iter_t)
        self._last_iter_t = now
        if len(self._iter_times) > self._ETA_WINDOW:
            self._iter_times.pop(0)

        if value is not None:
            self._n_success += 1
            is_best = self._update_best(value, index)
        else:
            self._n_failed += 1
            is_best = False

        val_str    = f"{value:.4g}" if value is not None else "N/A"
        best_mark  = " ★" if is_best else ""
        eta_str    = self._eta_str(index)
        param_str  = "  ".join(
            f"{k.split(chr(92))[-1]}={v:.4g}" for k, v in params.items()
        )

        self._log.info(
            "[%d/%d] %s=%s%s  status=%s  t=%.1fs  ETA=%s",
            index, self._total,
            self._objective_name, val_str, best_mark,
            status, elapsed, eta_str,
        )
        if error:
            self._log.warning("  └─ sim error: %s", error)
        self._log.debug("  params: %s", param_str)

    def end(self) -> None:
        """记录阶段完成摘要（成功率、最优值、总耗时）。"""
        total_elapsed = time.perf_counter() - self._start_time
        n_total       = self._n_success + self._n_failed
        rate          = self._n_success / n_total * 100 if n_total > 0 else 0.0

        self._log.info(self._SEP)
        self._log.info(
            "完成：%d/%d 成功（%.1f%%），总耗时 %.1f s",
            self._n_success, n_total, rate, total_elapsed,
        )
        if self._best_value is not None:
            self._log.info(
                "最优 %s = %.4g（第 %d 次迭代）",
                self._objective_name, self._best_value, self._best_index,
            )
        else:
            self._log.warning("未找到有效目标值，所有工况均失败。")
        self._log.info(self._SEP)

    @property
    def best_value(self) -> float | None:
        """当前历史最优目标值（None 表示尚无成功工况）。"""
        return self._best_value

    @property
    def best_index(self) -> int | None:
        """历史最优值对应的迭代编号（None 表示尚无成功工况）。"""
        return self._best_index

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _update_best(self, value: float, index: int) -> bool:
        """更新历史最优值，返回是否创造了新最优。"""
        if self._best_value is None:
            self._best_value = value
            self._best_index = index
            return True
        if self._minimize and value < self._best_value:
            self._best_value = value
            self._best_index = index
            return True
        if not self._minimize and value > self._best_value:
            self._best_value = value
            self._best_index = index
            return True
        return False

    def _eta_str(self, current_index: int) -> str:
        """基于最近 N 次迭代的平均耗时估算剩余时间。"""
        remaining = self._total - current_index
        if remaining <= 0 or not self._iter_times:
            return "done"
        avg     = sum(self._iter_times) / len(self._iter_times)
        eta_sec = avg * remaining
        if eta_sec < 60:
            return f"{eta_sec:.0f}s"
        if eta_sec < 3600:
            return f"{eta_sec / 60:.1f}min"
        return f"{eta_sec / 3600:.1f}h"


# ---------------------------------------------------------------------------
# suppress_loggers
# ---------------------------------------------------------------------------

def suppress_loggers(*names: str, level: int = logging.WARNING) -> None:
    """
    将指定 logger 的级别设置为 level（默认 WARNING），用于静默第三方噪声。

    setup_logging() 已自动静默常见的噪声库（skopt、sklearn 等），
    此函数用于按需追加静默其他 logger。

    用法
    ----
        suppress_loggers("skopt", "sklearn.gaussian_process")
        suppress_loggers("matplotlib", level=logging.ERROR)
    """
    for name in names:
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# section()：阶段分隔线上下文管理器
# ---------------------------------------------------------------------------

@contextmanager
def section(
    logger: logging.Logger,
    title: str,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """
    上下文管理器，在代码块前后打印带标题的分隔线，并记录耗时。

    用法
    ----
        with section(log, "初始 DOE 阶段"):
            for i in range(n_initial):
                run_case(...)
        # 输出：
        # ──────────────  初始 DOE 阶段  ──────────────
        # ... （阶段内日志）
        # ── 初始 DOE 阶段 完成，耗时 45.2 s ──────────

    参数
    ----
    logger : 日志器
    title  : 阶段标题
    level  : 日志级别（默认 INFO）
    """
    inner = f"  {title}  "
    pad   = max(0, (60 - len(inner)) // 2)
    logger.log(level, "%s%s%s", "─" * pad, inner, "─" * max(0, 60 - pad - len(inner)))
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        tail    = max(0, 50 - len(title))
        logger.log(level, "── %s 完成，耗时 %.1f s %s", title, elapsed, "─" * tail)
