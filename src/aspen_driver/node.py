"""
node.py — Aspen Plus 树节点的高级封装。

基于 Aspen Plus User Guide 10.2 第 38 章 IHNode / IHNodeCol 接口规范实现。

职责：在 driver.py 的原始 COM 节点之上提供类型感知的读写、
子节点遍历、以及批量操作。不持有 COM 连接，所有操作通过
AspenDriver 实例委托执行。

HAP 常量加载时机
-----------------
HAPAttributeNumber 枚举值定义在 Aspen Plus type library（happ.tlb）中，
手册只给出名称，不给出整数值。

正确的加载顺序：
    1. 调用 AspenDriver.connect()（内部使用 EnsureDispatch 填充 gencache）
    2. AspenDriver 持有已验证的 hap_constants 字典
    3. AspenNode 从 driver.hap_constants 获取常量，不在 import 时固化

如果 driver 未连接或常量未验证，info() 会抛出 AspenNodeError，
而不是静默使用未验证的回退值。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterator

from .errors import AspenNodeError, AspenNodeValueError

if TYPE_CHECKING:
    from .driver import AspenDriver

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValueType 枚举（手册 38-10 / 38-40）
# 这些值由手册明确列出，可信，在 import 时固化是安全的。
# ---------------------------------------------------------------------------
class ValueType(IntEnum):
    UNDEFINED    = 0
    INTEGER      = 1
    REAL         = 2
    STRING       = 3
    NODE         = 4
    MEMORY_BLOCK = 5


# ---------------------------------------------------------------------------
# HAP 常量名称列表（仅名称，不含整数值）
# 整数值由 AspenDriver.connect() 从 type library 加载后存入 driver.hap_constants。
# ---------------------------------------------------------------------------
HAP_NAMES = (
    # 节点属性
    "HAP_VALUE",
    "HAP_UNITROW",
    "HAP_UNITCOL",
    "HAP_BASIS",
    "HAP_OPTIONLIST",
    "HAP_RECORDTYPE",
    "HAP_COMPSTATUS",
    "HAP_OUTVAR",
    "HAP_ENTERABLE",
    "HAP_UPPERLIMIT",
    "HAP_LOWERLIMIT",
    "HAP_VALUEDEFAULT",
    "HAP_PROMPT",
    "HAP_FIRSTPAIR",
    "HAP_INOUT",
    "HAP_PORTSEX",
    "HAP_MULTIPORT",
    "HAP_PORTTYPE",
    "HAP_HASCHILDREN",
    "HAP_SECTION",
    # HAP_COMPSTATUS 位掩码（手册 38-13）
    "HAP_RESULTS_SUCCESS",
    "HAP_RESULTS_ERRORS",
    "HAP_RESULTS_WARNINGS",
    "HAP_NORESULTS",
    "HAP_RESULTS_INCOMPAT",
    "HAP_RESULTS_INACCESS",
)


@dataclass
class NodeInfo:
    """节点的静态元数据快照（仅包含手册明确支持的字段）。"""

    path: str
    name: str
    value: Any
    unit_string: str        # IHNode.UnitString
    value_type: int         # IHNode.ValueType (0-5)
    dimension: int          # IHNode.Dimension (0=标量叶节点)
    is_output: bool         # HAP_OUTVAR: 是否为只读结果变量
    is_enterable: bool      # HAP_ENTERABLE: 值是否可修改
    record_type: str        # HAP_RECORDTYPE: 如 "RADFRAC"/"MATERIAL"
    has_children: bool      # HAP_HASCHILDREN
    children: list[str] = field(default_factory=list)


class AspenNode:
    """
    对单个 Aspen Plus 树节点的高级封装。

    通过 AspenDriver 访问底层 COM 对象，提供：
    - 类型感知的值读写（ValueType）
    - 单位字符串读取（UnitString，手册 38-14）
    - 子节点枚举（Elements / IHNodeCol，手册 38-8/38-41）
    - 批量子节点值读取

    info() 依赖 driver.hap_constants，需要在 driver.connect() 之后调用。
    """

    def __init__(self, driver: AspenDriver, path: str) -> None:
        self._driver = driver
        self._path = path

    # ------------------------------------------------------------------ #
    # 基本属性
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> str:
        return self._path

    @property
    def name(self) -> str:
        return self._path.rsplit("\\", 1)[-1]

    @property
    def exists(self) -> bool:
        return self._driver.node_exists(self._path)

    # ------------------------------------------------------------------ #
    # 值读写
    # ------------------------------------------------------------------ #

    @property
    def value(self) -> Any:
        return self._driver.get_value(self._path)

    @value.setter
    def value(self, new_value: Any) -> None:
        self._driver.set_value(self._path, new_value)

    def get_float(self) -> float:
        """读取节点值并强制转换为 float，适用于 ValueType.REAL 节点。"""
        raw = self.value
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise AspenNodeValueError(self._path, raw, "float") from exc

    def get_int(self) -> int:
        """读取节点值并强制转换为 int，适用于 ValueType.INTEGER 节点。"""
        raw = self.value
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise AspenNodeValueError(self._path, raw, "int") from exc

    def get_str(self) -> str:
        """读取节点值并转换为字符串，适用于 ValueType.STRING 节点。"""
        raw = self.value
        return "" if raw is None else str(raw)

    @property
    def value_type(self) -> ValueType:
        """返回节点的 ValueType（手册 38-10）。"""
        com_node = self._raw_com_node()
        try:
            return ValueType(int(com_node.ValueType))
        except Exception as exc:
            raise AspenNodeError(
                f"读取节点 '{self._path}' 的 ValueType 失败：{exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # 单位（手册 38-14：UnitString 属性）
    # ------------------------------------------------------------------ #

    def get_unit(self) -> str:
        """
        返回节点的工程单位字符串（IHNode.UnitString，手册 38-14/38-41）。
        若节点无单位则返回空字符串。
        """
        com_node = self._raw_com_node()
        try:
            unit = com_node.UnitString
            return str(unit) if unit is not None else ""
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    # 属性访问（AttributeValue 接受整数编号，手册 38-11/38-40）
    # ------------------------------------------------------------------ #

    def get_attribute(self, attr_number: int) -> Any:
        """
        读取 COM 节点的 AttributeValue。

        Parameters
        ----------
        attr_number:
            HAPAttributeNumber 整数编号。
            使用 driver.hap_constants["HAP_OUTVAR"] 等方式获取已验证的编号。
        """
        com_node = self._raw_com_node()
        try:
            return com_node.AttributeValue(attr_number)
        except Exception as exc:
            raise AspenNodeError(
                f"读取节点 '{self._path}' 属性编号 {attr_number} 失败：{exc}"
            ) from exc

    def has_attribute(self, attr_number: int) -> bool:
        """检查节点是否支持指定属性（IHNode.HasAttribute，手册 38-40）。"""
        com_node = self._raw_com_node()
        try:
            return bool(com_node.HasAttribute(attr_number))
        except Exception:
            return False

    @property
    def dimension(self) -> int:
        """
        返回节点的 Dimension（手册 38-8/38-39）。
        0 = 标量叶节点；>0 = 多维变量节点，值为维度数。
        """
        com_node = self._raw_com_node()
        try:
            return int(com_node.Dimension)
        except Exception as exc:
            raise AspenNodeError(
                f"读取节点 '{self._path}' 的 Dimension 失败：{exc}"
            ) from exc

    def info(self) -> NodeInfo:
        """
        返回节点的完整元数据快照。

        依赖 driver.hap_constants（由 driver.connect() 从 type library 加载）。
        若常量未验证，抛出 AspenNodeError，不使用未验证的回退值。
        """
        hap = self._driver.hap_constants
        if hap is None:
            raise AspenNodeError(
                f"无法获取节点 '{self._path}' 的元数据：HAP 常量未加载。"
                "请确保 driver.connect() 已成功调用，且 EnsureDispatch 未回退到 Dispatch。"
                "运行 scripts/verify_hap_constants.py 诊断。"
            )

        com_node = self._raw_com_node()

        def _attr(name: str, default: Any = None) -> Any:
            num = hap.get(name)
            if num is None:
                return default
            try:
                return com_node.AttributeValue(num)
            except Exception:
                return default

        def _unit() -> str:
            try:
                u = com_node.UnitString
                return str(u) if u is not None else ""
            except Exception:
                return ""

        def _vtype() -> int:
            try:
                return int(com_node.ValueType)
            except Exception:
                return 0

        def _dim() -> int:
            try:
                return int(com_node.Dimension)
            except Exception:
                return 0

        return NodeInfo(
            path=self._path,
            name=self.name,
            value=self.value,
            unit_string=_unit(),
            value_type=_vtype(),
            dimension=_dim(),
            is_output=bool(_attr("HAP_OUTVAR", False)),
            is_enterable=bool(_attr("HAP_ENTERABLE", False)),
            record_type=str(_attr("HAP_RECORDTYPE", "") or ""),
            has_children=bool(_attr("HAP_HASCHILDREN", False)),
            children=self.child_names(),
        )

    # ------------------------------------------------------------------ #
    # 子节点遍历（IHNodeCol，手册 38-8/38-41/38-42）
    # ------------------------------------------------------------------ #

    def child_names(self) -> list[str]:
        """
        返回直接子节点的名称列表。

        Dimension=0 表示标量叶节点，直接返回 []，不调用 Elements
        （Aspen COM 对叶节点调用 Elements 会抛出错误码 2010）。

        对 Dimension>0 的节点，三层 fallback（手册 38-41/38-42）：
          1. COM For Each 枚举（IHNodeCol 迭代）
          2. Count + Item(index).Name（IHNodeCol.Count / Item）
          3. RowCount(0) + ItemName(i, 0)（多维集合接口）

        三层均失败时抛出 AspenNodeError，不静默返回空列表。
        """
        com_node = self._raw_com_node()

        # Dimension=0：标量叶节点，无子节点，不能调 Elements
        try:
            if int(com_node.Dimension) == 0:
                return []
        except Exception:
            pass  # Dimension 读取失败时继续尝试 Elements

        try:
            elements = com_node.Elements
        except Exception as exc:
            raise AspenNodeError(
                f"获取节点 '{self._path}' 的 Elements 失败：{exc}"
            ) from exc

        if elements is None:
            return []

        # 层 1：COM For Each 枚举
        try:
            return [e.Name for e in elements]
        except Exception:
            pass

        # 层 2：Count + Item(index).Name（手册 38-41）
        try:
            count = int(elements.Count)
            return [str(elements.Item(i).Name) for i in range(count)]
        except Exception:
            pass

        # 层 3：RowCount(0) + ItemName(i, 0)（手册 38-42，多维集合）
        try:
            count = int(elements.RowCount(0))
            return [str(elements.ItemName(i, 0)) for i in range(count)]
        except Exception as exc:
            raise AspenNodeError(
                f"枚举节点 '{self._path}' 的子节点失败"
                f"（For Each、Count/Item、RowCount/ItemName 均不可用）：{exc}"
            ) from exc

    def children(self) -> list[AspenNode]:
        """返回所有直接子节点的 AspenNode 列表。"""
        return [
            AspenNode(self._driver, f"{self._path}\\{name}")
            for name in self.child_names()
        ]

    def iter_children(self) -> Iterator[AspenNode]:
        """惰性迭代所有直接子节点。"""
        for name in self.child_names():
            yield AspenNode(self._driver, f"{self._path}\\{name}")

    def child(self, name: str) -> AspenNode:
        """
        按名称获取直接子节点。

        优先通过 Elements.Item(name) 获取（手册 38-8/38-41），
        避免仅靠路径拼接而跳过 COM 验证。
        """
        com_node = self._raw_com_node()
        try:
            elements = com_node.Elements
            if elements is not None:
                child_com = elements.Item(name)
                if child_com is not None:
                    return AspenNode(self._driver, f"{self._path}\\{name}")
        except Exception:
            pass

        child_path = f"{self._path}\\{name}"
        if not self._driver.node_exists(child_path):
            raise AspenNodeError(f"子节点不存在：'{child_path}'")
        return AspenNode(self._driver, child_path)

    # ------------------------------------------------------------------ #
    # 批量操作
    # ------------------------------------------------------------------ #

    def read_children_values(
        self,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        读取所有直接子节点的值。

        Parameters
        ----------
        strict:
            True（默认）：任意子节点读取失败时抛出聚合 AspenNodeError。
                适用于优化循环、数据库写入等不能容忍脏数据的场景。
            False：容错模式，返回 {name: {"value": ..., "error": str|None}}。
                "error" 为 None 表示读取成功；非 None 表示失败原因。
                调用方必须检查 "error" 字段，不能直接使用 "value"。

        Returns
        -------
        strict=True:  {name: value}
        strict=False: {name: {"value": Any | None, "error": str | None}}
        """
        if strict:
            result: dict[str, Any] = {}
            errors: list[str] = []
            for node in self.iter_children():
                try:
                    result[node.name] = node.value
                except AspenNodeError as exc:
                    errors.append(f"  {node.name}: {exc}")
            if errors:
                raise AspenNodeError(
                    f"读取 '{self._path}' 的子节点值时发生 {len(errors)} 个错误：\n"
                    + "\n".join(errors)
                )
            return result
        else:
            tolerant: dict[str, dict[str, Any]] = {}
            for node in self.iter_children():
                try:
                    tolerant[node.name] = {"value": node.value, "error": None}
                except AspenNodeError as exc:
                    tolerant[node.name] = {"value": None, "error": str(exc)}
            return tolerant  # type: ignore[return-value]

    def write_values(self, values: dict[str, Any]) -> None:
        """
        批量写入子节点值。

        Parameters
        ----------
        values:
            {子节点名称: 值} 字典，键为相对于当前节点的直接子节点名。
        """
        errors: list[str] = []
        for name, val in values.items():
            child_path = f"{self._path}\\{name}"
            try:
                self._driver.set_value(child_path, val)
            except AspenNodeError as exc:
                errors.append(str(exc))
        if errors:
            raise AspenNodeError(
                f"批量写入时发生 {len(errors)} 个错误：\n" + "\n".join(errors)
            )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _raw_com_node(self) -> Any:
        """返回底层 COM 节点对象（IHNode）。"""
        return self._driver.get_node(self._path)

    def __repr__(self) -> str:
        return f"AspenNode(path={self._path!r})"
