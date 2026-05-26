"""
node_catalog.py — Aspen 节点目录（catalog）和语义适配层的数据模型。

CatalogEntry   : 单个 Aspen 树节点的发现记录（catalog scan 产出）。
CatalogScan    : 一次 catalog 扫描的元数据（绑定到 Aspen 文件）。
SemanticField  : 单个语义字段的读取结果（manifest runtime 产出）。
SemanticBlock  : 单个 block 的全部语义字段集合。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    """
    Aspen 树中单个节点的发现记录。

    由 CatalogScanner 扫描产出，写入 NodeDB.node_catalog 表。
    abs_path 是全局唯一键（同一 catalog_id 内）。
    """
    catalog_id: str          # 所属扫描批次 ID
    abs_path: str            # 绝对路径，如 \Data\Blocks\T1\Output\REB_DUTY
    rel_path: str            # 相对于扫描根节点的路径，如 Output\REB_DUTY
    parent_path: str         # 父节点绝对路径
    depth: int               # 相对于扫描根节点的深度（根=0）
    name: str                # 节点名称（路径最后一段）
    block_name: str          # 所属 block 名称，如 T1；stream 节点为 ""
    block_type: str          # HAP_RECORDTYPE，如 RADFRAC；stream 节点为 MATERIAL
    stream_name: str         # 所属 stream 名称；block 节点为 ""
    is_leaf: bool            # True 表示叶节点（无子节点）
    has_children: bool       # True 表示有子节点
    value_type: int          # IHNode.ValueType (0-5)
    unit_string: str         # IHNode.UnitString；无单位时为 ""
    dimension: int           # IHNode.Dimension；0=标量
    sample_value: Any        # 扫描时读取的样本值；读取失败时为 None
    sample_error: str        # 读取失败原因；成功时为 ""
    cached_at: str           # ISO 8601 时间戳


@dataclass
class CatalogScan:
    """
    一次 catalog 扫描的元数据，绑定到特定 Aspen 文件。

    catalog_id 是 node_catalog 表的外键，也是 read_manifests 的关联键。
    """
    catalog_id: str          # UUID，全局唯一
    aspen_file_path: str     # Aspen 文件绝对路径
    aspen_file_hash: str     # 文件 MD5/SHA256 摘要（用于检测文件变更）
    aspen_version: str       # Aspen Plus 版本字符串（如可获取）
    n_blocks: int            # 扫描到的 block 数量
    n_streams: int           # 扫描到的 stream 数量
    n_entries: int           # 扫描到的节点总数
    scan_depth: int          # 扫描时使用的 max_depth
    created_at: str          # ISO 8601 时间戳
    notes: str = ""          # 扫描备注（如失败节点数量）


# ---------------------------------------------------------------------------
# Semantic fields（manifest runtime 产出）
# ---------------------------------------------------------------------------

@dataclass
class SemanticField:
    """
    单个语义字段的读取结果。

    由 manifest runtime reader 产出，存储在 SemanticBlock 中。
    available=True 且 error="" 时值可信。
    """
    field_name: str          # 语义字段名，如 reboiler_duty
    abs_path: str            # 实际读取的 Aspen 绝对路径
    value: Any               # 读取到的原始值；失败时为 None
    unit: str                # IHNode.UnitString；无单位时为 ""
    value_type: int          # IHNode.ValueType
    available: bool          # True 表示读取成功且值非 None
    error: str               # 失败原因；成功时为 ""
    required: bool           # 是否为 required 字段（来自 manifest item）
    rule_id: str             # 匹配的规则 ID，用于诊断


@dataclass
class SemanticBlock:
    """
    单个 block 的全部语义字段集合。

    由 manifest runtime reader 产出，挂载到 ProcessCase.semantic_blocks。
    is_complete=True 表示所有 required 字段均可用。
    """
    block_name: str
    block_type: str
    fields: dict[str, SemanticField] = field(default_factory=dict)
    is_complete: bool = False        # 所有 required 字段均 available
    missing_required: list[str] = field(default_factory=list)  # 缺失的 required 字段名
    manifest_id: str = ""

    def get(self, field_name: str) -> SemanticField | None:
        """按语义字段名查找，不存在时返回 None。"""
        return self.fields.get(field_name)

    def get_value(self, field_name: str) -> Any:
        """返回字段值；字段不存在或不可用时返回 None。"""
        f = self.fields.get(field_name)
        return f.value if (f is not None and f.available) else None

    def get_unit(self, field_name: str) -> str:
        """返回字段单位；字段不存在时返回 ""。"""
        f = self.fields.get(field_name)
        return f.unit if f is not None else ""
