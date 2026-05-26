"""
read_manifest.py — ReadManifest 和 ReadManifestItem 数据模型。

ReadManifest     : 一次 manifest 构建的元数据（绑定到 catalog + objective_names）。
ReadManifestItem : manifest 中单条读取项（block/stream/global 节点 → 语义字段映射）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReadManifestItem:
    """
    manifest 中单条读取项，描述"从哪个路径读取什么语义字段"。

    source_type: "block" / "stream" / "global"
    semantic_field: 语义字段名（如 reboiler_duty）；
                    约束路径用 "constraint:<name>"；
                    显式 output_paths 用 "output:<path>"。
    required: True 时若读取失败则 manifest 整体标记 invalid。
    confidence: 0.0-1.0，规则匹配置信度（priority 归一化后）。
    """
    manifest_id: str
    source_type: str          # block / stream / global
    source_name: str          # block 名称（如 T1）或 stream 名称
    equipment_type: str       # RADFRAC / HEATX / MATERIAL 等
    semantic_field: str       # 语义字段名
    abs_path: str             # Aspen 绝对路径
    rel_path: str             # 相对路径（相对于 block/stream 根）
    unit_string: str          # catalog 中记录的 UnitString
    value_type: int           # catalog 中记录的 ValueType
    required: bool            # 是否必须
    confidence: float         # 匹配置信度 0.0-1.0
    rule_id: str              # 匹配的规则 ID（如 radfrac.reboiler_duty.0）
    error: str                # 构建时的错误信息；成功时为 ""


@dataclass
class ReadManifest:
    """
    一次 manifest 构建的完整结果。

    is_valid=False 时不应进行正式优化（required 字段缺失或单位不合规）。
    items 列表按 source_name + semantic_field 排序。
    """
    manifest_id: str
    catalog_id: str
    objective_names: list[str]        # 触发此 manifest 的目标函数名称列表
    items: list[ReadManifestItem] = field(default_factory=list)
    is_valid: bool = True
    error: str = ""                   # manifest 级别的错误（如 required 字段缺失）
    created_at: str = ""

    def get_items_for_block(self, block_name: str) -> list[ReadManifestItem]:
        """返回指定 block 的所有 manifest items。"""
        return [i for i in self.items if i.source_name == block_name]

    def get_item(self, source_name: str, semantic_field: str) -> ReadManifestItem | None:
        """按 source_name + semantic_field 查找 item。"""
        for item in self.items:
            if item.source_name == source_name and item.semantic_field == semantic_field:
                return item
        return None

    def required_items(self) -> list[ReadManifestItem]:
        """返回所有 required=True 的 items。"""
        return [i for i in self.items if i.required]

    def missing_required(self) -> list[ReadManifestItem]:
        """返回所有 required=True 且 error 非空的 items（构建时未找到路径）。"""
        return [i for i in self.items if i.required and i.error]
