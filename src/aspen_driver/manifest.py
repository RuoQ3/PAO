"""
manifest.py — Aspen 语义适配层：从 catalog + semantic rules 生成 ReadManifest。

职责
----
1. 加载 configs/aspen_semantics/*.yaml 中的语义规则。
2. 从 NodeDB catalog 中识别 block_name → block_type 映射。
3. 对每个 block_type 查找对应规则，按 objective_names 筛选 required_for 字段。
4. 用 glob pattern 在 catalog 路径中匹配候选节点。
5. 用 priority + validators 选最佳路径，生成 ReadManifestItem。
6. 对 constraints/output_paths 直接加入 manifest。
7. required 字段缺失时 manifest.is_valid=False。
8. 将 manifest 写入 NodeDB。

典型用法
--------
    from src.aspen_driver.manifest import ManifestBuilder

    builder = ManifestBuilder(
        node_db=db,
        rules_dir="configs/aspen_semantics",
    )
    manifest = builder.build(
        catalog_id=scan.catalog_id,
        objective_names=["TAC", "EMISSIONS"],
        extra_paths=config.output_paths,
    )
    if not manifest.is_valid:
        raise RuntimeError(f"manifest 无效：{manifest.error}")
"""
from __future__ import annotations

import fnmatch
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.read_manifest import ReadManifest, ReadManifestItem

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_log = logging.getLogger(__name__)

# 单位维度 → 可接受的 UnitString 关键词（大小写不敏感子串匹配）
_UNIT_DIMENSION_KEYWORDS: dict[str, list[str]] = {
    "energy_rate": ["gj", "mj", "kj", "j/", "w", "btu", "cal", "kcal", "mw", "kw"],
    "length":      ["m", "ft", "in", "cm", "mm"],
    "area":        ["m2", "m²", "ft2", "sqm", "sqft"],
    "power":       ["kw", "mw", "w", "hp", "btu/hr"],
    "temperature": ["c", "k", "f", "°"],
    "pressure":    ["pa", "bar", "atm", "psi", "kpa", "mpa"],
    "dimensionless": [""],   # 空单位或无单位均可
    "any":         [],       # 不校验
}


# ---------------------------------------------------------------------------
# 规则加载
# ---------------------------------------------------------------------------

def load_semantic_rules(rules_dir: str | Path) -> dict[str, dict[str, Any]]:
    """
    加载 rules_dir 下所有 *.yaml 文件，返回 {equipment_type: rule_dict} 映射。

    rule_dict 结构：
        {
          "equipment_type": "RADFRAC",
          "fields": {
            "reboiler_duty": {
              "required_for": ["TAC", "EMISSIONS"],
              "required": True,
              "candidates": [{"pattern": "Output\\REB_DUTY", "priority": 100}, ...],
              "validators": {"value_type": "numeric", "unit_dimension": "energy_rate"},
            },
            ...
          }
        }

    YAML 不可用时返回空字典并记录 WARNING。
    """
    if not _YAML_AVAILABLE:
        _log.warning("PyYAML 未安装，无法加载语义规则。请 pip install pyyaml。")
        return {}

    rules_path = Path(rules_dir)
    if not rules_path.exists():
        _log.warning("语义规则目录不存在：%s", rules_path)
        return {}

    result: dict[str, dict[str, Any]] = {}
    for yaml_file in sorted(rules_path.glob("*.yaml")):
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = _yaml.safe_load(f)
            if not isinstance(data, dict):
                _log.warning("规则文件格式错误（非 dict）：%s", yaml_file)
                continue
            eq_type = data.get("equipment_type", "").upper()
            if not eq_type:
                _log.warning("规则文件缺少 equipment_type：%s", yaml_file)
                continue
            result[eq_type] = data
            _log.debug("加载语义规则：%s → %s", yaml_file.name, eq_type)
        except Exception as exc:
            _log.warning("加载规则文件失败（%s）：%s", yaml_file, exc)

    _log.info("加载语义规则 %d 个：%s", len(result), list(result.keys()))
    return result


# ---------------------------------------------------------------------------
# ManifestBuilder
# ---------------------------------------------------------------------------

class ManifestBuilder:
    """
    从 NodeDB catalog + semantic rules 构建 ReadManifest。

    Parameters
    ----------
    node_db:
        NodeDB 实例，提供 catalog 数据，manifest 写入此数据库。
    rules_dir:
        语义规则 YAML 目录路径，默认 "configs/aspen_semantics"。
    """

    def __init__(
        self,
        node_db: Any,
        rules_dir: str | Path = "configs/aspen_semantics",
    ) -> None:
        self._node_db  = node_db
        self._rules    = load_semantic_rules(rules_dir)

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def build(
        self,
        catalog_id: str,
        objective_names: list[str],
        *,
        extra_paths: list[str] | None = None,
        manifest_id: str | None = None,
    ) -> ReadManifest:
        """
        构建 ReadManifest 并写入 NodeDB。

        Parameters
        ----------
        catalog_id:
            目标 catalog ID（来自 CatalogScan.catalog_id）。
        objective_names:
            当前优化目标名称列表，如 ["TAC", "EMISSIONS"]。
            用于筛选 required_for 字段。
        extra_paths:
            额外需要读取的 Aspen 绝对路径（如 output_paths、constraint paths）。
            这些路径直接加入 manifest，semantic_field 为 "output:<path>"。
        manifest_id:
            指定 manifest_id；None 时自动生成 UUID。

        Returns
        -------
        ReadManifest
            已写入 NodeDB。manifest.is_valid=False 时包含 error 说明。
        """
        mid  = manifest_id or str(uuid.uuid4())
        now  = datetime.now(timezone.utc).isoformat()
        items: list[ReadManifestItem] = []
        errors: list[str] = []

        # 获取 catalog 中所有 block 的类型映射
        block_types = self._node_db.get_catalog_block_types(catalog_id)
        if not block_types:
            _log.warning("catalog '%s' 中未找到任何 block，manifest 可能为空", catalog_id)

        # 为每个 block 生成 manifest items
        for block_name, block_type in block_types.items():
            rule = self._rules.get(block_type.upper())
            if rule is None:
                _log.debug("block '%s'（%s）无对应语义规则，跳过", block_name, block_type)
                continue

            block_items, block_errors = self._build_block_items(
                mid, catalog_id, block_name, block_type, rule, objective_names
            )
            items.extend(block_items)
            errors.extend(block_errors)

        # 加入 extra_paths（output_paths / constraint paths）
        if extra_paths:
            for path in extra_paths:
                items.append(ReadManifestItem(
                    manifest_id=mid,
                    source_type="global",
                    source_name="",
                    equipment_type="",
                    semantic_field=f"output:{path}",
                    abs_path=path,
                    rel_path=path,
                    unit_string="",
                    value_type=0,
                    required=True,
                    confidence=1.0,
                    rule_id="explicit_path",
                    error="",
                ))

        # 判断 manifest 是否有效
        missing = [i for i in items if i.required and i.error]
        is_valid = len(missing) == 0
        error_msg = ""
        if missing:
            details = "; ".join(
                f"{i.source_name}.{i.semantic_field}（{i.error}）"
                for i in missing[:5]
            )
            if len(missing) > 5:
                details += f"（共 {len(missing)} 个）"
            error_msg = f"required 字段缺失：{details}"
            _log.warning("manifest '%s' 无效：%s", mid, error_msg)

        manifest = ReadManifest(
            manifest_id=mid,
            catalog_id=catalog_id,
            objective_names=objective_names,
            items=items,
            is_valid=is_valid,
            error=error_msg,
            created_at=now,
        )

        self._node_db.save_manifest(manifest)
        _log.info(
            "manifest 构建完成：manifest_id=%s，%d items，valid=%s",
            mid, len(items), is_valid,
        )
        return manifest

    # ------------------------------------------------------------------ #
    # 内部：为单个 block 生成 items
    # ------------------------------------------------------------------ #

    def _build_block_items(
        self,
        manifest_id: str,
        catalog_id: str,
        block_name: str,
        block_type: str,
        rule: dict[str, Any],
        objective_names: list[str],
    ) -> tuple[list[ReadManifestItem], list[str]]:
        """为单个 block 的所有语义字段生成 ReadManifestItem 列表。"""
        items: list[ReadManifestItem] = []
        errors: list[str] = []
        fields: dict[str, Any] = rule.get("fields", {})

        # 获取该 block 的所有 catalog 节点（abs_path → entry dict）
        catalog_entries = self._node_db.get_catalog_entries(
            catalog_id, block_name=block_name
        )
        path_index = {e["abs_path"].upper(): e for e in catalog_entries}

        for field_name, field_def in fields.items():
            required_for: list[str] = field_def.get("required_for", [])
            is_required: bool = field_def.get("required", False)

            # 判断此字段是否需要读取
            if required_for and not any(obj in required_for for obj in objective_names):
                continue  # 当前目标不需要此字段

            candidates: list[dict[str, Any]] = field_def.get("candidates", [])
            validators: dict[str, Any] = field_def.get("validators", {})

            # 按 priority 降序排列候选
            sorted_candidates = sorted(
                candidates, key=lambda c: c.get("priority", 0), reverse=True
            )

            best_item: ReadManifestItem | None = None
            for idx, cand in enumerate(sorted_candidates):
                pattern = cand.get("pattern", "")
                priority = cand.get("priority", 0)
                rule_id = f"{block_type.lower()}.{field_name}.{idx}"

                matched = self._match_pattern(
                    block_name, pattern, path_index, catalog_id
                )
                if matched is None:
                    continue

                entry = matched
                # 校验
                val_err = self._validate_entry(entry, validators)
                if val_err:
                    _log.debug(
                        "block '%s' 字段 '%s' 候选 '%s' 校验失败：%s",
                        block_name, field_name, pattern, val_err,
                    )
                    continue

                confidence = min(1.0, priority / 100.0)
                best_item = ReadManifestItem(
                    manifest_id=manifest_id,
                    source_type="block",
                    source_name=block_name,
                    equipment_type=block_type,
                    semantic_field=field_name,
                    abs_path=entry["abs_path"],
                    rel_path=entry["rel_path"],
                    unit_string=entry.get("unit_string", ""),
                    value_type=entry.get("value_type", 0),
                    required=is_required,
                    confidence=confidence,
                    rule_id=rule_id,
                    error="",
                )
                break  # 找到最高优先级匹配，停止

            if best_item is None:
                # 未找到匹配路径
                err_msg = (
                    f"block '{block_name}'（{block_type}）字段 '{field_name}' "
                    f"在 catalog 中未找到匹配路径（尝试了 {len(sorted_candidates)} 个候选）"
                )
                if is_required:
                    errors.append(err_msg)
                    _log.warning(err_msg)
                else:
                    _log.debug(err_msg)
                # 仍然写入一条 error item，便于诊断
                items.append(ReadManifestItem(
                    manifest_id=manifest_id,
                    source_type="block",
                    source_name=block_name,
                    equipment_type=block_type,
                    semantic_field=field_name,
                    abs_path="",
                    rel_path="",
                    unit_string="",
                    value_type=0,
                    required=is_required,
                    confidence=0.0,
                    rule_id=f"{block_type.lower()}.{field_name}.missing",
                    error=err_msg,
                ))
            else:
                items.append(best_item)

        return items, errors

    # ------------------------------------------------------------------ #
    # 内部：pattern 匹配
    # ------------------------------------------------------------------ #

    def _match_pattern(
        self,
        block_name: str,
        pattern: str,
        path_index: dict[str, dict[str, Any]],
        catalog_id: str,
    ) -> dict[str, Any] | None:
        """
        在 path_index 中查找与 pattern 匹配的 catalog entry。

        pattern 是相对于 block 根节点的路径（如 "Output\\REB_DUTY"）。
        支持 glob 通配符（* 匹配单段，** 匹配多段）。
        返回优先级最高的匹配 entry，无匹配时返回 None。
        """
        block_root = f"\\DATA\\BLOCKS\\{block_name.upper()}"
        # 构造完整 abs_path 模式（大写，用于匹配 path_index 的 key）
        full_pattern = f"{block_root}\\{pattern.upper()}"

        # 精确匹配优先
        if full_pattern in path_index:
            return path_index[full_pattern]

        # glob 匹配（fnmatch，* 不跨 \）
        matches: list[dict[str, Any]] = []
        for abs_path_upper, entry in path_index.items():
            if fnmatch.fnmatch(abs_path_upper, full_pattern):
                matches.append(entry)

        if not matches:
            return None

        # 多个匹配时，优先选叶节点，再按路径长度升序（最短路径 = 最直接）
        matches.sort(key=lambda e: (not e.get("is_leaf", True), len(e["abs_path"])))
        return matches[0]

    # ------------------------------------------------------------------ #
    # 内部：validators
    # ------------------------------------------------------------------ #

    def _validate_entry(
        self,
        entry: dict[str, Any],
        validators: dict[str, Any],
    ) -> str:
        """
        校验 catalog entry 是否满足 validators 要求。

        返回错误信息字符串；通过时返回 ""。
        """
        vtype_rule = validators.get("value_type", "any")
        udim_rule  = validators.get("unit_dimension", "any")

        # value_type 校验
        if vtype_rule == "numeric":
            vt = entry.get("value_type", 0)
            if vt not in (1, 2):  # INTEGER=1, REAL=2
                return f"value_type={vt} 不是 numeric（期望 1 或 2）"
        elif vtype_rule == "string":
            if entry.get("value_type", 0) != 3:
                return f"value_type={entry.get('value_type')} 不是 string（期望 3）"

        # unit_dimension 校验
        if udim_rule and udim_rule != "any":
            unit_str = (entry.get("unit_string") or "").lower()
            keywords = _UNIT_DIMENSION_KEYWORDS.get(udim_rule, [])
            if udim_rule == "dimensionless":
                # 无单位或空单位均可
                pass
            elif keywords:
                if not any(kw in unit_str for kw in keywords if kw):
                    # 单位为空时不强制拒绝（Aspen 部分节点 UnitString 为空但值有效）
                    if unit_str:
                        return (
                            f"unit_string='{unit_str}' 不符合 {udim_rule} 维度要求"
                            f"（期望含 {keywords[:3]}）"
                        )
                    # unit_str 为空：记录 debug，不拒绝（由 TAC 层处理单位归一化失败）
                    _log.debug(
                        "节点 '%s' unit_string 为空，跳过 unit_dimension 校验（%s）",
                        entry.get("abs_path", ""), udim_rule,
                    )

        return ""
