"""
case_memory.py — 优化案例记忆与检索

核心功能：
---------
1. 存储成功的优化案例（问题 → 动作序列 → 结果）
2. 记录失败模式及应对策略
3. 基于相似度检索历史案例
4. 从 checkpoint 自动学习

设计原则：
---------
- 案例驱动：从历史成功经验中学习
- 失败记忆：记住典型失败及解决方案
- 轻量级存储：JSON 格式，无需外部数据库
- 增量学习：每次优化结束自动更新案例库

数据结构：
---------
- OptimizationCase: 单个优化案例（问题定义、动作序列、最终结果）
- FailurePattern: 失败模式（症状、根本原因、成功的应对策略）
- CaseMemory: 案例库管理器（存储、检索、学习）

使用：
-----
    # 加载案例库
    memory = CaseMemory("cases/memory/case_library.json")

    # 检索相似案例
    similar = memory.retrieve_similar_cases(current_problem, top_k=3)

    # 匹配失败模式
    patterns = memory.match_failure_patterns(current_state)

    # 从 checkpoint 学习
    memory.learn_from_checkpoint("cases/demo/output/checkpoint.db")
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptimizationCase:
    """
    单个优化案例。

    Attributes:
        case_id: 案例唯一标识
        problem: 问题定义（目标、约束、初始状态）
        actions: 动作序列（按时间顺序）
        final_result: 最终结果（target_verified/completed_verified/...）
        metrics: 最终指标（HV、收敛率等）
        timestamp: 创建时间戳
        tags: 标签（如 "distillation", "multi_objective"）
    """
    case_id: str
    problem: dict[str, Any]
    actions: list[dict]
    final_result: str
    metrics: dict[str, float]
    timestamp: str
    tags: list[str] = field(default_factory=list)

    def similarity_to(self, other_problem: dict) -> float:
        """
        计算与另一个问题的相似度（0-1）。

        相似度计算维度：
        1. 目标函数相似度（40%）
        2. 变量数量相似度（30%）
        3. 约束数量相似度（30%）

        Args:
            other_problem: 待比较的问题定义

        Returns:
            相似度分数（0-1，越高越相似）
        """
        score = 0.0

        # 1. 目标函数相似度（比较目标名称集合）
        self_objs = set(self.problem.get("objectives", []))
        other_objs = set(other_problem.get("objectives", []))
        if self_objs and other_objs:
            obj_sim = len(self_objs & other_objs) / len(self_objs | other_objs)
            score += 0.4 * obj_sim

        # 2. 变量数量相似度
        self_n_vars = len(self.problem.get("design_variables", []))
        other_n_vars = len(other_problem.get("design_variables", []))
        if max(self_n_vars, other_n_vars) > 0:
            var_sim = min(self_n_vars, other_n_vars) / max(self_n_vars, other_n_vars)
            score += 0.3 * var_sim

        # 3. 约束数量相似度
        self_cons = len(self.problem.get("constraints", []))
        other_cons = len(other_problem.get("constraints", []))
        if max(self_cons, other_cons) > 0:
            con_sim = min(self_cons, other_cons) / max(self_cons, other_cons)
            score += 0.3 * con_sim

        return score


@dataclass
class FailurePattern:
    """
    失败模式。

    记录典型失败场景及应对策略。

    Attributes:
        pattern_id: 模式唯一标识
        symptoms: 症状特征列表（错误关键词、参数范围等）
        root_cause: 根本原因描述
        occurrences: 出现次数
        successful_remedies: 成功的应对策略列表
        timestamp: 创建时间戳
    """
    pattern_id: str
    symptoms: list[str]
    root_cause: str
    occurrences: int
    successful_remedies: list[dict]
    timestamp: str

    def matches(self, case_state: dict) -> bool:
        """
        判断当前状态是否匹配此失败模式。

        匹配规则：至少 50% 的症状特征出现

        Args:
            case_state: 当前状态（包含 errors、parameters 等）

        Returns:
            True 表示匹配此失败模式
        """
        if not self.symptoms:
            return False

        # 提取当前状态的特征
        current_features = set()

        # 错误关键词
        errors = case_state.get("errors", [])
        for err in errors:
            current_features.add(str(err).lower())

        # 参数范围（简化：检查变量名）
        params = case_state.get("parameters", {})
        for key in params:
            current_features.add(f"param:{key}")

        # 检查症状匹配度
        matched = 0
        for symptom in self.symptoms:
            symptom_lower = symptom.lower()
            # 部分匹配：症状关键词在当前特征中出现
            if any(symptom_lower in str(feat).lower() for feat in current_features):
                matched += 1

        # 至少 50% 匹配
        return matched >= len(self.symptoms) * 0.5


class CaseMemory:
    """
    案例记忆库管理器。

    功能：
    - 存储和加载案例库（JSON 格式）
    - 检索相似案例
    - 匹配失败模式
    - 从 checkpoint 自动学习

    存储路径：
        默认 cases/memory/case_library.json
    """

    def __init__(self, storage_path: str = "cases/memory/case_library.json"):
        self.storage_path = Path(storage_path)
        self.cases: list[OptimizationCase] = []
        self.failure_patterns: list[FailurePattern] = []
        self._load()

    def _load(self):
        """从磁盘加载案例库"""
        if not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.cases = [OptimizationCase(**c) for c in data.get("cases", [])]
            self.failure_patterns = [FailurePattern(**f) for f in data.get("failure_patterns", [])]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("加载案例库失败：%s", exc)

    def save(self):
        """保存到磁盘"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "cases": [asdict(c) for c in self.cases],
            "failure_patterns": [asdict(f) for f in self.failure_patterns],
        }
        self.storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def add_case(self, case: OptimizationCase):
        """
        添加新案例。

        Args:
            case: OptimizationCase 实例
        """
        # 检查是否已存在（避免重复）
        if any(c.case_id == case.case_id for c in self.cases):
            return

        self.cases.append(case)
        self.save()

    def retrieve_similar_cases(
        self,
        problem: dict,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[tuple[OptimizationCase, float]]:
        """
        检索相似案例。

        Args:
            problem: 当前问题定义
            top_k: 返回前 k 个最相似的案例
            min_similarity: 最小相似度阈值

        Returns:
            (案例, 相似度) 元组列表，按相似度降序排列
        """
        scores = [(case, case.similarity_to(problem)) for case in self.cases]
        scores = [(c, s) for c, s in scores if s >= min_similarity]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def match_failure_patterns(self, case_state: dict) -> list[FailurePattern]:
        """
        匹配失败模式。

        Args:
            case_state: 当前状态（包含 errors、parameters 等）

        Returns:
            匹配的失败模式列表
        """
        return [p for p in self.failure_patterns if p.matches(case_state)]

    def learn_from_checkpoint(self, checkpoint_path: str):
        """
        从 checkpoint 学习（自动提取案例和失败模式）。

        实现策略：
        1. 读取 checkpoint.db（SQLite）
        2. 提取问题定义、动作序列、最终结果
        3. 如果是成功案例（target_verified/completed_verified），添加到 cases
        4. 如果有失败，聚类到 failure_patterns

        Args:
            checkpoint_path: checkpoint 文件路径（.db）
        """
        import logging
        _log = logging.getLogger(__name__)

        try:
            case = self._extract_case_from_checkpoint(checkpoint_path)
            if case:
                self.add_case(case)
                _log.info("从 %s 学习到案例 %s", checkpoint_path, case.case_id)

            # TODO: 提取失败模式（需要解析 checkpoint 中的失败记录）

        except Exception as exc:
            _log.warning("从 %s 学习失败：%s", checkpoint_path, exc)

    def _extract_case_from_checkpoint(self, checkpoint_path: str) -> OptimizationCase | None:
        """
        从 checkpoint 提取案例（内部实现）。

        需要读取 checkpoint.db 的以下信息：
        - problem_definition: 问题定义
        - decisions: 动作序列
        - final_state: 最终结果
        - metrics: 最终指标（HV、收敛率等）

        Returns:
            OptimizationCase 或 None（失败时）
        """
        import sqlite3
        from datetime import datetime

        # 连接 checkpoint 数据库
        conn = sqlite3.connect(checkpoint_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # 读取问题定义（假设存储在 metadata 表）
            cursor.execute("SELECT value FROM metadata WHERE key = 'problem_definition'")
            row = cursor.fetchone()
            if not row:
                return None
            problem = json.loads(row["value"])

            # 读取动作序列（假设存储在 decisions 表）
            cursor.execute("SELECT action_plan FROM decisions ORDER BY id")
            actions = [json.loads(r["action_plan"]) for r in cursor.fetchall()]

            # 读取最终结果
            cursor.execute("SELECT value FROM metadata WHERE key = 'final_result'")
            row = cursor.fetchone()
            final_result = row["value"] if row else "unknown"

            # 读取最终指标
            cursor.execute("SELECT value FROM metadata WHERE key = 'final_metrics'")
            row = cursor.fetchone()
            metrics = json.loads(row["value"]) if row else {}

            # 生成 case_id（基于 checkpoint 路径的哈希）
            case_id = hashlib.md5(checkpoint_path.encode()).hexdigest()[:12]

            # 提取标签（基于目标函数）
            tags = []
            if "objectives" in problem:
                tags.extend(problem["objectives"][:3])  # 最多 3 个目标作为标签

            return OptimizationCase(
                case_id=case_id,
                problem=problem,
                actions=actions,
                final_result=final_result,
                metrics=metrics,
                timestamp=datetime.now().isoformat(),
                tags=tags,
            )

        except Exception:
            return None
        finally:
            conn.close()

    def get_statistics(self) -> dict:
        """获取案例库统计信息"""
        return {
            "total_cases": len(self.cases),
            "successful_cases": len([c for c in self.cases if "verified" in c.final_result]),
            "failure_patterns": len(self.failure_patterns),
            "tags": list(set(tag for c in self.cases for tag in c.tags)),
        }
