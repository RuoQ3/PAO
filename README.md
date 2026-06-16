# PAO — Process Aspen Optimization

> 基于贝叶斯优化 + LangGraph 多智能体的 Aspen Plus 化工流程多目标优化框架

PAO 通过 Windows COM 自动化驱动 Aspen Plus，将贝叶斯代理模型优化算法、帕累托多目标搜索、LangGraph 协作状态机、HITL（人机协同）工作流与经济/排放目标计算集成为一套完整的流程优化工具链，并提供 FastAPI 后端 + Vue 3 前端可视化界面。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| **单目标贝叶斯优化** | 支持 GP / RF / ET / GBRT 代理模型，EI / UCB / PI 采集函数 |
| **多目标帕累托优化** | NSGA-II 快速非支配排序 + 拥挤距离，WFG 超体积指标 |
| **自适应区域搜索** | 三阶段策略：空间分区 → DOE 采样 → 贝叶斯精化 |
| **参数扫描 (DOE)** | 网格 / LHS 拉丁超立方 / 随机采样 |
| **经济目标** | TAC 总年费用（Turton 方法，CAPEX + OPEX） |
| **排放目标** | LCA 全生命周期 CO₂ 当量核算 |
| **AI 协作 Onboarding** | LLM 自动扫描 Aspen 变量、解析优化意图、生成配置草案 |
| **HITL 人机协同** | LangGraph 状态机在关键节点暂停，等待用户确认或决策 |
| **实时进度推送** | FastAPI SSE 流推送 Pareto 前沿 + 超体积历史 |
| **SQLite 持久化** | 所有运行案例自动存入本地数据库，支持断点续算 |
| **YAML 配置驱动** | 无需修改代码，通过配置文件定义优化问题 |
| **优雅降级** | scikit-optimize / numpy 缺失时自动回退到随机采样 |

---

## 系统架构

```
┌─────────────────────────────────────────┐
│           Vue 3 前端 (frontend/)         │
│  Pareto 图 · SSE 实时流 · HITL 交互面板  │
└────────────────┬────────────────────────┘
                 │ HTTP / SSE
┌────────────────▼────────────────────────┐
│       FastAPI 后端 (backend/)            │
│  /session/start  /stream  /resume        │
│  /status  /report                        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     LangGraph 协作状态机 (src/agents/)   │
│                                         │
│  onboarding → human_confirm             │
│      → optimization → analysis          │
│      → human_decide → done              │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   Aspen Plus COM 驱动 (src/aspen_driver/)│
│   优化算法层 (src/optimization/)         │
│   SQLite 持久化 (src/database/)          │
└─────────────────────────────────────────┘
```

---

## 环境要求

- **操作系统**：Windows 10 / 11（Aspen Plus COM 接口仅支持 Windows）
- **Python**：3.9 或以上
- **Node.js**：18 或以上（仅前端开发需要）
- **Aspen Plus**：已安装并激活许可证（支持 V11 及以上版本）
- 依赖库见 [requirements.txt](requirements.txt)

---

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd PAO

# 2. 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 安装前端依赖（可选，仅 Web UI 开发时需要）
cd frontend
npm install
cd ..
```

---

## 快速开始

### 方式一：命令行直接运行

```bash
# 单次确定性运行
python src/main.py cases/demo_case/case_config.yaml

# 单目标贝叶斯优化
python src/main.py cases/demo_case/case_config.yaml --db results.db

# 多目标帕累托优化
python src/main.py cases/demo_case/pareto_config.yaml --db results.db

# 干运行（仅验证配置，不启动 Aspen）
python src/main.py cases/demo_case/pareto_config.yaml --dry-run
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `config` | YAML 配置文件路径 | 必填 |
| `--db` | SQLite 数据库路径 | `results.db` |
| `--log` | 日志级别 (DEBUG/INFO/WARNING) | `INFO` |
| `--log-file` | 日志文件路径 | 无（仅控制台） |
| `--dry-run` | 仅解析配置，不运行仿真 | `False` |

---

### 方式二：Web UI（后端 + 前端）

```bash
# 1. 启动后端
python -m backend.server
# 或：uvicorn backend.app:app --reload --port 8000

# 2. 启动前端（另开终端）
cd frontend
npm run dev
# 浏览器访问 http://localhost:5173
```

前端提供：
- 多智能体 7 节点状态矩阵可视化
- Pareto 前沿实时图表
- HITL 确认/决策交互面板
- 优化完成后 Markdown 报告展示

---

## AI Agent 协作工作流

PAO 内置基于 LangGraph 的多智能体协作状态机（`src/agents/graph.py`），节点流程如下：

```
START
  └→ onboarding_node        # 扫描 Aspen 文件 + 解析用户意图 + 生成配置草案
       └→ human_confirm_node  # [HITL 暂停] 用户审查草案、修改边界/约束
            └→ optimization_node  # 运行 Pareto 贝叶斯优化（写入 SQLite）
                 └→ analysis_node    # Process Advisor 只读分析
                      └→ human_decide_node  # [HITL 暂停] 继续 / 调整 / 结束
                           ├→ continue → human_confirm_node
                           ├→ adjust  → onboarding_node
                           └→ done   → done_node → END
```

**HITL 交互方式：**
- 图在 `human_confirm` 和 `human_decide` 节点执行前自动暂停
- 通过 SSE 推送 `hitl_prompt` 事件，前端展示对应界面
- 用户提交后调用 `POST /api/session/{id}/resume` 恢复执行

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/session/start` | 启动新优化会话，返回 `session_id` |
| `GET` | `/api/session/{id}/stream` | SSE 事件流（message / hitl_prompt / progress / done / error） |
| `POST` | `/api/session/{id}/resume` | 提交 HITL 响应，恢复暂停会话 |
| `GET` | `/api/session/{id}/status` | 心跳查询（running / hitl_paused / done / error） |
| `GET` | `/api/session/{id}/report` | 获取优化完成后的 Markdown 报告 |
| `GET` | `/` | 健康检查 |

所有端点均支持 `?mock=true` 参数切换到 mock 模式，不依赖真实 Aspen，适合前端开发与演示。

---

## 配置文件格式

```yaml
# 仿真器设置
simulator:
  filepath: path/to/simulation.bkp   # Aspen 备份文件路径
  visible: false
  suppress_dialogs: true
  timeout: 300

# 设计变量
design_variables:
  - name: feed_flow
    aspen_path: \Data\Streams\FEED\Input\TOTFLOW\MIXED
    type: continuous          # continuous | integer
    lower_bound: 100.0
    upper_bound: 500.0
    initial_value: 300.0

# 目标函数
objectives:
  - name: TAC
    type: tac                 # tac | emissions | aspen_path
    minimize: true
    unit: $/yr
  - name: CO2
    type: emissions
    minimize: true
    unit: kg/yr

# 约束条件（可选）
constraints:
  - name: purity
    aspen_path: \Data\Streams\PROD\Output\MOLEFRAC\MIXED\ETHANOL
    operator: ">="
    threshold: 0.995

# 优化器
optimizer:
  type: pareto_bayesian       # run | bayesian | pareto_bayesian | param_scan | adaptive_region
  surrogate_model: GP         # GP | RF | ET | GBRT | random
  acquisition: EI             # EI | UCB | PI
  n_initial: 10
  n_iterations: 40
  scalarization: Tschebycheff
```

---

## 优化工作流说明

```
src/main.py (CLI 入口)
    │
    ├── run_case                  单次确定性运行
    ├── optimize_case             单目标贝叶斯优化循环
    ├── optimize_pareto_case      多目标帕累托 + 贝叶斯
    ├── param_scan                DOE 参数扫描
    └── adaptive_region_search    三阶段自适应区域搜索
            │
            ├── Phase 0：输入空间分区（超立方网格）
            ├── Phase 1：各区域 DOE 采样 → 收敛率 / 灵敏度分析
            └── Phase 2：高优先级区域 → optimize_pareto_case 精化
```

所有工作流均通过 `SimulationDB` 将案例实时写入 SQLite，支持中断恢复、按目标排序查询与跨会话历史对比。

---

## 目录结构

```
PAO/
├── src/
│   ├── main.py                  # CLI 入口
│   ├── agents/                  # LangGraph 多智能体层
│   │   ├── graph.py             # 协作状态机（B2 阶段）
│   │   ├── onboarding_agent/    # Onboarding：变量发现 + 配置草案生成
│   │   ├── hitl_protocol.py     # HITL 交互协议定义
│   │   └── tools/               # Agent 工具集（validate_config 等）
│   ├── aspen_driver/            # Aspen Plus COM 底层驱动
│   ├── models/                  # 数据模型（ProcessCase、SimulationResult 等）
│   ├── workflows/               # 高层工作流（优化、扫描、自适应搜索）
│   ├── optimization/            # 算法层（代理模型、帕累托、灵敏度）
│   ├── database/                # SQLite 持久化（node_db、simulation_db）
│   ├── economics/               # TAC / 排放目标计算
│   ├── reporting/               # 摘要报告生成（Markdown）
│   └── utils/                   # 日志、文件 IO、单位换算
├── backend/                     # FastAPI 后端
│   ├── app.py                   # API 路由（H4-1）
│   ├── models.py                # Pydantic 请求/响应模型
│   ├── session_store.py         # 会话状态管理
│   └── server.py                # 启动入口
├── frontend/                    # Vue 3 前端
│   ├── src/
│   │   ├── views/               # 页面视图
│   │   ├── components/          # 组件（Pareto 图、Agent 矩阵、HITL 面板）
│   │   └── stores/              # Pinia 状态管理
│   └── package.json
├── tests/                       # pytest 测试套件
├── cases/                       # 示例案例与配置文件
│   └── demo_case/
│       ├── pareto_config.yaml
│       └── *.bkp                # Aspen 备份文件
├── configs/
│   └── aspen_semantics/         # Aspen 语义元数据映射（RADFRAC、HEATX 等）
├── docs/                        # 开发文档
├── reports/                     # 运行报告输出
├── scripts/                     # 辅助脚本
├── requirements.txt
└── README.md
```

---

## 运行测试

```bash
# 运行全部测试
pytest tests/

# 仅运行不依赖 Aspen 的单元测试
pytest tests/ -k "not aspen"

# 查看详细输出
pytest tests/ -v
```

> 集成测试（`smoke_test_*.py`）需要有效的 Aspen Plus 安装与许可证。不含 Aspen 环境时，单元测试（`test_*_logic.py`）可独立运行。

---

## 依赖说明

| 包 | 用途 | 是否必需 |
|----|------|----------|
| `pywin32` | Aspen Plus COM 自动化 | ✅ 必需 |
| `PyYAML` | 配置文件解析 | ✅ 必需 |
| `fastapi` + `uvicorn` | HTTP / SSE 后端 | ✅ 后端必需 |
| `langgraph` | 多智能体状态机 | ✅ Agent 工作流必需 |
| `numpy` | LHS 采样加速 | ⚡ 可选（缺失时回退随机采样） |
| `scikit-optimize` | 贝叶斯代理模型 | ⚡ 可选（缺失时回退随机采样） |
| `scipy` / `scikit-learn` | scikit-optimize 依赖 | ⚡ 随 skopt 安装 |
| `matplotlib` | 帕累托前沿可视化脚本 | 📊 仅脚本 |
| `pytest` | 测试框架 | 🧪 仅开发 |

---

## License

本项目为私有研究代码，未经授权禁止分发。
