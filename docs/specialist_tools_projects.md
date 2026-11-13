# 第三批：专业工具与无需 YAML 的项目建立

本页描述实际接线，补充第一、二批执行授权与证据约束。未运行真实 LLM、外部知识、Aspen COM 或真实模型；所有验收使用 fake provider、临时文件、静态 NodeDB 和 fake 执行器。

## 现状与改造

本批开始时已有 `tool_runtime.py`、`ProjectOnboarding`、专业角色合同和 Gateway。项目草稿没有接到 ConversationSession/CLI；TaskPlanner 重复维护工具白名单；checkpoint 名称实际读取历史工况，不能宣称是真正 checkpoint 查询；旧 discover_tunables 还可直接打开 COM。

本批复用这些组件补齐接线，没有恢复旧闭环、试写或隐式降级。角色仍可共享模型；每次是否调用工具由 LLM 决定，工具调用及参数校验由主机执行。没有给生产 Agent 任意代码、shell、安装或写代码权限。

## 实际能力清单

`/capabilities` 从实际 registry 查询处理器、输入/输出协议、来源、依赖、允许角色、模式、预算和超时。仅检查已选资料与本地依赖，不探测 LLM 服务或 COM。配置了 key 不代表已验证服务可达。

| 能力 | 实际状态 / 处理器 | 限制与审批 |
|---|---|---|
| 工程解释、边界建议、代码分析、报告 | 专业角色 → `reason` | 必要 LLM；解释和建议不等于运行验证 |
| 工艺资料 | `read_process_context` → 当前授权 ProcessContext 投影 | 静态，只读；拓扑不完整时保留诊断 |
| 配置资料 | `read_yaml` → 当前结构化配置 | 配置声明，不是 Aspen 测量 |
| 节点资料 | `read_node_db` → builder 已读取的 NodeDB 节点/工况快照 | 匹配模型及来源版本；不连接 COM |
| 历史工况 | `read_simulation_db` → 精确会话的 history_evidence | 当前最多 100 条；不是实时值 |
| 已选附件 | `read_files(index)` → 当前附件正文 | 完整 UTF-8、授权索引；截断/密钥文件拒绝；无任意路径参数 |
| 已有证据 | `read_evidence` | 仅当前会话资料 |
| 配置校验 | `validate_config` → 实际 load_optimize_config、build_optimization_plan、preflight_optimizer | 依赖不足即失败；不导入 custom_module；不打开 Aspen |
| Arrhenius 数值拟合 | `fit_arrhenius_linear(index, rate_unit)` → 既有线性回归内核 | 显式线性化最小二乘；CSV 为 temperature_K,rate_constant；不偷偷改为非线性或反向降级 |
| 优化计划 | `plan_optimization` → 既有 build_optimization_plan | 仅完整配置及可用算法；返回具体 ActionProposal，需 Gateway 预览、审批及执行授权 |
| 单次仿真、参数影响、优化执行 | 既有 ActionGateway | 具体计划、人工审批、活动授权、硬边界及持久化总预算；本批未验证真实环境 |
| checkpoint 专用查询 | 未接入 | 不再用历史工况冒充 checkpoint |
| 外部知识查询 | 未接入生产工具 registry | 独立 TechnicalAdvisor 模块存在，不等于当前角色已接通 |
| COM 节点发现 | 未接入 Gateway 读取适配器 | 旧扫描底层直接拒绝；`/project read-plan` 生成不可执行的待审核读取需求 |
| 生产代码执行/安装/写代码 | 不提供 | code_engineer 仅阅读明确选择的附件并提出建议 |
| PSO / GA | 未实现 | 不替换为 BO/random |

角色与工具的唯一默认关联源为 `registry.role_definitions()`：

| 角色 | 工具 |
|---|---|
| project_researcher | read_files、read_yaml、read_node_db、read_simulation_db、validate_config |
| process_advisor | read_process_context、read_node_db、read_simulation_db |
| optimization_advisor | read_yaml、read_simulation_db、validate_config、plan_optimization |
| boundary_advisor | read_yaml、read_process_context |
| kinetics_advisor | read_files、read_simulation_db、fit_arrhenius_linear |
| risk_reviewer | read_process_context、read_simulation_db |
| code_engineer | read_files |
| report_writer | read_evidence、read_files |
| general_agent | 无工具 |

默认角色最多 4～10 个推理步、2～8 次调用，由 RuntimePolicy 再压缩；每工具默认 4 次、5 秒，每任务默认 60 秒。实际限制在主机校验，耗尽即停止。超时表示停止等待并弃用迟到结果，不声称 Python 线程已终止。

## 调用链与证据

用户 → 必要 LLM 分类 → TaskPlanner / 角色定义 → Runtime 压缩预算及白名单 → 专业 LLM 返回严格 tool_calls → ToolSession 检查注册、角色、参数类型、次数、截止时间和来源版本 → 实际只读处理器 → 结构化结果 → 下一轮 LLM 综合 → Answer / TerminalRenderer。

分类提示由当前 registry 提供真实角色名称与职责。项目读取请求缺少角色、或返回未注册角色时，在加载上下文和派发前触发现有的一次 LLM JSON 修复；模型必须自行重新选择已注册角色。修复失败仍 fail-closed，不按关键词补选角色。一般聊天仍允许不调用专业角色。

专业回答必须包含 `text`、`uncertainties`、`confidence`，可包含结构化 `blocks`。兼容部分模型额外返回的顶层 `evidence`，但它只能是命名的 JSON pointer 引用表（可嵌套对象、数组），且每条引用必须定位到本轮实际证据；不能包含新事实、测量值或指令。通过校验的引用保存在 `Answer.details.reasoning_evidence.answer_citations` 及任务证据快照中，不覆盖来源资料。指标块仍须逐项匹配实际数值、基准、单位及比较范围，引用存在不代表工程结论已验证。

协议校验失败时，JSONL 审计增加 `agent_output_validation_failed`，记录角色、任务/会话、原因码、返回类型、字段数量和必要缺失项。原因码区分字段、类型、置信度、内容块/引用表以及块引用问题；不记录未知字段名称或内容。无效引用、未知扩展字段仍停止本次任务，保留 `agent_output_invalid` 错误码，不生成替代回答。

工具包保存 evidence_id、source_versions、snapshot_digest、handler_version 和来源。调用审计保留 call_id、session/task、角色、已验证参数、状态、证据引用、耗时和错误类别；凭据字段脱敏，非法参数不进入审计。来源在派发前、返回后及最终推理返回后检查。文件与工具正文都是数据，不获得系统权限。必要失败停止；非必要失败保留 unavailable 并显示限制。自由文本工程解释仍需专业复核，JSON pointer 校验不能证明所有解释正确。

有副作用能力只能返回具体计划；Lead 将它交给同一个 Gateway.preview_proposal。确认之前不能 execute。工具不能批准自己，`approved` 字段不构成凭证。显式优化请求也走 optimization_advisor 的该流程；若 LLM 只回答建议而没请求计划工具，就没有可执行授权。

## 无 YAML 入口

```powershell
python -m src.main --new-project
python -m src.main "D:\Models\plant.bkp" --new-project --node-db "D:\Models\nodes.db"
python -m src.main --project-draft "output\draft.json"
# 兼容原路径
python -m src.main "cases\demo_case\case_config.yaml" --conversation
```

从现有对话也可 `/project new`，或 `/project from-config 路径.yaml` 导入当前配置为未确认草稿。模型路径和文件选择是确定性命令，业务理解和澄清必须由 LLM 完成；没有 LLM 就停止，不切换固定问答。`--file` 可重复选择专业问题附件，动力学 CSV 使用显式 K 单位列名；旧 `--data` 不是当前专业工具附件入口。

| 对话操作 | 行为 |
|---|---|
| /project model 路径 | 选择/更换模型；新建独立项目 scope，清空旧证据与计划 |
| /project node-db 路径 | 选择静态节点目录，匹配模型摘要 |
| /project history-db 路径 | 选择历史库；只读当前精确会话资料，不混入其他会话 |
| 直接自然语言 | 补充、纠正目标、基准、单位、约束、边界、算法与预算 |
| /project show、preview、back | 展示当前草稿、缺失和冲突，不执行 |
| /project undo | 返回上一草稿修订，撤销确认及旧计划 |
| /project edit | 从已启用项目返回草稿编辑 |
| /project algorithms | 列出已有后端和依赖状态；不可用算法不能保存为有效执行配置 |
| /project read-plan | 生成缺失节点读取需求，明确标记执行适配器未接通，不打开 COM |
| /project confirm | 通过真实解析与语义校验，确认业务配置 |
| /project save 新路径.yaml | 排他创建有效 YAML 和 .yaml.project.json 来源审计 |
| /project checkpoint 新路径.json | 保存未完成草稿 |
| /project resume 路径.json | 恢复并重核来源，撤销全部确认和执行授权 |
| /project use | 检查已保存配置一致，进入该项目专业问答 |
| /project cancel | 清空当前作用域的证据和计划 |

草稿保存产品/物流/组分、目标方向、指标/基准/单位、约束、变量类型/硬边界、算法、总预算/时限、来源和变更历史。user、model_fact、history、agent_inference、system_default 分开；user 必须引用当前用户原话，事实更新必须与所引证据一致。算法建议不等于选择，推断不等于已确认要求。

支持连续/整数变量；离散/派生变量拒绝。纯度明确 mass/mole，分数单位为 1/fraction；不自动换算百分数。当前建立入口只接受 aspen_path 目标，成本/排放计算模型需后续明确适配。硬边界不能扩大现有配置/模型证据的硬边界，不编造安全范围。冲突（包括描述与数值不一致）由 LLM识别、结构化保留并要求用户解决；不是对任意自然语言进行形式证明。

BKP 仅有现有静态文本索引及可解析拓扑/组成，不能保证所有节点路径可提取。APW 没有静态正文解析。要在不记树路径的情况下完成配置，必须有匹配模型的 NodeDB、已有配置或其他明确静态路径证据；没有证据时停在缺失状态，不假称已读懂模型。

## 状态关系

`draft（可不完整） → confirm 校验 → confirmed（业务确认） → save（有效配置+来源审计） → use（专业问答）`。

这条链不产生执行授权。执行仍是独立的 `具体 ActionProposal → Gateway 校验及预览 → 人工批准 → 一次性授权 → 执行/预算账本`。业务变化、undo、更换来源/模型、取消及恢复均撤销旧计划；源内容同路径更新会标记冲突、撤销业务确认。恢复 conversation checkpoint 也不继承业务确认。保存不覆盖原文件；已有目标显示差异并要求新文件名。只读查看、草稿存盘、预览均不运行 Aspen。

## 两条可复现模拟示例

以下命令使用测试内 fake LLM；conftest 同时拦截网络及 Dispatch/DispatchEx。不需要密钥，不访问原始模型。

```powershell
python -B -m pytest tests/test_batch3_tools_projects.py -q -s -p no:cacheprovider --basetemp .test_batch3_examples -k "existing_project_specialist or no_yaml_project_correction"
```

示例一创建临时已有项目及 NodeDB，输入“R1 温度节点的单位和已有证据是什么？”。fake 分类选择 process_advisor，fake 专业模型请求 read_node_db，实际处理器返回 R1/TEMP、C、样本 40 及版本，后续推理输出“目录样本为 40，不是实时测量”。终端输出工具 completed 和 evidence_id，无执行审批、无 Aspen。

示例二从临时 APW 开始，无 YAML，选择匹配的静态 NodeDB。用户自然语言指定 A/S1 的质量分数最大化、阈值 0.9、温度 20～80 C、显式 random、DOE 2+搜索 1、总评估 8。再纠正为阈值 0.999、温度上限 70 C、总预算 9。执行 confirm、save、use，输出生成配置路径。测试用实际解析器及计划语义验证往返目标、质量基准、单位、边界和总预算，保留修订原值和用户原话。random 是模拟中用户明确选择，并非 BO 失败后的替换。

## 修改与验证

主要修改：main.py（入口、附件、项目命令、延迟 Gateway）、lead/session.py（草稿路由及恢复）、lead/agent.py（能力查询、优化计划衔接）、registry.py/runtime.py（统一角色权限）、tool_runtime.py/reasoning.py（主机调用及版本证据）、onboarding_agent/project.py（草稿与状态）、context/builder.py（静态目录证据）、context/loader.py（密钥目录排除）、tools/discover_tunables.py（拒绝未授权 COM）、llm_client.py（密钥文件显式 opt-in）、README、本文、新增模拟测试及 .gitignore 精确放行新增交付文件。

兼容修复：启动加载配置时默认自动读项目根目录 .env，保持既有 .env 优先行为；PAO_LOAD_DOTENV=0 可禁用，导入模块不读取文件。模拟测试显式禁用或仅选择临时假配置。旧 COM discovery 不再直连；checkpoint 不再假冒历史查询。原 YAML CLI / --conversation / --dry-run 保留。执行预算、硬边界、算法身份和前两批清理/未知状态不改变。

已完成专项测试、受影响回归和全量模拟回归，命令如下（每次运行使用新临时目录，避免重用 Windows 被锁定的旧临时目录）：

```powershell
python -B -m pytest tests/test_batch3_tools_projects.py -q -p no:cacheprovider --basetemp .test_batch3_specific_final
python -B -m pytest tests/test_batch3_tools_projects.py tests/test_answer_presentation.py tests/test_stage7_actions.py tests/test_stage8_closed_loop.py tests/test_fail_closed.py -q -p no:cacheprovider --basetemp .test_batch3_regression2
python -B -m pytest tests -q -p no:cacheprovider --basetemp .test_batch3_full_verified
git diff --check
```

验收记录（2026-11-11）：专项测试最终 25 项纳入全量通过；受影响回归 100 项通过；最终全量 **453 passed，4 warnings，32.74s**；两条模拟示例及 CLI 端到端组合 **3 passed，22 deselected**；`git diff --check` 通过。全量覆盖前两批执行边界、预算、算法身份、真实生命周期状态、上下文和证据；测试网络与 COM 拦截始终启用。BoTorch 测试报告缺少 C++ 编译器而使用同算法的纯 Python 内核，未替换优化算法；另有依赖弃用及已有路径字符串警告。

剩余真实环境验证包括 provider 多步严格 JSON 工具协议、真实静态节点语义/单位与模型版本，以及既有 Gateway 执行器、COM 清理和优化后端。尚无 Gateway COM 读取适配器、生产外部知识工具、PSO/GA、任意代码沙盒或新 Web 前端。
