# 对话终端、诊断日志与意图修复

## 默认输出

使用原来的 `--conversation` 启动命令即可。终端展示日志位置、简洁回答和回测进度。
每个工况开始/完成时更新 `completed/total`，超过 15 秒时定时显示“运行中”。
基准失败后报告真实完成数及停止状态，不把未执行工况算作完成。
进度回调可选；仅提示线程在后台运行，Aspen COM 保持在调用线程，退出工况时回收提示线程。

对话模式下，内部日志默认不进入终端，文件默认 DEBUG，UTF-8 编码并轮转保存。
完整 errors/warnings 留在日志、Answer 的结构化数据中；终端只显示最多三条简短失败原因和限制。

- `--console-log DEBUG`：显式开启详细终端诊断。
- `--file-log INFO`：单独改变文件级别。
- `--log-file PATH`：指定日志文件。
- 非对话模式继续沿用原有 `--log` 行为；独立 console/file 选项可覆盖它。

## 意图响应协议

模型必须返回完整 JSON，包含 intent、task_type、project_scope、context_required、subagents、action_type、requires_confirmation、confidence、missing_information、reason、analysis。
普通意图的 analysis 必须为 `{}`。
parameter_impact 必须包含 equipment、target_stream、target_component、metric、candidate_parameters。
前四个槽位允许未确定时为 null，并须在 missing_information 中列出字段名；candidate_parameters 是字符串数组。

独立的 `src/agents/lead/intent_response_adapter.py` 中 `IntentResponseAdapter` 解析 JSON/代码围栏，校验字段存在性和类型，最后交给 IntentSchema.from_dict 做枚举和确认规则等业务校验。适配器还负责构造结构化修复请求；分类器保留重试次数、审计、置信度判断和规则降级的控制。
非法字段不被转换成默认空值。

意图分类默认调用 `llm_client.chat_json()`。DeepSeek 首次及修复请求均携带 `response_format: {"type": "json_object"}`，提示词包含 JSON 要求与完整示例。普通 `chat()` 默认不启用 JSON mode，其他 provider 仍使用现有文本调用路径。原生 JSON mode 不替代业务校验；空白或 null 内容进入空响应修复流程。

首次格式或业务校验失败后，向同一模型发送一次结构化修复请求，包括原问题、当前关注对象、原响应和具体校验错误，要求仅返回完整修正 JSON。
修复仍失败时才使用本地规则。无 API key、请求失败或合法响应置信度不足直接降级；不会为这些情况盲目补发修复请求。

classification_source 为 llm、llm_repaired 或 rule_fallback；retry_count 为 0 或 1。
SessionState 保存 classification_retry_count。审计记录 intent_classification_trace 保存每次原始响应、具体错误和请求；DEBUG 日志同样保留完整修复过程。终端只显示简短中文降级原因。

历史检查点仍可省略新增元数据字段；新的模型响应必须遵守完整协议。

## 编码

修复 boundary_advisor 提示词中的损坏字符，添加 UTF-8 编辑配置及源码扫描测试。
stdout/stderr 使用 UTF-8；Aspen history 按 BOM、UTF-8、本机编码和 GB18030 解码，无法解码的字节以转义形式保留，不直接替换成丢失信息的字符。
历史文件中已经写入的乱码不会被自动改写；编码检测也不能保证识别所有无 BOM 编码。

## 验证与限制

测试覆盖日志隔离、独立级别、心跳及线程清理、基准停止、终端摘要与完整诊断保留、中文编码、意图合法响应与一次修复、规则降级、审批链和会话序列化。
新增测试直接截获 urllib 请求，检查首次和修复请求的 JSON mode 参数、普通对话隔离、空响应处理及独立适配器。测试未调用付费 API。
Windows Aspen COM 和真实 DeepSeek 服务需要现场复测；T0302 模型收敛不属于本次输出层修复。
