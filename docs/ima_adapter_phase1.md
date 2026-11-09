# IMA 适配层（阶段 1）

`src/integrations/ima/` 是与 IMA SDK 解耦的边界。主 Agent 只依赖 `ImaClient`，传输实现可替换为 HTTP、MCP 或 Mock。

## 接口

- `ImaClient.health() -> dict`
- `ImaClient.capabilities() -> dict`
- `ImaClient.query(ImaRequest) -> ImaAnswer`
- `ImaClient.query_with_sources(ImaRequest) -> ImaAnswer`
- `ImaClient.close() -> None`

`ImaAnswer` 保留 `answer`、`confidence`、`assumptions`、`request_id` 和 `ImaSource` 列表。来源包含 `source/title/chapter/page/snippet/confidence/metadata`，并可投影为阶段 0 的公共 `src.agents.evidence.models.ImaAnswer`。

## 配置

配置只从环境变量读取，不进入 checkpoint 业务内容：

`PAO_IMA_TRANSPORT`（`none`/`mock`/`http`）、`PAO_IMA_BASE_URL`、`PAO_IMA_AUTH_TOKEN`、`PAO_IMA_AUTH_HEADER`、`PAO_IMA_TIMEOUT_SECONDS`、`PAO_IMA_RETRIES`、`PAO_IMA_MODEL`、`PAO_IMA_KNOWLEDGE_BASE`、`PAO_IMA_MAX_CONTEXT_BYTES`。

日志使用脱敏配置，不输出 token。请求上下文超过限制或为空答案会明确失败。

## 错误策略

- `ImaNotConfiguredError`：未配置服务或未注册传输。
- `ImaTimeoutError`：网络超时。
- `ImaAuthError`：401/403 或认证失败。
- `ImaServiceUnavailableError`：连接失败或 5xx/其他 HTTP 错误。
- `ImaAnswerError`：JSON/字段格式错误、空答案、非法置信度或上下文过大。

传输错误按配置重试；解析错误不重试。该层没有 Aspen 导入，也不执行任何 Aspen 操作。

## 协议待确认

真实 IMA 团队仍需确认：HTTP/MCP 调用名称和路径、请求字段命名（`query/context/model/knowledge_base`）、鉴权头格式、响应答案字段（当前兼容 `answer`/`text`）、来源字段及页码类型、置信度范围、健康检查和能力发现返回结构。确认后只需替换传输或解析映射，不修改 Agent 接口。

## 测试

```text
python -m pytest tests/integrations/ima tests/agents/contracts -q
```

结果：`12 passed`。测试全部使用 Mock/标准库，不调用真实外部服务。

\r\n