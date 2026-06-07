"""
llm_client.py — PAO 大模型客户端（多 provider，密钥仅来自环境变量）。

设计原则：
  - API key 绝不写进代码或 YAML，只从环境变量读取。
  - 多 provider 可切换：PAO_LLM_PROVIDER = anthropic | openai | deepseek。
  - 缺 key 时 is_configured() 返回 False，由上层降级，绝不在此抛错或伪装。
  - provider SDK 延迟导入（lazy import），未选用的 provider 不需要安装。

环境变量约定：
  PAO_LLM_PROVIDER     provider 名（默认 anthropic）
  PAO_LLM_MODEL        模型名（缺省时用 provider 默认模型）
  PAO_LLM_TEMPERATURE  采样温度（默认 0.2，偏确定性）
  PAO_LLM_MAX_TOKENS   最大输出 token（默认 2048）
  ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY  各 provider 密钥

.env 文件支持（项目根目录）：
  复制 .env.example 为 .env，填入对应 key，本模块首次导入时自动加载。
  .env 已被 .gitignore 排除，不会提交到 git。

典型用法：
  cfg = load_llm_config()                 # 从环境变量（含 .env）构造
  cfg = load_llm_config(model="...")      # 覆盖模型
  if is_configured(cfg):
      text = chat(cfg, system="...", user="...")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 自动加载 .env（项目根目录）
# ---------------------------------------------------------------------------
# 只在 .env 存在时才加载；未安装 python-dotenv 时静默跳过（不阻断正常使用）。
# override=False：已有的系统环境变量优先于 .env，避免覆盖 CI/CD 注入的 key。

def _load_dotenv_if_present(override: bool = False) -> None:
    env_file = Path(__file__).parent.parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file, override=override)
        _log.debug("已加载 .env（override=%s）：%s", override, env_file)
    except ImportError:
        _log.debug("python-dotenv 未安装，跳过 .env 加载（可 pip install python-dotenv）")


_load_dotenv_if_present()

_log = logging.getLogger(__name__)

# provider → (默认模型, 密钥环境变量名)
_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "anthropic": ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    "openai":    ("gpt-4o-mini",       "OPENAI_API_KEY"),
    "deepseek":  ("deepseek-chat",     "DEEPSEEK_API_KEY"),
}

_DEFAULT_PROVIDER = "anthropic"


@dataclass(frozen=True)
class LLMConfig:
    """大模型配置。api_key 仅来自环境变量，不参与日志/序列化展示。"""

    provider: str
    model: str
    api_key_env: str
    api_key: str | None
    temperature: float = 0.2
    max_tokens: int = 2048

    def redacted(self) -> str:
        """用于日志/报告的脱敏摘要，绝不输出 key 本身。"""
        key_state = "已设置" if self.api_key else f"缺失（请设置 {self.api_key_env}）"
        return (
            f"provider={self.provider}, model={self.model}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}, "
            f"api_key={key_state}"
        )


# ---------------------------------------------------------------------------
# 配置构造
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning("环境变量 %s=%r 无法转为 float，使用默认值 %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        _log.warning("环境变量 %s=%r 无法转为 int，使用默认值 %s", name, raw, default)
        return default


def load_llm_config(
    provider: str | None = None,
    model: str | None = None,
) -> LLMConfig:
    """从环境变量构造 LLMConfig，显式参数优先。

    每次调用都重新从 .env 加载一次（override=True），确保 .env 里的 key
    始终优先于系统环境变量中可能残留的无效值（如 COM 调用后被清空的 token）。
    这样即使 Aspen COM 扫描过程中环境变量状态改变，此后的调用仍能拿到正确 key。

    Args:
        provider: 覆盖 PAO_LLM_PROVIDER；None 时读环境变量，再缺省为 anthropic。
        model:    覆盖 PAO_LLM_MODEL；None 时读环境变量，再缺省为 provider 默认模型。

    Returns:
        LLMConfig。未知 provider 回退到默认 provider 并告警。
        api_key 仅从对应环境变量读取，缺失时为 None（由 is_configured 判定）。
    """
    # 每次调用重新加载 .env（override=True：.env 优先于系统环境变量）
    # 先把空字符串 key 从 os.environ 中删除，防止 dotenv 把空值视为"已设置"而跳过覆盖
    _prov_check = (provider or os.environ.get("PAO_LLM_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()
    if _prov_check in _PROVIDER_DEFAULTS:
        _, _key_env = _PROVIDER_DEFAULTS[_prov_check]
        if os.environ.get(_key_env) == "":
            del os.environ[_key_env]
    _load_dotenv_if_present(override=True)

    prov = (provider or os.environ.get("PAO_LLM_PROVIDER") or _DEFAULT_PROVIDER).strip().lower()
    if prov not in _PROVIDER_DEFAULTS:
        _log.warning(
            "未知 PAO_LLM_PROVIDER=%r，回退到 %s。支持：%s",
            prov, _DEFAULT_PROVIDER, sorted(_PROVIDER_DEFAULTS),
        )
        prov = _DEFAULT_PROVIDER

    default_model, key_env = _PROVIDER_DEFAULTS[prov]
    chosen_model = (model or os.environ.get("PAO_LLM_MODEL") or default_model).strip()
    api_key = os.environ.get(key_env) or None

    return LLMConfig(
        provider=prov,
        model=chosen_model,
        api_key_env=key_env,
        api_key=api_key,
        temperature=_env_float("PAO_LLM_TEMPERATURE", 0.2),
        max_tokens=_env_int("PAO_LLM_MAX_TOKENS", 2048),
    )


def is_configured(cfg: LLMConfig) -> bool:
    """是否具备调用大模型的条件（当前仅要求 api_key 存在）。"""
    return bool(cfg.api_key)


# ---------------------------------------------------------------------------
# Chat 模型构造与调用
# ---------------------------------------------------------------------------

def build_chat_model(cfg: LLMConfig):
    """按 provider 延迟导入并构造 LangChain Chat 模型。

    Raises:
        RuntimeError: api_key 缺失，或 provider SDK 未安装/构造失败。
    """
    if not is_configured(cfg):
        raise RuntimeError(
            f"未配置 API key：请设置环境变量 {cfg.api_key_env}（provider={cfg.provider}）。"
        )

    try:
        if cfg.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                api_key=cfg.api_key,
            )
        if cfg.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                api_key=cfg.api_key,
            )
        if cfg.provider == "deepseek":
            from langchain_deepseek import ChatDeepSeek
            return ChatDeepSeek(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                api_key=cfg.api_key,
            )
    except ImportError as exc:
        raise RuntimeError(
            f"provider={cfg.provider} 的 SDK 未安装或导入失败 — {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"构造 {cfg.provider} Chat 模型失败 [{type(exc).__name__}] — {exc}"
        ) from exc

    raise RuntimeError(f"不支持的 provider：{cfg.provider!r}")


def chat(cfg: LLMConfig, system: str, user: str) -> str:
    """单轮对话：发送 system + user 消息，返回纯文本回复。

    Raises:
        RuntimeError: 未配置 / 构造失败 / 调用失败（由上层降级处理）。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    model = build_chat_model(cfg)
    try:
        resp = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"大模型调用失败 [{type(exc).__name__}] — {exc}"
        ) from exc

    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        # 部分 provider 返回分段内容块，拼接文本部分
        parts = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict) and "text" in blk:
                parts.append(str(blk["text"]))
        return "".join(parts)
    return str(content)
