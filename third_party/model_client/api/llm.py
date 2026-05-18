"""
LLM 客户端 — 通过 OpenAI 兼容 API 调用远程 LLM 服务（vLLM / OpenAI / DeepSeek 等）。

基于 LiteLLM，接入本地 vLLM / OpenAI 兼容端点。

配置优先级：
  构造函数传入 LLMConfig  >  .env 文件  >  系统环境变量

依赖::

    pip install litellm python-dotenv openai
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from model_client._utils import first_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen2.5-7B-Instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096

_THINKING_MODELS = ("qwen3", "qwen-3", "deepseek-r1", "deepseek_r1")
"""需要自动禁用思考模式的推理模型名称片段（小写匹配）。"""

_LITELLM_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "azure/", "gemini/", "google/",
    "mistral/", "cohere/", "huggingface/", "deepseek/", "ollama/",
    "bedrock/", "vertex_ai/", "groq/", "together_ai/", "replicate/",
    "xinference/", "fireworks_ai/",
)
"""LiteLLM 已知的提供商前缀，用于判断模型名是否已含路由前缀。"""


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _is_thinking_model(model: str) -> bool:
    """若模型名称表明它是推理型模型，返回 True。"""
    lower = model.lower()
    return any(fragment in lower for fragment in _THINKING_MODELS)


def _needs_openai_prefix(model: str) -> bool:
    """若模型名称不含已知提供商前缀，则需要追加 ``openai/`` 前缀。"""
    return not any(model.startswith(p) for p in _LITELLM_PROVIDER_PREFIXES)


def _build_thinking_extra_body(
    resolved_thinking: Optional[bool],
    base_url: Optional[str],
) -> dict:
    """
    构建思考模式控制参数的 ``extra_body`` 字典。

    有 ``base_url``（vLLM）时嵌套在 ``chat_template_kwargs`` 下；
    无 ``base_url``（云端 API）时放顶层。
    ``resolved_thinking`` 为 ``None`` 时返回空字典。
    """
    if resolved_thinking is None:
        return {}
    if base_url:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": resolved_thinking}}}
    return {"extra_body": {"enable_thinking": resolved_thinking}}


def _configure_litellm(model_name: str) -> None:
    """
    配置 LiteLLM 全局状态：

    - 禁用 success/failure 回调（抑制 async handler GC 噪音）
    - 注册本地模型零价格条目（消除 "not in cost map" DEBUG 日志）
    - 将 LiteLLM logger 提升至 WARNING 级别
    - 关闭 SSL 证书校验（适配内网自签名证书 / HTTP vLLM 端点）
    """
    import logging as _logging
    _litellm_logger = _logging.getLogger("LiteLLM")
    _litellm_logger.propagate = False
    _litellm_logger.setLevel(_logging.WARNING)

    try:
        import litellm  # type: ignore[import]

        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []
        litellm.suppress_debug_info = True
        # SSL 证书验证：内网 vLLM 端点常用自签名证书或纯 HTTP。
        # 默认关闭验证以适配常见部署场景；生产环境可设 LLM_SSL_VERIFY=true 开启。
        import os as _os
        _ssl_verify_str = _os.environ.get("LLM_SSL_VERIFY", "false").lower()
        litellm.ssl_verify = _ssl_verify_str not in ("false", "0", "no", "off")

        _zero_cost: dict = {
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 32768,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
        model_registry = {key: _zero_cost for key in (model_name, f"openai/{model_name}")}
        try:
            litellm.register_model(model_registry)
        except Exception:
            for key, val in model_registry.items():
                litellm.model_cost.setdefault(key, val)
    except Exception as exc:
        logger.debug("[LLMClient] litellm 配置失败（非致命）: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    OpenAI 兼容端点的运行时配置。

    Usage::

        # 从环境变量（.env）自动加载
        config = LLMConfig.from_env()

        # 手动指定
        config = LLMConfig(
            api_key="sk-xxxx",
            model="Qwen2.5-7B-Instruct",
            base_url="http://10.x.x.x:18023/v1",
        )
    """

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    enable_thinking: Optional[bool] = None
    """
    控制推理模型（Qwen3 / DeepSeek-R1）的思考模式。

    - ``None``  — 对已知推理模型自动设为 ``False``（禁用链式推理输出）
    - ``False`` — 强制禁用
    - ``True``  — 强制启用
    """

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """
        从环境变量（或 .env 文件）加载配置。

        环境变量读取优先级（高 → 低）：
          ``QWEN3_*`` > ``LLM_*`` > ``CONTEXTMIND_*`` > ``OPENAI_*`` > ``DEEPSEEK_*``

        与 ``temp/llm/.env`` 中的 ``QWEN3_*`` 命名完全兼容。

        Raises:
            ValueError: 若无法找到任何 API Key。
        """
        api_key = first_env([
            "QWEN3_API_KEY", "LLM_API_KEY", "CONTEXTMIND_API_KEY",
            "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY",
        ])
        if not api_key:
            raise ValueError(
                "缺少 LLM API Key。\n"
                "请在 .env 文件中设置 LLM_API_KEY（或 QWEN3_API_KEY / OPENAI_API_KEY）。"
            )

        base_url = first_env([
            "QWEN3_BASE_URL", "LLM_BASE_URL", "CONTEXTMIND_BASE_URL",
            "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "QWEN_BASE_URL",
        ])
        model = first_env([
            "QWEN3_MODEL", "LLM_MODEL", "CONTEXTMIND_MODEL",
            "OPENAI_MODEL", "DEEPSEEK_MODEL", "QWEN_MODEL",
        ], default=DEFAULT_MODEL)

        temperature_str = first_env(["QWEN3_TEMPERATURE", "LLM_TEMPERATURE", "OPENAI_TEMPERATURE"])
        temperature = float(temperature_str) if temperature_str else DEFAULT_TEMPERATURE

        max_tokens = (
            int(v)
            if (v := first_env(["QWEN3_MAX_TOKENS", "LLM_MAX_TOKENS", "MAX_TOKENS"]))
            else DEFAULT_MAX_TOKENS
        )

        enable_thinking_str = first_env(["LLM_ENABLE_THINKING", "ENABLE_THINKING"])
        if enable_thinking_str is not None:
            enable_thinking: Optional[bool] = (
                enable_thinking_str.lower() not in ("false", "0", "no", "off")
            )
        else:
            enable_thinking = None

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    基于 LiteLLM 的轻量级异步 LLM 客户端（API 模式）。

    每次 ``chat_async()`` 调用恰好发起一次 API 请求（无 Agent 框架开销）。
    通过 OpenAI 兼容接口访问远程 LLM 服务（vLLM / OpenAI / DeepSeek 等）。

    依赖::

        pip install litellm

    Usage::

        from model_client.api.llm import LLMClient, LLMConfig

        # 从 .env 自动加载
        client = LLMClient()
        reply = await client.chat_async([{"role": "user", "content": "你好"}])

        # 手动指定配置
        config = LLMConfig(api_key="sk-...", model="Qwen2.5-7B-Instruct",
                           base_url="http://10.x.x.x:18023/v1")
        client = LLMClient(config=config)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Args:
            config: LLM 配置；为 ``None`` 时调用 :meth:`LLMConfig.from_env` 自动加载。

        Raises:
            ImportError: 若未安装 ``litellm``。
        """
        try:
            import litellm as _litellm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "LLMClient 需要安装 litellm：pip install litellm"
            ) from exc
        self._litellm = _litellm

        self._config = config or LLMConfig.from_env()
        _configure_litellm(self._config.model)

        # 推理模型思考模式解析
        resolved_thinking: Optional[bool] = self._config.enable_thinking
        if resolved_thinking is None and _is_thinking_model(self._config.model):
            resolved_thinking = False
        self._resolved_thinking = resolved_thinking

        logger.debug(
            f"[LLMClient] 初始化完成  model={self._config.model}"
            f"  base_url={self._config.base_url}"
        )

    async def chat_async(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
    ) -> str:
        """
        发送一次 LLM 对话请求，返回助手回复文本（已剥除推理模型的 ``<think>`` 块）。

        Args:
            messages:    OpenAI 格式消息列表（``[{"role": ..., "content": ...}]``）。
            temperature: 覆盖 ``LLMConfig.temperature``；``None`` 则使用配置中的值。

        Returns:
            清洁后的助手回复文本。
        """
        temp = temperature if temperature is not None else self._config.temperature
        extra_kwargs = _build_thinking_extra_body(
            self._resolved_thinking, self._config.base_url
        )
        model_str = (
            f"openai/{self._config.model}"
            if _needs_openai_prefix(self._config.model)
            else self._config.model
        )

        try:
            response = await self._litellm.acompletion(
                model=model_str,
                messages=messages,
                api_base=self._config.base_url,
                api_key=self._config.api_key,
                temperature=temp,
                max_tokens=self._config.max_tokens,
                num_retries=0,  # 连接失败时快速报错，避免无效重试延迟
                **extra_kwargs,
            )
        except Exception as exc:
            _msg = str(exc).lower()
            if any(k in _msg for k in ("connection", "connect", "refused", "unreachable", "timeout")):
                url = self._config.base_url or "(未配置)"
                raise RuntimeError(
                    f"无法连接到 LLM 服务（{url}）。\n"
                    "请确认：\n"
                    "  1. vLLM / LLM 服务已启动并在该地址上正常监听\n"
                    "  2. .env 中的 QWEN3_BASE_URL / LLM_BASE_URL 端口和地址正确\n"
                    "  3. 网络可达性（防火墙 / 端口转发 / VPN）\n"
                    f"原始错误：{exc}"
                ) from exc
            raise

        text: str = response.choices[0].message.content or ""
        # 剥除推理模型（Qwen3 / DeepSeek-R1）输出的思考块
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
