"""
tools.llm.client
~~~~~~~~~~~~~~~~

LLM 客户端 — 通过 OpenAI 兼容 API 调用远程 LLM 服务（vLLM / OpenAI / DeepSeek 等）。

基于 LiteLLM，接入本地 vLLM / OpenAI 兼容端点。

配置优先级：
  构造函数传入 LLMConfig  >  .env 文件  >  系统环境变量

依赖::

    pip install litellm python-dotenv
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

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

def _first_env(keys: list, default: Optional[str] = None) -> Optional[str]:
    """按优先级读取环境变量，返回首个非空值。"""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def _load_dotenv() -> None:
    """尝试加载 .env 文件。

    查找顺序（先找到即加载，不继续查找）：
      1. 当前工作目录的 ``.env``
      2. 当前工作目录的 ``third_party/.env``（开发期目录结构）
      3. 脚本调用位置向上逐级查找（最多 5 层），直至根目录
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        return

    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, ".env")):
        load_dotenv(os.path.join(cwd, ".env"))
        return

    third_party_env = os.path.join(cwd, "third_party", ".env")
    if os.path.exists(third_party_env):
        load_dotenv(third_party_env)
        return

    # 向上逐级查找
    current = os.path.abspath(__file__)
    for _ in range(5):
        current = os.path.dirname(current)
        candidate = os.path.join(current, ".env")
        if os.path.exists(candidate):
            load_dotenv(candidate)
            return
        tp_candidate = os.path.join(current, "third_party", ".env")
        if os.path.exists(tp_candidate):
            load_dotenv(tp_candidate)
            return


def _is_thinking_model(model: str) -> bool:
    """若模型名称表明它是推理型模型（Qwen3 / DeepSeek-R1），返回 True。"""
    lower = model.lower()
    return any(fragment in lower for fragment in _THINKING_MODELS)


def _needs_openai_prefix(model: str) -> bool:
    """若模型名称不含已知提供商前缀，则需要追加 ``openai/`` 前缀（LiteLLM 路由规则）。"""
    return not any(model.startswith(p) for p in _LITELLM_PROVIDER_PREFIXES)


def _build_thinking_extra_body(
    resolved_thinking: Optional[bool],
    base_url: Optional[str],
) -> dict:
    """构建推理模型思考模式控制参数的 ``extra_body`` 字典。

    - 有 ``base_url``（vLLM 部署）时：嵌套在 ``chat_template_kwargs`` 下。
    - 无 ``base_url``（云端 API）时：放顶层。
    - ``resolved_thinking`` 为 ``None`` 时：返回空字典。
    """
    if resolved_thinking is None:
        return {}
    if base_url:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": resolved_thinking}}}
    return {"extra_body": {"enable_thinking": resolved_thinking}}


def bridge_parlant_litellm_env() -> dict:
    """将 .env 中的 LLM 配置（QWEN3_*/LLM_*/OPENAI_*）桥接到 Parlant LiteLLM 所需环境变量。

    Parlant 使用 LiteLLM 作为 NLP 后端时读取 ``LITELLM_PROVIDER_*`` 环境变量。
    本函数将现有 .env 中的 LLM 配置自动回填，优先保留用户已显式设置的 ``LITELLM_*`` 值。

    Returns:
        已解析并写入环境的 ``LITELLM_*`` 键值对。
    """
    _load_dotenv()
    resolved: dict = {}

    if base_url := os.environ.get("LITELLM_PROVIDER_BASE_URL") or _first_env([
        "QWEN3_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "QWEN_BASE_URL",
    ]):
        os.environ.setdefault("LITELLM_PROVIDER_BASE_URL", base_url)
        resolved["LITELLM_PROVIDER_BASE_URL"] = os.environ["LITELLM_PROVIDER_BASE_URL"]

    if api_key := os.environ.get("LITELLM_PROVIDER_API_KEY") or _first_env([
        "QWEN3_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY",
    ]):
        os.environ.setdefault("LITELLM_PROVIDER_API_KEY", api_key)
        resolved["LITELLM_PROVIDER_API_KEY"] = os.environ["LITELLM_PROVIDER_API_KEY"]

    if model := os.environ.get("LITELLM_PROVIDER_MODEL_NAME") or _first_env([
        "QWEN3_MODEL", "LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL", "QWEN_MODEL",
    ]):
        if _needs_openai_prefix(model):
            model = f"openai/{model}"
        os.environ.setdefault("LITELLM_PROVIDER_MODEL_NAME", model)
        resolved["LITELLM_PROVIDER_MODEL_NAME"] = os.environ["LITELLM_PROVIDER_MODEL_NAME"]

    return resolved


def _configure_litellm(model_name: str) -> None:
    """配置 LiteLLM 全局状态，抑制不必要的日志和警告。

    - 禁用 success/failure 回调（抑制 async handler GC 噪音）
    - 注册本地模型零价格条目（消除 "not in cost map" DEBUG 日志）
    - 将 LiteLLM logger 提升至 WARNING 级别
    - 关闭 SSL 证书校验（适配内网自签名证书 / HTTP vLLM 端点）
      可通过环境变量 ``LLM_SSL_VERIFY=true`` 开启校验
    """
    _litellm_logger = logging.getLogger("LiteLLM")
    _litellm_logger.propagate = False
    _litellm_logger.setLevel(logging.WARNING)

    try:
        import litellm  # type: ignore[import]

        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []
        litellm.suppress_debug_info = True

        _ssl_verify_str = os.environ.get("LLM_SSL_VERIFY", "false").lower()
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
          ``QWEN3_*`` > ``LLM_*`` > ``OPENAI_*`` > ``DEEPSEEK_*``

        与 ``third_party/.env`` 中的 ``QWEN3_*`` 命名完全兼容。

        Raises:
            ValueError: 若无法找到任何 API Key。
        """
        _load_dotenv()

        api_key = _first_env([
            "QWEN3_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY", "QWEN_API_KEY",
        ])
        if not api_key:
            raise ValueError(
                "缺少 LLM API Key。\n"
                "请在 .env 文件或环境变量中设置 QWEN3_API_KEY 或 LLM_API_KEY。"
            )

        base_url = _first_env([
            "QWEN3_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL",
            "DEEPSEEK_BASE_URL", "QWEN_BASE_URL",
        ])
        model = _first_env([
            "QWEN3_MODEL", "LLM_MODEL", "OPENAI_MODEL",
            "DEEPSEEK_MODEL", "QWEN_MODEL",
        ], default=DEFAULT_MODEL)

        temperature_str = _first_env(["QWEN3_TEMPERATURE", "LLM_TEMPERATURE", "OPENAI_TEMPERATURE"])
        temperature = float(temperature_str) if temperature_str else DEFAULT_TEMPERATURE

        max_tokens = (
            int(v)
            if (v := _first_env(["QWEN3_MAX_TOKENS", "LLM_MAX_TOKENS", "MAX_TOKENS"]))
            else DEFAULT_MAX_TOKENS
        )

        enable_thinking_str = _first_env(["LLM_ENABLE_THINKING", "ENABLE_THINKING"])
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

    特性：
      - 支持 OpenAI / vLLM / DeepSeek 等 OpenAI 兼容端点
      - 推理模型（Qwen3 / DeepSeek-R1）自动禁用思考模式
      - 自动添加 ``openai/`` 路由前缀（LiteLLM 路由规则）
      - 剥除响应中的 ``<think>...</think>`` 块
      - 连接失败时给出友好提示

    依赖::

        pip install litellm

    用法::

        from tools.llm import LLMClient, LLMConfig

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
                "LLMClient 需要安装 litellm：\n    pip install litellm"
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
            "[LLMClient] 初始化完成  model=%s  base_url=%s",
            self._config.model, self._config.base_url,
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

        Raises:
            RuntimeError: LLM 服务不可达（连接超时/拒绝）。
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
        # 剥除推理模型（Qwen3 / DeepSeek-R1）输出的 <think>...</think> 块
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()
