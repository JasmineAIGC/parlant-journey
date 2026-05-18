"""
journey.generator
~~~~~~~~~~~~~~~~~

LLM 驱动的 Journey 配置自动生成器。

根据用户输入的自然语言（业务描述、对话示例、规则要求等），
调用 LLM 分析并生成符合 JourneyConfig schema 的 YAML / JSON 配置文件，
实现新业务场景的"一键接入"。

**LLM 调用基于 LiteLLM，参考 third_party/model_client/api/llm.py 实现，
但完全独立于 third_party/ 路径，不导入其中任何代码。**

依赖::

    pip install litellm python-dotenv pyyaml

环境变量（与 third_party/.env 中的 QWEN3_* 命名完全兼容）::

    QWEN3_API_KEY   或  LLM_API_KEY    （必填）
    QWEN3_BASE_URL  或  LLM_BASE_URL   （可选，默认 OpenAI 官方地址）
    QWEN3_MODEL     或  LLM_MODEL      （可选，默认 Qwen2.5-7B-Instruct）

命令行用法::

    # 输出到控制台（YAML 格式）
    python -m journey.generator "我需要一个贷款申请流程，包含资料收集、审核和放款三个步骤"

    # 指定输出文件和格式
    python -m journey.generator "..." --output loan.yaml
    python -m journey.generator "..." --output loan.json

Python API 用法::

    import asyncio
    from journey.generator import JourneyGenerator

    gen = JourneyGenerator()
    config, content = asyncio.run(gen.generate("贷款申请流程：资料收集 → 审核 → 放款"))
    print(content)                          # 输出 YAML 字符串
    gen.save("loan.yaml", content)          # 保存到文件
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from .loader import load_journey_dict
from .models import JourneyConfig

# ---------------------------------------------------------------------------
# 常量（与 third_party/model_client/api/llm.py 保持一致）
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
# 内部工具函数（参考 third_party/model_client/api/llm.py）
# ---------------------------------------------------------------------------

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
        logger.debug("[generator] litellm 配置失败（非致命）: %s", exc, exc_info=True)


_SCHEMA_DESC = """\
{
  "title": "Journey 标题（字符串）",
  "description": "Journey 整体描述（字符串）",
  "triggers": ["触发此 Journey 的用户意图或关键词，至少 2 条"],
  "states": [
    {
      "name": "状态唯一 ID（英文小写+下划线）",
      "description": "可读描述（中文，1 句话）",
      "chat_state": "Agent 在此状态下的行为指令（中文，1-3 句话）",
      "guidelines": [                        // 可选，节点级规则
        {"condition": "触发条件", "action": "执行动作", "priority": 80}
      ],
      "canned_response": {                   // 可选，合规话术
        "template": "锁定输出的话术模板",
        "strict": true
      }
    }
  ],
  "transitions": [                           // 按拓扑顺序排列
    {"source": "源状态 name", "target": "目标状态 name", "condition": "转移条件（可选）"}
  ],
  "global_guidelines": [                     // 可选，全局规则
    {"condition": "全局触发条件", "action": "全局执行动作", "priority": 60}
  ]
}"""

_SYSTEM_PROMPT = f"""\
你是 Journey 配置生成专家，专门为 Parlant 对话 AI 框架设计业务流程。

任务：根据用户提供的业务场景描述，生成一份完整的 Journey 配置。

输出要求：
1. 只输出 JSON，不要添加任何说明文字、markdown 代码块标记或其他内容。
2. JSON 结构必须严格符合以下 schema（字段含义见注释）：
{_SCHEMA_DESC}
3. states 中每个状态必须有 name（英文小写+下划线）和 chat_state（中文）。
4. transitions 按拓扑顺序排列（确保父节点的出边先于子节点的出边）。
5. 终止状态（无出边的状态）通常命名为 closing 或 end，作为流程收尾。
6. title、description、triggers 必须填写，不得为空。
7. 规则（guidelines）的 priority 范围 0–100，数值越大越优先。
"""


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """连接 LLM 服务所需的运行时配置。

    参考 third_party/model_client/api/llm.py 中的 LLMConfig 实现，
    但独立于 third_party/ 路径，不导入其中任何代码。

    通常通过 :meth:`from_env` 从环境变量自动加载，
    也可在代码中直接构造。

    Args:
        api_key:         LLM 服务 API Key（必填）。
        model:           使用的模型名称。
        base_url:        OpenAI 兼容 API 的 base URL；None 表示使用 OpenAI 官方地址。
        temperature:     生成温度（0.0 = 确定性输出，推荐结构化 JSON 生成时使用）。
        max_tokens:      最大输出 token 数。
        enable_thinking: 控制推理模型的思考模式（Qwen3 / DeepSeek-R1）。

            - ``None``  — 对已知推理模型自动设为 ``False``（禁用链式推理输出）
            - ``False`` — 强制禁用思考
            - ``True``  — 强制启用思考
    """

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    enable_thinking: Optional[bool] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量（或 .env 文件）加载配置。

        读取优先级（高 → 低），与 ``third_party/.env`` 命名完全兼容：
          ``QWEN3_*`` > ``LLM_*`` > ``OPENAI_*`` > ``DEEPSEEK_*``

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

        temperature_str = _first_env(["QWEN3_TEMPERATURE", "LLM_TEMPERATURE"])
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
# LLMClient — 基于 LiteLLM 的轻量级异步客户端
# ---------------------------------------------------------------------------

class LLMClient:
    """基于 LiteLLM 的轻量级异步 LLM 客户端。

    参考 third_party/model_client/api/llm.py 中的 LLMClient 实现，
    但独立于 third_party/ 路径，不导入其中任何代码。

    特性：
      - 支持 OpenAI / vLLM / DeepSeek 等 OpenAI 兼容端点
      - 推理模型（Qwen3 / DeepSeek-R1）自动禁用思考模式
      - 自动添加 ``openai/`` 路由前缀（LiteLLM 路由规则）
      - 剥除响应中的 ``<think>...</think>`` 块
      - 连接失败时给出友好提示

    依赖::

        pip install litellm

    用法::

        from journey.generator import LLMClient, LLMConfig

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
        """发送一次 LLM 对话请求，返回已清洁的助手回复文本。

        Args:
            messages:    OpenAI 格式消息列表（``[{"role": ..., "content": ...}]``）。
            temperature: 覆盖 ``LLMConfig.temperature``；``None`` 则使用配置中的值。

        Returns:
            剥除 ``<think>`` 块后的助手回复文本。

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


# ---------------------------------------------------------------------------
# JourneyGenerator
# ---------------------------------------------------------------------------

class JourneyGenerator:
    """LLM 驱动的 Journey 配置生成器。

    工作流程：
      1. 将用户输入的业务描述发送给 LLM（附带结构化 schema 指引）。
      2. LLM 输出符合 JourneyConfig schema 的 JSON 字符串。
      3. 解析并验证 JSON，构建 :class:`~journey.models.JourneyConfig` 对象。
      4. 可选择将结果序列化为 YAML 或 JSON 并写入文件。

    内部使用 :class:`LLMClient`（基于 LiteLLM），不依赖 third_party/ 中的任何代码。

    用法::

        gen = JourneyGenerator()                    # 从环境变量读取配置
        config, text = await gen.generate("...")    # 返回 (JourneyConfig, yaml_str)
        gen.save("output.yaml", text)               # 保存到文件
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None) -> None:
        """
        Args:
            llm_config: LLM 配置；为 None 时调用 :meth:`LLMConfig.from_env` 自动加载。

        Raises:
            ImportError: 若未安装 ``litellm``。
        """
        self._llm = LLMClient(llm_config)
        self._config = self._llm._config

    async def generate(
        self,
        user_input: str,
        fmt: str = "yaml",
        extra_instruction: Optional[str] = None,
    ) -> Tuple[JourneyConfig, str]:
        """根据用户输入生成 Journey 配置。

        Args:
            user_input:         业务场景的自然语言描述（对话示例、需求列表、规则等）。
            fmt:                输出格式，``"yaml"`` 或 ``"json"``（默认 ``"yaml"``）。
            extra_instruction:  追加到 system prompt 末尾的额外指令（可选）。

        Returns:
            ``(config, text)`` 元组：
              - config: 解析完成的 :class:`~journey.models.JourneyConfig`。
              - text:   对应格式的序列化字符串（可直接写入文件）。

        Raises:
            ValueError:  LLM 返回的内容无法解析为合法的 Journey 配置。
            RuntimeError: LLM 服务不可达或返回错误。
        """
        system = _SYSTEM_PROMPT
        if extra_instruction:
            system += f"\n\n额外要求：\n{extra_instruction}"

        raw = await self._call_llm(system, user_input)
        data = _extract_json(raw)
        config = load_journey_dict(data)
        text = _serialize(data, fmt)
        return config, text

    def save(self, path: str, content: str) -> None:
        """将配置内容写入文件。

        Args:
            path:    输出文件路径（扩展名决定格式，建议 ``.yaml`` 或 ``.json``）。
            content: :meth:`generate` 返回的序列化字符串。
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    async def generate_to_file(
        self,
        user_input: str,
        output_path: str,
        extra_instruction: Optional[str] = None,
    ) -> JourneyConfig:
        """生成配置并直接写入文件的便捷方法。

        文件格式由 ``output_path`` 的扩展名自动推断：
          ``.yaml`` / ``.yml`` → YAML；其余 → JSON。

        Args:
            user_input:        业务场景描述。
            output_path:       输出文件路径。
            extra_instruction: 追加额外指令（可选）。

        Returns:
            解析完成的 :class:`~journey.models.JourneyConfig`。
        """
        ext = os.path.splitext(output_path)[1].lower()
        fmt = "yaml" if ext in (".yaml", ".yml") else "json"
        config, text = await self.generate(user_input, fmt=fmt,
                                           extra_instruction=extra_instruction)
        self.save(output_path, text)
        return config

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _call_llm(self, system: str, user: str) -> str:
        """调用 LLM 并返回回复文本（已剥除 think 块）。"""
        return await self._llm.chat_async([
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ])


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------

async def generate_journey_config(
    user_input: str,
    fmt: str = "yaml",
    output_path: Optional[str] = None,
    llm_config: Optional[LLMConfig] = None,
    extra_instruction: Optional[str] = None,
) -> Tuple[JourneyConfig, str]:
    """根据用户输入生成 Journey 配置的便捷函数（无需手动创建 JourneyGenerator）。

    Args:
        user_input:        业务场景的自然语言描述。
        fmt:               输出格式，``"yaml"`` 或 ``"json"``（默认 ``"yaml"``）。
        output_path:       若指定，将配置写入该路径（格式由扩展名决定）。
        llm_config:        LLM 配置；为 None 时从环境变量自动读取。
        extra_instruction: 追加的额外生成指令（可选）。

    Returns:
        ``(config, text)`` 元组。
    """
    gen = JourneyGenerator(llm_config)
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        fmt = "yaml" if ext in (".yaml", ".yml") else "json"
        config, text = await gen.generate(user_input, fmt=fmt,
                                          extra_instruction=extra_instruction)
        gen.save(output_path, text)
        return config, text
    return await gen.generate(user_input, fmt=fmt, extra_instruction=extra_instruction)


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

    **注意：** 若包以 ``pip install`` 方式安装（site-packages），
    则依赖方式 1 或 3 找到项目根目录的 ``.env``。
    开发模式下方式 2 兼容 ``third_party/.env`` 命名惯例。
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        return  # python-dotenv 未安装，跳过

    # 方式1：当前工作目录 .env
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, ".env")):
        load_dotenv(os.path.join(cwd, ".env"))
        return

    # 方式2：当前工作目录下的 third_party/.env（开发模式惯例）
    third_party_env = os.path.join(cwd, "third_party", ".env")
    if os.path.exists(third_party_env):
        load_dotenv(third_party_env)
        return

    # 方式3：从当前工作目录逐级向上查找 .env（最多 5 层）
    search_dir = cwd
    for _ in range(5):
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break  # 已到达根目录
        candidate = os.path.join(parent, ".env")
        if os.path.exists(candidate):
            load_dotenv(candidate)
            return
        search_dir = parent


def _extract_json(text: str) -> dict:
    """从 LLM 回复中提取并解析 JSON 对象。

    兼容两种情况：
      1. LLM 直接返回纯 JSON
      2. LLM 将 JSON 包裹在 markdown 代码块中（```json ... ```）

    Raises:
        ValueError: 无法从回复中提取合法 JSON。
    """
    # 尝试剥除 markdown 代码块标记
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # 提取第一个 { ... } 块（兼容 LLM 在 JSON 前后附加说明文字的情况）
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(
            "LLM 返回的内容中未找到合法的 JSON 对象。\n"
            f"原始回复（前 500 字符）：\n{text[:500]}"
        )
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 解析失败：{exc}\n"
            f"原始回复（前 500 字符）：\n{text[:500]}"
        ) from exc


def _serialize(data: dict, fmt: str) -> str:
    """将字典序列化为 YAML 或 JSON 字符串。

    Args:
        data: 已解析的 Journey 配置字典（不含 ``_comment`` 字段）。
        fmt:  ``"yaml"`` 或 ``"json"``。

    Returns:
        序列化后的字符串（UTF-8 编码，中文不转义）。
    """
    # 过滤掉注释字段，避免被当作配置字段处理
    clean = {k: v for k, v in data.items() if not k.startswith("_")}

    if fmt == "json":
        return json.dumps(clean, ensure_ascii=False, indent=2)

    # YAML 序列化
    try:
        import yaml  # type: ignore[import]
        return yaml.dump(clean, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except ImportError:
        raise ImportError(
            "输出 YAML 格式需要安装 pyyaml：\n"
            "    pip install pyyaml\n"
            "或使用 fmt='json' 改为 JSON 输出。"
        )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _build_arg_parser():
    """构建命令行参数解析器。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m journey.generator",
        description="根据自然语言描述自动生成 Journey 配置文件（YAML / JSON）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例：
  # 直接输出 YAML 到控制台
  python -m journey.generator "贷款申请流程：资料收集 → 信用审核 → 放款通知"

  # 生成到文件（格式由扩展名决定）
  python -m journey.generator "..." --output loan.yaml
  python -m journey.generator "..." --output loan.json

  # 附加额外要求
  python -m journey.generator "..." --extra "状态不超过5个，规则使用中文"

  # 从文件读取需求描述
  python -m journey.generator --input requirements.txt --output journey.yaml
""",
    )
    parser.add_argument(
        "description",
        nargs="?",
        help="业务场景的自然语言描述（与 --input 二选一）",
    )
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="从文件读取业务描述（与位置参数二选一）",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="输出文件路径（.yaml/.yml → YAML，其余 → JSON）；不指定则打印到控制台",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["yaml", "json"],
        default="yaml",
        help="输出格式（当未指定 --output 时生效，默认 yaml）",
    )
    parser.add_argument(
        "--extra", "-e",
        metavar="INSTRUCTION",
        help="追加给 LLM 的额外生成要求（如：'状态不超过5个'）",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="覆盖环境变量中的模型名称",
    )
    return parser


async def _run_cli(args) -> None:
    """CLI 主逻辑（async）。"""
    # 读取用户输入
    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            user_input = fh.read().strip()
    elif args.description:
        user_input = args.description
    else:
        print("错误：请提供业务描述（位置参数）或通过 --input 指定输入文件。", file=sys.stderr)
        sys.exit(1)

    if not user_input:
        print("错误：业务描述不能为空。", file=sys.stderr)
        sys.exit(1)

    # 构建 LLM 配置（允许 --model 覆盖环境变量）
    try:
        llm_cfg = LLMConfig.from_env()
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

    if args.model:
        llm_cfg.model = args.model

    # 确定输出格式
    fmt = args.format
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        fmt = "yaml" if ext in (".yaml", ".yml") else "json"

    # 生成配置
    gen = JourneyGenerator(llm_cfg)
    print(f"🤖  正在调用 LLM（{llm_cfg.model}）生成 Journey 配置...", file=sys.stderr)
    try:
        config, text = await gen.generate(
            user_input, fmt=fmt, extra_instruction=args.extra
        )
    except (ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

    # 输出结果
    if args.output:
        gen.save(args.output, text)
        print(f"✅  Journey 配置已保存至：{args.output}", file=sys.stderr)
        print(f"   状态数：{len(config.states)}  转移数：{len(config.transitions)}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    parser = _build_arg_parser()
    _args = parser.parse_args()
    asyncio.run(_run_cli(_args))
