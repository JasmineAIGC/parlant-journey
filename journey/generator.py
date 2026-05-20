"""
journey.generator
~~~~~~~~~~~~~~~~~

LLM 驱动的 Journey 配置自动生成器。

根据用户输入的自然语言（业务描述、对话示例、规则要求等），
调用 LLM 分析并生成符合 JourneyConfig schema 的 YAML / JSON 配置文件，
实现新业务场景的"一键接入"。

**LLM 调用能力由 ``tools.llm`` 提供（LLMClient / LLMConfig）。**

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
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

from .loader import load_journey_dict
from .models import JourneyConfig

# ---------------------------------------------------------------------------
# 从 tools.llm 导入 LLM 客户端（单一来源，消除代码重复）
# ---------------------------------------------------------------------------
from tools.llm.client import (  # noqa: E402
    LLMClient,
    LLMConfig,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    _is_thinking_model,
    _needs_openai_prefix,
    _build_thinking_extra_body,
    _configure_litellm,
    _first_env,
    _load_dotenv,
)


_SCHEMA_DESC = """\
{
  "title": "Journey 标题（字符串）",
  "description": "Journey 整体描述（字符串）",
  "conditions": ["触发此 Journey 的用户意图或关键词，至少 2 条"],
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
6. title、description、conditions 必须填写，不得为空。
7. 规则（guidelines）的 priority 范围 0–100，数值越大越优先。
"""


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
# 内部工具函数（journey-specific）
# ---------------------------------------------------------------------------

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
