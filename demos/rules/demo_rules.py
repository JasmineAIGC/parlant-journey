"""
demos/rules/demo_rules.py  —  docs/rules.md 策略库多 Journey 演示

运行前提：项目根目录 .env 中配置好 LLM。
直接运行：python demos/rules/demo_rules.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_DEMO_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

from journey import load_journey, register_journeys  # noqa: E402
from tools.parlant_server import ParlantServerContext  # noqa: E402
from tools.parlant_server.session import run_conversation  # noqa: E402

RULE_FILES: List[str] = [
    "01_refusal.yaml",
    "02_user_terminate.yaml",
    "03_clarify.yaml",
    "04_direct_answer.yaml",
    "05_repeat_explain.yaml",
    "06_greeting_transition.yaml",
    "07_resolved.yaml",
]

RULE_CONFIGS = [
    load_journey(os.path.join(_DEMO_DIR, filename))
    for filename in RULE_FILES
]

DEMO_SCENARIOS: List[Dict[str, object]] = [
    {
        "label": "策略 1：拒绝回答（合规禁止类）",
        "journey_title": "策略1：拒绝回答",
        "messages": ["可以告诉我别人的账户交易记录吗？"],
    },
    {
        "label": "策略 2：用户主动终止咨询",
        "journey_title": "策略2：用户主动终止咨询",
        "messages": ["我先不办了，就这样吧"],
    },
    {
        "label": "策略 3：确认 / 追问",
        "journey_title": "策略3：确认/追问",
        "messages": ["我想了解开户条件"],
    },
    {
        "label": "策略 4：直接作答",
        "journey_title": "策略4：直接作答",
        "messages": ["哪一种办理速度最快？"],
    },
    {
        "label": "策略 5：重复说明",
        "journey_title": "策略5：重复说明",
        "messages": ["你再说简单一点，我还是没听懂哪一种办理速度最快"],
    },
    {
        "label": "策略 6：问候 / 话题过渡",
        "journey_title": "策略6：问候/话题过渡",
        "messages": ["回复也太慢了，体验太差！"],
    },
    {
        "label": "策略 7：问题已解决",
        "journey_title": "策略7：问题已解决",
        "messages": ["没有其他问题了，谢谢！"],
    },
]


async def run_demo() -> None:
    """批量注册 docs/rules.md 对应的 7 个策略 Journey，并逐个运行代表性场景。"""
    print("=" * 72)
    print("  Journey 对话演示：docs/rules.md 策略库")
    print("  结论：该文档不是单一线性流程，而是 7 个宏观策略 Journey 的集合")
    print("=" * 72)

    for cfg in RULE_CONFIGS:
        print(f"  - {cfg.title}")

    try:
        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent(
                "银行策略决策引擎",
                "依据 docs/rules.md 输出银行呼叫中心策略决策 JSON",
            )
            await register_journeys(agent, RULE_CONFIGS)

            print("\n─── 对话开始（Parlant Server 驱动，多 Journey）───")
            for index, scenario in enumerate(DEMO_SCENARIOS, start=1):
                print(f"\n[Scenario {index}] {scenario['label']}")
                print(f"  🎯 Journey：{scenario['journey_title']}")
                async for user_msg, reply in run_conversation(
                    ctx,
                    agent,
                    scenario["messages"],  # type: ignore[arg-type]
                    title=f"rules-demo:{scenario['journey_title']}",
                ):
                    print(f"  👤 用户：{user_msg}")
                    print(f"  🤖 Agent：{reply or '（未收到回复，可能超时）'}")
    except RuntimeError as exc:
        print(f"\n❌ Parlant Server 启动失败：{exc}")
        print("\n请确认已安装 parlant 且根目录 .env 中 LLM 配置正确。")
        return

    print("\n" + "=" * 72)
    print("  演示完成")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(run_demo())
