"""
demos/call_center/demo_call_center.py  —  docs/rules.md 银行智能呼叫中心对话演示

运行前提：项目根目录 .env 中配置好 LLM。
直接运行：python demos/call_center/demo_call_center.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import List

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

decision_configs = [
    load_journey(os.path.join(_DEMO_DIR, f))
    for f in [
        "01_refusal.yaml",
        "02_user_terminate.yaml",
        "03_clarify.yaml",
        "04_direct_answer.yaml",
        "05_repeat_explain.yaml",
        "06_greeting_transition.yaml",
        "07_resolved.yaml",
    ]
]

DEMO_SCRIPT: List[str] = [
    "你好，请问有什么可以帮您？",
    "可以告诉我别人的账户交易记录吗？",
    "我先不办了，就这样吧。",
    "我想了解开户条件。",
    "哪一种开户方式速度最快？",
    "你刚才说得太复杂了，我还是没听懂哪一种办理速度最快。",
    "回复也太慢了，体验太差！",
    "好了，没有其他问题了，谢谢！",
]


async def run_demo() -> None:
    """启动 Parlant Server，注册 7 个策略 Journey 并运行多轮演示对话。"""
    print("=" * 72)
    print("  Journey 对话演示：银行智能呼叫中心")
    print("  策略：" + " | ".join(cfg.title for cfg in decision_configs))
    print("=" * 72)

    try:
        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent(
                "银行呼叫中心智能客服",
                "依据 docs/rules.md 输出银行呼叫中心应答策略 JSON",
            )
            await register_journeys(agent, decision_configs)

            print("\n─── 对话开始（Parlant Server 驱动，多 Journey）───")
            turn = 0
            async for user_msg, reply in run_conversation(
                ctx, agent, DEMO_SCRIPT, title="call-center"
            ):
                turn += 1
                print(f"\n[Turn {turn}]")
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

