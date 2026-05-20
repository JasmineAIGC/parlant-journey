"""
demos/fraud/demo_fraud.py  —  欺诈处理场景完整演示

运行前提：项目根目录 .env 中配置好 LLM。
直接运行：python demos/fraud/demo_fraud.py
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

from journey import create_journey, load_journey  # noqa: E402
from tools.parlant_server import ParlantServerContext  # noqa: E402
from tools.parlant_server.session import run_conversation  # noqa: E402

fraud_config = load_journey(os.path.join(_DEMO_DIR, "fraud.yaml"))

DEMO_SCRIPT: List[str] = [
    "你好，我刚才发现我的银行卡好像被盗刷了，有一笔我完全不认识的交易",
    "我叫李明，身份证后四位是6789，银行卡后四位是4321",
    "是昨天晚上11点半，有一笔1800元，商户是'某某网络科技有限公司'，我没操作过",
    "好的，那我应该怎么处理？",
    "我已经修改了密码，也设置了单日限额，申诉单也提交了",
    "谢谢你的帮助，非常专业",
]


async def run_demo() -> None:
    """启动 Parlant Server，注册 Journey 并运行多轮演示对话。"""
    print("=" * 65)
    print(f"  Journey 对话演示：{fraud_config.title}")
    print(f"  状态：{' → '.join(s.name for s in fraud_config.states)}")
    print("=" * 65)

    try:
        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent("银行客服", "专业的银行客服 Agent，处理欺诈交易等业务")
            await create_journey(agent, fraud_config)

            print("\n─── 对话开始（Parlant Server 驱动）───")
            turn = 0
            async for user_msg, reply in run_conversation(
                ctx, agent, DEMO_SCRIPT, title=f"演示：{fraud_config.title}"
            ):
                turn += 1
                print(f"\n[Turn {turn}]")
                print(f"  👤 用户：{user_msg}")
                print(f"  🤖 Agent：{reply or '（未收到回复，可能超时）'}")
    except RuntimeError as exc:
        print(f"\n❌ Parlant Server 启动失败：{exc}")
        print("\n请确认已安装 parlant 且根目录 .env 中 LLM 配置正确。")
        return

    print("\n" + "=" * 65)
    print("  演示完成")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_demo())
