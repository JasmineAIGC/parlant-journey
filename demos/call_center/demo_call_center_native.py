"""
demos/call_center/demo_call_center_native.py  —  银行智能呼叫中心（Parlant 原生 SDK）

与 demo_call_center.py 的差异：
  Journey 创建方式完全相同（load_journey + register_journeys from journey/），
  但 Parlant Server 的启动、Agent 创建均直接使用 parlant.sdk 原生 API，
  不再依赖自定义的 ParlantServerContext / run_conversation。
  对话通过 Parlant 内置浏览器 UI 进行。

运行：
    python demos/call_center/demo_call_center_native.py

服务器就绪后，在浏览器中访问：
    http://localhost:8800
"""

from __future__ import annotations

import asyncio
import os
import sys

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_DEMO_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

import parlant.sdk as p  # noqa: E402

from journey import load_journey, register_journeys  # noqa: E402
from tools.llm import prepare_litellm_environment  # noqa: E402

PORT = 8800

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


async def run_demo() -> None:
    """启动 Parlant Server，注册 7 个策略 Journey，开放浏览器 UI 进行对话。"""
    print("=" * 72)
    print("  Journey 对话演示：银行智能呼叫中心（Parlant 原生 SDK）")
    print("  策略：" + " | ".join(cfg.title for cfg in decision_configs))
    print("=" * 72)

    # 桥接 .env 中的 LLM 配置到 Parlant LiteLLM 服务所需的环境变量
    prepare_litellm_environment()

    async with p.Server(
        nlp_service=p.NLPServices.litellm,
        port=PORT,
    ) as server:
        agent = await server.create_agent(
            name="银行呼叫中心智能客服",
            description="依据 docs/rules.md 输出银行呼叫中心应答策略 JSON",
        )
        await register_journeys(agent, decision_configs)

        print(f"\n✅ {agent.name} 配置完成（{len(decision_configs)} 个 Journey 已注册）")
        print(f"🌐 浏览器 UI 即将就绪，请访问：http://localhost:{PORT}")
        print("   在对话框中输入消息即可与 Agent 交互。")
        print("   按 Ctrl+C 停止服务器。\n")
    # async with 正常退出时，parlant.sdk.Server.__aexit__ 启动 uvicorn 并持续监听，
    # 直到收到 Ctrl+C 信号才返回。


if __name__ == "__main__":
    asyncio.run(run_demo())
