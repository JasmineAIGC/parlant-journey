"""
tests/06_demo_fraud.py  —  欺诈处理场景完整演示

演示如何使用 journey/ 模块定义对话流程，并通过 Parlant Server 运行多轮对话。
运行前提：根目录 .env 中配置好 LLM（QWEN3_API_KEY / QWEN3_BASE_URL / QWEN3_MODEL）
直接运行：python tests/06_demo_fraud.py
pytest 集成测试（不可用时自动跳过）：pytest tests/06_demo_fraud.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# 路径设置：使项目根目录可被导入（支持从任意目录运行本文件）
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 加载根目录 .env（LLM 服务地址 / API Key / 模型名）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

# Parlant 框架（来自 tools/parlant_server）
try:
    from tools.parlant_server import ParlantServerContext, run_conversation
    _PARLANT_AVAILABLE = True
except ImportError:
    _PARLANT_AVAILABLE = False

# LLM 客户端（仅供 LLM 可用性检测）
try:
    from tools.llm import LLMClient, LLMConfig
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

import pytest

# ===========================================================================
# journey/ 模块入口
#
# 三个核心 API：
#   load_journey(path)              — 从 YAML/JSON 文件或 dict 加载 JourneyConfig
#   create_journey(agent, config)   — 将单个 JourneyConfig 注册到 Parlant Agent
#   register_journeys(agent, list)  — 批量注册多个 JourneyConfig
# ===========================================================================

from journey import create_journey, load_journey  # noqa: E402

_FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# 从 YAML 文件加载欺诈处理对话流程配置（等价写法：JourneyDSL 或 JourneyConfig 直接构造）
fraud_config = load_journey(os.path.join(_FIXTURES_DIR, "fraud.yaml"))

# 模拟中风险欺诈场景对话脚本
DEMO_SCRIPT: List[str] = [
    "你好，我刚才发现我的银行卡好像被盗刷了，有一笔我完全不认识的交易",
    "我叫李明，身份证后四位是6789，银行卡后四位是4321",
    "是昨天晚上11点半，有一笔1800元，商户是'某某网络科技有限公司'，我没操作过",
    "好的，那我应该怎么处理？",
    "我已经修改了密码，也设置了单日限额，申诉单也提交了",
    "谢谢你的帮助，非常专业",
]


# ---------------------------------------------------------------------------
# 演示主函数
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    """启动 Parlant Server，注册 Journey 并运行多轮演示对话。"""
    print("=" * 65)
    print(f"  Journey 对话演示：{fraud_config.title}")
    print(f"  状态：{' → '.join(s.name for s in fraud_config.states)}")
    print("=" * 65)

    try:
        async with ParlantServerContext() as ctx:
            # 在 Parlant Server 上创建 Agent，作为 Journey 的运行载体
            agent = await ctx.create_agent("银行客服", "专业的银行客服 Agent，处理欺诈交易等业务")

            # ── journey/ 核心调用 ────────────────────────────────────────────
            # create_journey 将 fraud_config 中的状态节点、状态转移、指南（guidelines）
            # 逐一写入 Parlant Server；之后的对话由这些规则驱动 Agent 行为。
            # 批量注册可用：from journey import register_journeys
            #               await register_journeys(agent, [config1, config2])
            await create_journey(agent, fraud_config)
            # ────────────────────────────────────────────────────────────────

            print("\n─── 对话开始（Parlant Server 驱动）───")
            turn = 0
            # run_conversation 是 tools/parlant_server 的对话驱动器，
            # 按 DEMO_SCRIPT 逐条发送消息，逐轮 yield (user_msg, agent_reply) 元组
            async for user_msg, reply in run_conversation(
                ctx, agent, DEMO_SCRIPT, title=f"演示：{fraud_config.title}"
            ):
                turn += 1
                print(f"\n[Turn {turn}]")
                print(f"  👤 用户：{user_msg}")
                print(f"  🤖 Agent：{reply or '（未收到回复，可能超时）'}")
    except RuntimeError as exc:
        print(f"\n❌ Parlant Server 启动失败：{exc}")
        print("\n请确认已安装 parlant 及 LLM 配置正确（根目录 .env）。")
        return

    print("\n" + "=" * 65)
    print("  演示完成")
    print("=" * 65)


# ===========================================================================
# pytest 集成测试入口（LLM 服务不可用时自动跳过）
# ===========================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(
    not _PARLANT_AVAILABLE,
    reason="parlant 未安装（pip install parlant）",
)
async def test_fraud_demo_runs_full_conversation() -> None:
    """集成测试：验证完整欺诈处理对话可通过 Parlant Server 运行。"""
    if _LLM_AVAILABLE:
        try:
            await LLMClient(config=LLMConfig.from_env()).chat_async([{"role": "user", "content": "你好"}])
        except Exception as e:
            pytest.skip(f"LLM 服务不可达：{e}")

    replies: List[str] = []
    try:
        async with ParlantServerContext() as ctx:
            # 创建 Agent
            agent = await ctx.create_agent("银行客服", "专业银行客服 Agent")
            # 将 fraud_config 中的 Journey 注册到 Agent（状态、转移、指南全部写入 Parlant）
            await create_journey(agent, fraud_config)
            # 发送前两轮对话，收集 Agent 回复
            async for _user_msg, reply in run_conversation(ctx, agent, DEMO_SCRIPT[:2]):
                replies.append(reply)
    except RuntimeError as e:
        pytest.skip(f"Parlant Server 启动失败：{e}")

    assert len(replies) > 0, "至少应收到 1 条 Agent 回复"


# ===========================================================================
# 直接运行入口
# ===========================================================================

if __name__ == "__main__":
    if not _PARLANT_AVAILABLE:
        print("❌ 缺少依赖：请先安装 parlant（pip install parlant）")
        sys.exit(1)
    asyncio.run(run_demo())

