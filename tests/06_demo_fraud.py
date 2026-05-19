"""
tests/06_demo_fraud.py  —  欺诈处理场景完整演示

运行前提：根目录 .env 中配置好 LLM（QWEN3_API_KEY / QWEN3_BASE_URL / QWEN3_MODEL）
直接运行：python tests/06_demo_fraud.py
pytest 集成测试（不可用时自动跳过）：pytest tests/06_demo_fraud.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import AsyncGenerator, List, Tuple

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
    pass  # python-dotenv 未安装时忽略，依赖系统环境变量

# Parlant 框架（来自 tools/parlant_server）
try:
    from tools.parlant_server import ParlantServerContext  # noqa: E402
    _PARLANT_AVAILABLE = True
except ImportError:
    _PARLANT_AVAILABLE = False

# LLM 客户端（可选，仅供 LLM 可用性检测使用）
try:
    from tools.llm import LLMClient, LLMConfig
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

import pytest

_FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# ---------------------------------------------------------------------------
# 加载 JourneyConfig（从 YAML 文件，等价方式还有 JourneyDSL 和直接构造数据类）
# ---------------------------------------------------------------------------
from journey import load_journey

fraud_config = load_journey(os.path.join(_FIXTURES_DIR, "fraud.yaml"))


class ParlantConversation:
    """
    基于 Parlant Server（无 UI）的对话运行器。

    使用 ParlantServerContext（tools/parlant_server）启动 Server，
    创建 Agent、注册 Journey，并通过 SessionModule 驱动多轮对话。
    """

    #: 等待 Agent 回复的超时秒数
    AGENT_REPLY_TIMEOUT: float = 120.0

    def __init__(
        self,
        config: "JourneyConfig",
        agent_name: str = "银行客服",
        agent_description: str = "专业的银行客服 Agent，负责处理欺诈交易等业务场景",
    ) -> None:
        self._config = config
        self._agent_name = agent_name
        self._agent_description = agent_description
        self._replies: List[str] = []

    @property
    def replies(self) -> List[str]:
        """已收到的全部 Agent 回复列表。"""
        return list(self._replies)

    async def run(self, messages: List[str]) -> AsyncGenerator[Tuple[str, str], None]:
        """逐轮发送消息并 yield (user_msg, agent_reply)。"""
        from journey import create_journey  # 避免顶层循环导入
        # parlant.core.* 在此延迟导入，避免与 conftest.py 中的 parlant.sdk 桩模块冲突
        from parlant.core.app_modules.sessions import SessionModule, Moderation
        from parlant.core.sessions import EventKind, EventSource, Timeout
        from parlant.core.customers import CustomerStore

        async with ParlantServerContext() as ctx:
            # 创建 Agent 并注册 Journey
            agent = await ctx.create_agent(self._agent_name, self._agent_description)
            await create_journey(agent, self._config)

            server = ctx.server
            session_module = server._container[SessionModule]

            # 以匿名访客身份创建会话
            session = await session_module.create(
                customer_id=CustomerStore.GUEST_ID,
                agent_id=agent.id,
                title=f"演示：{self._config.title}",
            )

            event_offset = 0
            for user_msg in messages:
                # 发送用户消息（触发引擎处理）
                evt = await session_module.create_customer_message(
                    session_id=session.id,
                    moderation=Moderation.NONE,
                    message=user_msg,
                    source=EventSource.CUSTOMER,
                    trigger_processing=True,
                    metadata=None,
                )
                next_offset = evt.offset + 1

                # 等待 AI Agent 回复消息事件
                await session_module.wait_for_more_events(
                    session_id=session.id,
                    min_offset=next_offset,
                    kinds=[EventKind.MESSAGE],
                    source=EventSource.AI_AGENT,
                    timeout=Timeout(self.AGENT_REPLY_TIMEOUT),
                )

                # 读取 Agent 回复文本
                agent_events = await session_module.find_events(
                    session_id=session.id,
                    min_offset=next_offset,
                    source=EventSource.AI_AGENT,
                    kinds=[EventKind.MESSAGE],
                    trace_id=None,
                )

                agent_reply = ""
                for ev in agent_events:
                    if isinstance(ev.data, dict) and "message" in ev.data:
                        agent_reply = ev.data["message"]
                        event_offset = ev.offset + 1
                        break
                else:
                    event_offset = next_offset

                self._replies.append(agent_reply)
                yield user_msg, agent_reply


# ---------------------------------------------------------------------------
# 预设对话脚本（模拟中风险场景完整流程）
# ---------------------------------------------------------------------------

DEMO_SCRIPT: List[str] = [
    "你好，我刚才发现我的银行卡好像被盗刷了，有一笔我完全不认识的交易",   # 触发 Journey
    "我叫李明，身份证后四位是6789，银行卡后四位是4321",                     # 配合身份验证
    "是昨天晚上11点半，有一笔1800元，商户是'某某网络科技有限公司'，我没操作过",  # 提供交易信息
    "好的，那我应该怎么处理？",                                              # 获取风险评估后询问
    "我已经修改了密码，也设置了单日限额，申诉单也提交了",                   # 确认已完成处置
    "谢谢你的帮助，非常专业",                                               # 收尾确认
]


# ---------------------------------------------------------------------------
# 演示主函数
# ---------------------------------------------------------------------------


async def run_demo() -> None:
    """运行 Parlant 对话演示：打印配置概览 → 启动 Server → 多轮对话。"""
    # Journey 配置概览
    print("=" * 65)
    print(f"  Journey 对话演示：{fraud_config.title}")
    print("=" * 65)
    print(f"  触发条件：{' | '.join(fraud_config.triggers)}")
    print(f"  状态数量：{len(fraud_config.states)}  "
          f"（{' → '.join(s.name for s in fraud_config.states)}）")
    print(f"  转移规则：{len(fraud_config.transitions)}")
    print(f"  全局规则：{len(fraud_config.global_guidelines)}")
    print()

    print("─" * 65)
    print("  启动 Parlant Server（无 UI）并注册 Journey …")
    print("─" * 65)

    conv = ParlantConversation(fraud_config)

    print()
    print("─" * 65)
    print("  对话开始（Parlant Server 驱动）")
    print("─" * 65)

    try:
        turn = 0
        async for user_msg, agent_reply in conv.run(DEMO_SCRIPT):
            turn += 1
            print(f"\n[Turn {turn}]")
            print(f"  👤 用户：{user_msg}")
            if agent_reply:
                print(f"  🤖 Agent：{agent_reply}")
            else:
                print("  🤖 Agent：（未收到回复，可能超时）")
    except RuntimeError as exc:
        print(f"\n❌ Parlant Server 启动失败：{exc}")
        print("\n请确认已安装 parlant 及 LLM 配置正确（根目录 .env）。")
        return
    except Exception as exc:
        print(f"\n❌ 对话过程发生意外错误：{exc}")
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
    """
    集成测试：验证 ParlantConversation 能通过 Parlant Server 完整运行欺诈处理对话。

    前提：根目录 .env 中 LLM 服务地址可达；否则自动 skip。
    """
    if _LLM_AVAILABLE:
        try:
            config = LLMConfig.from_env()
            llm = LLMClient(config=config)
            await llm.chat_async([{"role": "user", "content": "你好"}])
        except Exception as e:
            pytest.skip(f"LLM 服务不可达：{e}")

    replies: List[str] = []
    try:
        conv = ParlantConversation(fraud_config)
        async for _user_msg, agent_reply in conv.run(DEMO_SCRIPT[:2]):
            replies.append(agent_reply)
    except RuntimeError as e:
        pytest.skip(f"Parlant Server 启动失败：{e}")

    assert len(replies) > 0, "至少应收到 1 条 Agent 回复"


# ===========================================================================
# 直接运行入口
# ===========================================================================

if __name__ == "__main__":
    if not _PARLANT_AVAILABLE:
        print("❌ 缺少依赖：请先安装 parlant")
        print("   pip install parlant")
        sys.exit(1)

    asyncio.run(run_demo())
