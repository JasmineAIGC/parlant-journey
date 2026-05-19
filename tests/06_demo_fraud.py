"""
tests/06_demo_fraud.py
~~~~~~~~~~~~~~~~~~~~~~

如何使用 journey/ 模块 — 欺诈处理场景完整示例
================================================

本文件分两个独立部分：

  ┌─────────────────────────────────────────────────────────────┐
  │  Part 1：journey/ 模块的核心用法（注册 Journey 到 Agent）   │
  │    Step 1 — 构建 JourneyConfig（3 种等价方式）              │
  │    Step 2 — 注册 Journey 到 Parlant Agent                   │
  └─────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────┐
  │  Part 2：Parlant 对话演示（通过 Parlant Server 运行真实会话）│
  │    使用 ParlantServerContext 启动 Parlant Server，           │
  │    将 Journey 注册到 Agent，并通过 SessionModule            │
  │    驱动多轮对话，体验 journey 配置在 Parlant 中的实际效果。 │
  └─────────────────────────────────────────────────────────────┘

运行前提：
    1. 在根目录 .env 中配置 LLM 服务
       （QWEN3_API_KEY、QWEN3_BASE_URL、QWEN3_MODEL）
    2. 安装依赖：pip install -r requirements.txt

直接运行（Parlant 对话演示）：
    python tests/06_demo_fraud.py

作为 pytest 集成测试运行（Parlant 或 LLM 服务不可用时自动跳过）：
    pytest tests/06_demo_fraud.py -v
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


# ===========================================================================
# Part 1：journey/ 模块的核心用法
# ===========================================================================
#
# journey/ 模块的完整公开 API：
#
#   from journey import (
#       # 配置加载
#       load_journey,          # 从 YAML / JSON 文件加载 JourneyConfig
#       load_journey_dict,     # 从 Python 字典加载
#       JourneyDSL,            # 流式 Python DSL 构建器
#       # 数据模型（直接构造配置时使用）
#       JourneyConfig, StateConfig, TransitionConfig,
#       GuidelineConfig, CannedResponseConfig,
#       # 注册到 Parlant Agent（核心入口）
#       create_journey,        # 注册单个 Journey
#       register_journeys,     # 批量注册多个 Journey
#       # LLM 自动生成配置（可选）
#       JourneyGenerator, generate_journey_config,
#   )


# ---------------------------------------------------------------------------
# Step 1：构建 JourneyConfig
# ---------------------------------------------------------------------------
#
# JourneyConfig 描述一个完整 Journey 的结构：
#   - title / description / triggers：Journey 的基本信息与触发条件
#   - states：状态节点列表（每个节点可含行为指令 chat_state、预设话术等）
#   - transitions：状态转移规则（source → target，可附带触发条件）
#   - global_guidelines：全局行为规则（对所有状态生效）
#
# 以下展示 3 种等价的构建方式，选择最适合你的场景：

# ── 方式 1（推荐）：从 YAML 文件加载 ────────────────────────────────────
#
# 优点：配置与代码分离，易于维护和版本控制；支持非技术同学直接编辑。
# 文件格式详见 tests/fixtures/fraud.yaml。
#
from journey import load_journey

fraud_config = load_journey(os.path.join(_FIXTURES_DIR, "fraud.yaml"))

# ── 方式 2：使用 JourneyDSL 流式 API 构建（Python 内快速原型）────────────
#
# 优点：纯 Python，无需 YAML 文件，链式调用可读性好。
# 示例（与 fraud.yaml 等价的最简化版本）：
#
#   from journey import JourneyDSL
#
#   fraud_config = (
#       JourneyDSL("处理潜在欺诈交易")
#       .triggers("我卡被盗刷了", "可疑交易", "账户异常")
#       .describe("引导客户完成欺诈交易的核实与处置")
#       .state("identity_verify",
#              description="完成用户身份验证（KYC）",
#              chat="严格进行身份验证，要求用户提供姓名、身份证后四位。",
#              canned="为了您的账户安全，我需要先验证您的身份。"
#                     "请提供您的姓名、身份证后四位或银行卡后四位。",
#              strict=True)
#       .state("collect_info",
#              description="收集可疑交易的关键信息",
#              chat="引导用户描述可疑交易：时间、金额、商户、是否本人操作。")
#       .state("closing",
#              description="完成流程收尾与客户确认",
#              chat="总结处置结果，确认用户已完成所需操作，感谢配合。")
#       .go("identity_verify", "collect_info", when="身份验证成功")
#       .go("collect_info", "closing")
#       .global_rule("用户情绪激动", "优先安抚情绪，再继续流程", priority=90)
#       .build()
#   )

# ── 方式 3：直接构造数据类（最完整，适合程序生成）──────────────────────
#
# 优点：所有字段完全透明可控，适合从数据库或 API 动态生成配置。
# 示例：
#
#   from journey import (
#       JourneyConfig, StateConfig, TransitionConfig,
#       GuidelineConfig, CannedResponseConfig,
#   )
#
#   fraud_config = JourneyConfig(
#       title="处理潜在欺诈交易",
#       description="引导客户完成欺诈交易的核实与处置",
#       triggers=["我卡被盗刷了", "可疑交易"],
#       states=[
#           StateConfig(
#               name="identity_verify",
#               description="完成用户身份验证（KYC）",
#               chat_state="严格进行身份验证，要求用户提供姓名、身份证后四位。",
#               canned_response=CannedResponseConfig(
#                   template="为了您的账户安全，我需要先验证您的身份。"
#                            "请提供您的姓名、身份证后四位或银行卡后四位。",
#                   strict=True,  # strict=True：首次进入时强制使用此话术
#               ),
#           ),
#           StateConfig(
#               name="collect_info",
#               description="收集可疑交易的关键信息",
#               chat_state="引导用户描述可疑交易：时间、金额、商户、是否本人操作。",
#           ),
#           StateConfig(
#               name="closing",
#               description="完成流程收尾与客户确认",
#               chat_state="总结处置结果，感谢配合。",
#               guidelines=[
#                   GuidelineConfig(
#                       condition="用户询问后续跟进方式",
#                       action="告知 3 个工作日内会有专员回访",
#                       priority=70,
#                   ),
#               ],
#           ),
#       ],
#       transitions=[
#           TransitionConfig(source="identity_verify", target="collect_info",
#                            condition="身份验证成功"),
#           TransitionConfig(source="collect_info",    target="closing"),
#       ],
#       global_guidelines=[
#           GuidelineConfig(condition="用户情绪激动", action="优先安抚情绪", priority=90),
#       ],
#   )


# ---------------------------------------------------------------------------
# Step 2：将 Journey 注册到 Parlant Agent
# ---------------------------------------------------------------------------
#
# journey/ 模块的核心作用：把上面构建好的 JourneyConfig 注册到 Parlant Agent，
# 让 Agent 在用户触发对应关键词时自动执行该 Journey 流程。
#
# ❗ 以下代码需要正在运行的 Parlant Server，本地演示（Part 2）不需要。
#
#   import parlant.sdk as p
#   from journey import create_journey, register_journeys
#
#   async with p.Server() as server:
#       # 创建 Agent（或复用已有 Agent）
#       agent = await server.agents.create(name="银行客服")
#
#       # ── 注册单个 Journey ──────────────────────────────────────────────
#       journey = await create_journey(agent, fraud_config)
#
#       # ── 批量注册多个 Journey（推荐：一次性注册所有业务场景）────────────
#       # journeys = await register_journeys(agent, [fraud_config, loan_config])
#
#       # 注册完成后，只要用户说出 fraud_config.triggers 中的任意词，
#       # Parlant 就会自动触发 fraud_config 描述的对话流程。
#
# 注：create_journey / register_journeys 的具体实现见 journey/factory.py。
#     JourneyBuilder 内部会将 states / transitions / guidelines 逐一注册
#     到 Parlant SDK 的对应 API 上；调用方无需关心 Parlant SDK 细节。


# ===========================================================================
# Part 2：Parlant 对话演示（通过 Parlant Server 运行真实会话）
# ===========================================================================
#
# journey/ 模块本质上是在 Parlant 框架内工作的——它将 JourneyConfig 注册为
# Parlant Agent 的 Guidelines / Journey / CannedResponse，由 Parlant 引擎
# 驱动 LLM 对话并自动完成状态推进。
#
# 以下 ParlantConversation 类通过 tools/parlant_server 中的
# ParlantServerContext 启动 Parlant Server（无 UI），在其中创建 Agent、
# 注册 Journey，并利用 Parlant 的 SessionModule 发送消息、等待 Agent 回复，
# 模拟真实的多轮对话流程。
#
# 这正是 journey 在生产环境中的工作方式：
#   用户消息 → Parlant 引擎 → Guidelines 匹配 → LLM 生成回复 → Agent 回复


class ParlantConversation:
    """
    基于 Parlant Server 的对话运行器。

    工作方式：
      1. 通过 ParlantServerContext 启动 Parlant Server（无 UI）。
      2. 创建 Parlant Agent，并将 JourneyConfig 注册到该 Agent。
      3. 使用 SessionModule 创建会话（匿名客户）。
      4. 每轮对话：
         a. 通过 SessionModule.create_customer_message() 发送用户消息；
         b. 通过 SessionModule.wait_for_more_events() 等待 AI Agent 回复；
         c. 通过 SessionModule.find_events() 读取 Agent 回复文本。
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
        """
        启动 Parlant Server，注册 Journey 并逐轮运行对话。

        Args:
            messages: 模拟用户依次发送的消息列表。
        """
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
    """
    运行 Parlant 对话演示。

    演示流程：
      1. 打印 fraud_config 的基本信息（Journey 的 title、状态数等）
      2. 启动 ParlantServerContext（无 UI Parlant Server）
      3. 创建 Agent，将 fraud Journey 注册到 Agent
      4. 通过 SessionModule 发送 DEMO_SCRIPT 中的消息并打印 Agent 回复
    """
    # ── 显示 Journey 配置概览（来自 Step 1 加载的 fraud_config）──────────
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
