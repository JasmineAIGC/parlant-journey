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
  │  Part 2：本地 LLM 对话模拟（无需 Parlant Server）           │
  │    仅用于演示 journey 配置的对话效果，与 journey/ 无关。    │
  └─────────────────────────────────────────────────────────────┘

运行前提：
    1. 在 third_party/.env 中配置 LLM 服务
       （QWEN3_API_KEY、QWEN3_BASE_URL、QWEN3_MODEL）
    2. 安装依赖：pip install -r requirements.txt

直接运行（本地 LLM 对话演示）：
    python tests/06_demo_fraud.py

作为 pytest 集成测试运行（LLM 服务不可用时自动跳过）：
    pytest tests/06_demo_fraud.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 路径设置：使项目根目录可被导入（支持从任意目录运行本文件）
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 加载 third_party/.env（LLM 服务地址 / API Key / 模型名）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, "third_party", ".env"))
except ImportError:
    pass  # python-dotenv 未安装时忽略，依赖系统环境变量

# LLM 客户端（来自 tools.llm，与 journey/ 模块无直接耦合）
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
# Part 2：本地 LLM 对话模拟（演示用，与 journey/ 模块无关）
# ===========================================================================
#
# 以下 JourneySimulator 类是一个独立的本地模拟器，它直接读取 JourneyConfig
# 的状态机结构，通过 LLM 驱动对话并判断状态转移条件。
#
# ⚠️  注意：这 **不是** journey/ 模块的一部分，也不是 Parlant 的工作方式。
#     它的存在只是为了让你在没有 Parlant Server 的情况下，
#     也能通过本地 LLM 直观感受 journey 配置定义的对话流程效果。
#
# 真实部署时，Parlant Server 本身会处理状态推进、LLM 对话生成，
# 你只需完成 Part 1 的 Step 1 + Step 2，无需任何模拟器代码。


class JourneySimulator:
    """
    【本地演示专用】基于 JourneyConfig + LLMClient 的轻量对话模拟器。

    与 journey/ 模块及 Parlant Server 无耦合，仅用于演示 journey 配置效果。

    工作方式：
      1. 解析 JourneyConfig，构建状态 DAG（有向无环图）。
      2. 从根状态（无入边的状态）出发，逐状态推进对话。
      3. 每个状态：将 chat_state 指令 + guidelines 合并为 system prompt，
         调用 LLMClient 生成 Agent 回复。
      4. 每轮对话结束后，用 LLM 判断各转移条件是否满足，自动跳转到下一状态。
      5. 到达终止状态（无出边）后标记对话结束。
    """

    # 只取最近 N 轮历史用于条件判断，平衡上下文完整性与 token 消耗
    MAX_HISTORY_FOR_CONDITION: int = 6

    def __init__(self, config: "JourneyConfig", llm: "LLMClient") -> None:
        from journey.models import StateConfig  # 避免顶层循环导入

        self._config = config
        self._llm = llm
        self._state_cfg: Dict[str, StateConfig] = {s.name: s for s in config.states}

        # 构建转移图：source → [(condition, target), ...]
        self._edges: Dict[str, List[Tuple[Optional[str], str]]] = {}
        for t in config.transitions:
            self._edges.setdefault(t.source, []).append((t.condition, t.target))

        # 终止状态（无出边）
        self._terminals = {s.name for s in config.states if s.name not in self._edges}

        # 根状态（无入边），作为起始状态
        all_targets = {t.target for t in config.transitions}
        roots = [s.name for s in config.states if s.name not in all_targets]
        self._current = roots[0] if roots else config.states[0].name

        self._history: List[dict] = []
        self._state_first_reply: Dict[str, bool] = {}  # 控制 canned_response
        self.done = False

    @property
    def current_state(self) -> str:
        return self._current

    async def reply(self, user_message: str) -> str:
        """处理一条用户消息，返回 Agent 回复，并自动推进状态机。"""
        self._history.append({"role": "user", "content": user_message})

        state_cfg = self._state_cfg[self._current]
        agent_reply = await self._generate_reply(state_cfg)

        self._history.append({"role": "assistant", "content": agent_reply})
        self._state_first_reply[self._current] = True
        await self._try_advance()
        return agent_reply

    # ── 生成回复 ──────────────────────────────────────────────────────────

    async def _generate_reply(self, state_cfg) -> str:
        # canned_response strict 模式：首次进入该状态时强制使用预设话术
        if (
            state_cfg.canned_response
            and state_cfg.canned_response.strict
            and not self._state_first_reply.get(state_cfg.name)
        ):
            return state_cfg.canned_response.template

        system_prompt = self._build_system_prompt(state_cfg)
        messages = [{"role": "system", "content": system_prompt}] + self._history
        return await self._llm.chat_async(messages)

    def _build_system_prompt(self, state_cfg) -> str:
        """将当前状态的 chat_state 指令 + guidelines 组合成 system prompt。"""
        lines = [
            "你是一个专业、合规的银行客服 Agent。",
            f"\n【当前流程节点】{state_cfg.description or state_cfg.name}",
        ]
        if state_cfg.chat_state:
            lines.append(f"\n【行为指令】{state_cfg.chat_state}")

        # 节点级规则 + 全局规则，按优先级降序合并
        all_rules = sorted(
            state_cfg.guidelines
            + sorted(self._config.global_guidelines, key=lambda g: g.priority, reverse=True),
            key=lambda g: g.priority,
            reverse=True,
        )
        if all_rules:
            lines.append("\n【行为规则】")
            for g in all_rules:
                lines.append(f"  - 若「{g.condition}」，则「{g.action}」（优先级 {g.priority}）")

        # canned_response fluid 模式：作为参考话术（不强制使用）
        if state_cfg.canned_response and not state_cfg.canned_response.strict:
            lines.append(f"\n【参考话术】{state_cfg.canned_response.template}")

        lines.append("\n请用简洁、专业的中文回复用户，严格遵守上述指令和规则。")
        return "".join(lines)

    # ── 状态转移 ──────────────────────────────────────────────────────────

    async def _try_advance(self) -> None:
        """根据对话内容判断转移条件，推进状态机。"""
        if self._current in self._terminals:
            self.done = True
            return

        for condition, target in self._edges.get(self._current, []):
            if condition is None or await self._condition_met(condition):
                self._current = target
                if self._current in self._terminals:
                    self.done = True
                return

    async def _condition_met(self, condition: str) -> bool:
        """用 LLM 判断转移条件是否已在近期对话中满足（只取最近 6 轮）。"""
        recent = self._history[-self.MAX_HISTORY_FOR_CONDITION:]
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else 'Agent'}：{m['content']}"
            for m in recent
        )
        prompt = (
            f"根据以下对话历史，判断条件「{condition}」是否已满足。\n\n"
            f"对话历史：\n{history_text}\n\n"
            "请只回答 YES 或 NO，不要解释。"
        )
        result = await self._llm.chat_async(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return "YES" in result.upper()


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
    运行本地 LLM 对话演示。

    演示流程：
      1. 打印 fraud_config 的基本信息（Journey 的 title、状态数等）
      2. 初始化 LLMClient（从 third_party/.env 读取配置）
      3. 创建 JourneySimulator，逐轮模拟对话
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

    # ── 初始化 LLM 客户端（非 journey/ 模块，仅供本地演示使用）──────────
    llm_config = LLMConfig.from_env()
    print(f"  LLM 模型：{llm_config.model}")
    print(f"  API 地址：{llm_config.base_url}")
    print()

    llm = LLMClient(config=llm_config)

    # ── 创建模拟器并开始对话（非 Parlant，仅本地演示）───────────────────
    sim = JourneySimulator(fraud_config, llm)

    print("─" * 65)
    print("  对话开始（本地 LLM 模拟，非 Parlant Server）")
    print("─" * 65)

    for i, user_msg in enumerate(DEMO_SCRIPT, 1):
        print(f"\n[Turn {i}] 状态：{sim.current_state}")
        print(f"  👤 用户：{user_msg}")

        try:
            agent_reply = await sim.reply(user_msg)
        except RuntimeError as exc:
            print(f"\n❌ LLM 服务不可达，演示中止。\n{exc}")
            print("\n请确认 third_party/.env 中的 LLM 地址和端口正确，且服务正在运行。")
            return
        except Exception as exc:
            print(f"\n❌ 调用 LLM 时发生意外错误：{exc}")
            return

        print(f"  🤖 Agent：{agent_reply}")

        if sim.done:
            print(f"\n  ✅ Journey 已到达终止状态：{sim.current_state}")
            break

    if not sim.done:
        print(f"\n  ℹ️  演示脚本结束，当前状态：{sim.current_state}")

    print("\n" + "=" * 65)
    print("  演示完成")
    print("=" * 65)


# ===========================================================================
# pytest 集成测试入口（LLM 服务不可用时自动跳过）
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(not _LLM_AVAILABLE, reason="litellm 未安装（pip install litellm openai）")
async def test_fraud_demo_runs_full_conversation() -> None:
    """
    集成测试：验证 JourneySimulator 能完整走完欺诈处理 Journey 对话。

    前提：third_party/.env 中 LLM 服务地址可达；否则自动 skip。
    """
    try:
        config = LLMConfig.from_env()
    except (ValueError, Exception) as e:
        pytest.skip(f"LLM 配置不可用：{e}")

    try:
        llm = LLMClient(config=config)
        await llm.chat_async([{"role": "user", "content": "你好"}])
    except Exception as e:
        pytest.skip(f"LLM 服务不可达：{e}")

    sim = JourneySimulator(fraud_config, llm)
    for user_msg in DEMO_SCRIPT:
        reply = await sim.reply(user_msg)
        assert isinstance(reply, str) and len(reply) > 0, "Agent 回复不应为空"
        if sim.done:
            break

    assert sim._state_first_reply, "至少应有 1 个状态产生过回复"


# ===========================================================================
# 直接运行入口
# ===========================================================================

if __name__ == "__main__":
    if not _LLM_AVAILABLE:
        print("❌ 缺少依赖：请先安装 litellm 和 openai")
        print("   pip install litellm openai python-dotenv")
        sys.exit(1)

    asyncio.run(run_demo())
