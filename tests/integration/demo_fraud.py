"""
tests/integration/demo_fraud.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

欺诈处理 Journey — 完整对话演示
================================

演示 journey 框架的三层控制（流程 + 行为 + 话术），并通过 LLMClient 驱动一次
完整的模拟对话，无需启动 Parlant Server。

运行前提：
    1. 在 third_party/.env 中配置 LLM 服务（QWEN3_API_KEY、QWEN3_BASE_URL、QWEN3_MODEL）
    2. 安装依赖：pip install -r requirements.txt

运行方式：
    python tests/integration/demo_fraud.py

或以 pytest 跳过模式运行（LLM 服务不可用时自动跳过）：
    pytest tests/integration/demo_fraud.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 路径设置：使项目根目录可被导入
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 加载 third_party/.env（LLM 配置来源）
try:
    from dotenv import load_dotenv
    _THIRD_PARTY = os.path.join(_ROOT, "third_party")
    load_dotenv(os.path.join(_THIRD_PARTY, ".env"))
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 依赖检查：使用 journey.generator 提供的 LLMClient（不依赖 third_party/ 代码）
# ---------------------------------------------------------------------------

try:
    from tools.llm import LLMClient, LLMConfig
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

import pytest

# ---------------------------------------------------------------------------
# Journey 框架导入
# ---------------------------------------------------------------------------

from journey import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)

# ---------------------------------------------------------------------------
# 欺诈处理 Journey 配置（三层控制综合演示）
# ---------------------------------------------------------------------------

fraud_config = JourneyConfig(
    title="处理潜在欺诈交易",
    triggers=[
        "我卡被盗刷了",
        "可疑交易",
        "账户异常",
        "系统发出欺诈预警",
        "怀疑被骗",
    ],
    description=(
        "引导客户完成欺诈交易的核实与处置，覆盖：身份验证、信息采集、"
        "风险评估和针对性处置，确保流程合规、可审计。"
    ),
    states=[
        # ── 1. 身份验证 ──────────────────────────────────────────────────
        StateConfig(
            name="identity_verify",
            description="完成用户身份验证（KYC）",
            chat_state=(
                "严格进行身份验证，要求用户提供姓名、身份证后四位或银行卡后四位。"
                "验证失败时给予明确提示，最多允许 3 次重试。"
            ),
            canned_response=CannedResponseConfig(
                template=(
                    "为了您的账户安全，我需要先验证您的身份。"
                    "请提供您的姓名、身份证后四位或银行卡后四位。"
                ),
                strict=True,
            ),
        ),
        # ── 2. 信息采集 ──────────────────────────────────────────────────
        StateConfig(
            name="collect_info",
            description="收集可疑交易的关键信息",
            chat_state=(
                "引导用户详细描述可疑交易：发生时间、金额、商户名称、是否本人操作。"
                "记录所有细节，为后续风险评估做准备。"
            ),
        ),
        # ── 3. 风险评估（模拟工具调用） ──────────────────────────────────
        StateConfig(
            name="risk_assess",
            description="评估交易风险等级",
            chat_state=(
                "根据已收集的交易信息评估风险等级。"
                "在实际系统中此状态会调用风险评估工具；"
                "演示模式下请告知用户正在评估，并给出风险评级（HIGH / MEDIUM / LOW）。"
            ),
        ),
        # ── 4a. 高风险处置 ────────────────────────────────────────────────
        StateConfig(
            name="high_risk",
            description="执行高风险处置措施",
            chat_state=(
                "告知客户当前风险极高，立即引导：1) 冻结账户；"
                "2) 申请挂失银行卡；3) 拨打紧急客服热线。"
            ),
            guidelines=[
                GuidelineConfig(
                    condition="处于 high_risk 状态",
                    action="必须强调立即操作的紧迫性，并提供具体操作步骤",
                    priority=95,
                ),
                GuidelineConfig(
                    condition="用户犹豫不冻结账户",
                    action="再次强调风险，说明延迟操作的潜在损失",
                    priority=85,
                ),
            ],
        ),
        # ── 4b. 中风险处置 ────────────────────────────────────────────────
        StateConfig(
            name="medium_risk",
            description="执行中风险处置措施",
            chat_state=(
                "建议用户：1) 修改账户密码；2) 设置单日交易限额；"
                "3) 提交异议申诉，银行将在 3 个工作日内处理。"
            ),
        ),
        # ── 4c. 低风险处置 ────────────────────────────────────────────────
        StateConfig(
            name="low_risk",
            description="执行低风险处置措施",
            chat_state=(
                "告知用户当前交易风险较低，未发现明确欺诈迹象。"
                "建议保持警惕，避免点击陌生链接或向陌生人透露验证码。"
            ),
        ),
        # ── 5. 流程收尾 ──────────────────────────────────────────────────
        StateConfig(
            name="closing",
            description="完成流程收尾与客户确认",
            chat_state=(
                "总结本次处置结果，确认用户已完成所需操作，"
                "感谢客户的配合，并告知后续跟进方式。邀请客户对本次服务进行评价。"
            ),
        ),
    ],
    transitions=[
        TransitionConfig(
            source="identity_verify",
            target="collect_info",
            condition="身份验证成功",
        ),
        TransitionConfig(
            source="identity_verify",
            target="closing",
            condition="验证失败达到 3 次",
        ),
        TransitionConfig(source="collect_info", target="risk_assess"),
        TransitionConfig(
            source="risk_assess",
            target="high_risk",
            condition="风险等级为 HIGH",
        ),
        TransitionConfig(
            source="risk_assess",
            target="medium_risk",
            condition="风险等级为 MEDIUM",
        ),
        TransitionConfig(
            source="risk_assess",
            target="low_risk",
            condition="风险等级为 LOW",
        ),
        TransitionConfig(source="high_risk", target="closing"),
        TransitionConfig(source="medium_risk", target="closing"),
        TransitionConfig(source="low_risk", target="closing"),
    ],
    global_guidelines=[
        GuidelineConfig(
            condition="用户情绪激动或表达强烈不满",
            action="优先安抚情绪，表示理解和歉意，再继续流程",
            priority=90,
        ),
        GuidelineConfig(
            condition="用户多次询问同一问题",
            action="耐心重复解释，保持专业和耐心",
            priority=60,
        ),
        GuidelineConfig(
            condition="用户提出流程之外的问题",
            action="简短回答后，引导用户回到当前流程步骤",
            priority=55,
        ),
    ],
)

# ---------------------------------------------------------------------------
# 预设对话脚本（模拟中风险场景全流程）
# ---------------------------------------------------------------------------

DEMO_SCRIPT: List[str] = [
    # 触发 Journey
    "你好，我刚才发现我的银行卡好像被盗刷了，有一笔我完全不认识的交易",
    # 配合身份验证
    "我叫李明，身份证后四位是6789，银行卡后四位是4321",
    # 提供交易信息
    "是昨天晚上11点半左右，有一笔1800元的消费，商户是'某某网络科技有限公司'，我确实没有操作过",
    # 获取风险评估结果后继续
    "好的，那我应该怎么处理？",
    # 确认已完成处置操作
    "我已经修改了密码，也设置了单日限额，申诉单也提交了",
    # 收尾确认
    "谢谢你的帮助，非常专业",
]

# ---------------------------------------------------------------------------
# JourneySimulator — 无需 Parlant Server 的 Journey 模拟器
# ---------------------------------------------------------------------------


class JourneySimulator:
    """
    基于 :class:`~journey.models.JourneyConfig` 和 LLMClient 的轻量对话模拟器。

    工作方式：
      1. 解析 JourneyConfig，构建状态 DAG。
      2. 从根状态（无入边的状态）出发，逐状态推进对话。
      3. 每个 chat_state：构造 system prompt（含 chat_state 指令 + guidelines），
         调用 LLMClient 生成 Agent 回复。
      4. 每轮对话结束后，用 LLM 判断各转移条件是否满足，自动跳转到下一状态。
      5. 到达终止状态（无出边）后标记对话结束。
    """

    # 历史截取长度：只取最近 N 轮用于条件判断，在上下文完整性与 token 消耗之间取得平衡
    MAX_HISTORY_FOR_CONDITION: int = 6

    def __init__(self, config: JourneyConfig, llm: "LLMClient") -> None:
        self._config = config
        self._llm = llm
        self._state_cfg: Dict[str, StateConfig] = {s.name: s for s in config.states}
        # source → [(condition, target), ...]
        self._edges: Dict[str, List[Tuple[Optional[str], str]]] = {}
        for t in config.transitions:
            self._edges.setdefault(t.source, []).append((t.condition, t.target))
        # 终止状态：无出边
        self._terminals = {
            s.name for s in config.states
            if s.name not in self._edges
        }
        # 根状态：无入边
        all_targets = {t.target for t in config.transitions}
        roots = [s.name for s in config.states if s.name not in all_targets]
        self._current = roots[0] if roots else config.states[0].name
        self._history: List[dict] = []
        # 当前状态是否已发送过首次回复（用于 canned_response 控制）
        self._state_first_reply: Dict[str, bool] = {}
        self.done = False

    @property
    def current_state(self) -> str:
        return self._current

    async def reply(self, user_message: str) -> str:
        """处理一条用户消息，返回 Agent 回复文本，并自动推进状态机。"""
        self._history.append({"role": "user", "content": user_message})

        state_cfg = self._state_cfg[self._current]
        agent_reply = await self._generate_reply(state_cfg)

        self._history.append({"role": "assistant", "content": agent_reply})
        self._state_first_reply[self._current] = True

        # 尝试推进状态
        await self._try_advance()

        return agent_reply

    # ------------------------------------------------------------------
    # 内部：生成 Agent 回复
    # ------------------------------------------------------------------

    async def _generate_reply(self, state_cfg: StateConfig) -> str:
        """根据当前状态生成 Agent 回复。"""
        # ① canned_response（strict 模式）：首次进入该状态时直接使用模板
        if (
            state_cfg.canned_response
            and state_cfg.canned_response.strict
            and not self._state_first_reply.get(state_cfg.name)
        ):
            return state_cfg.canned_response.template

        # ② 构造 system prompt
        system_prompt = self._build_system_prompt(state_cfg)
        messages = [{"role": "system", "content": system_prompt}] + self._history

        return await self._llm.chat_async(messages)

    def _build_system_prompt(self, state_cfg: StateConfig) -> str:
        """将当前状态的 chat_state 指令 + guidelines 合并为 system prompt。"""
        lines = [
            "你是一个专业、合规的银行客服 Agent。",
            f"\n【当前流程节点】{state_cfg.description or state_cfg.name}",
        ]
        if state_cfg.chat_state:
            lines.append(f"\n【行为指令】{state_cfg.chat_state}")

        # 节点级 guidelines
        node_rules = state_cfg.guidelines
        # 全局 guidelines（按优先级降序）
        global_rules = sorted(
            self._config.global_guidelines, key=lambda g: g.priority, reverse=True
        )
        all_rules = sorted(
            node_rules + global_rules, key=lambda g: g.priority, reverse=True
        )
        if all_rules:
            lines.append("\n【行为规则】")
            for g in all_rules:
                lines.append(f"  - 若「{g.condition}」，则「{g.action}」（优先级 {g.priority}）")

        # canned_response（fluid 模式）：作为参考话术
        if (
            state_cfg.canned_response
            and not state_cfg.canned_response.strict
        ):
            lines.append(f"\n【参考话术】{state_cfg.canned_response.template}")

        lines.append("\n请用简洁、专业的中文回复用户，严格遵守上述指令和规则。")
        return "".join(lines)

    # ------------------------------------------------------------------
    # 内部：状态转移
    # ------------------------------------------------------------------

    async def _try_advance(self) -> None:
        """根据当前对话内容，判断是否满足转移条件并推进状态机。"""
        if self._current in self._terminals:
            self.done = True
            return

        edges = self._edges.get(self._current, [])
        if not edges:
            self.done = True
            return

        for condition, target in edges:
            if condition is None:
                # 无条件转移
                self._current = target
                if self._current in self._terminals:
                    self.done = True
                return
            if await self._condition_met(condition):
                self._current = target
                if self._current in self._terminals:
                    self.done = True
                return

    async def _condition_met(self, condition: str) -> bool:
        """
        使用 LLM 判断转移条件是否已在当前对话中满足。

        只取最近 6 轮历史，减少 token 消耗。
        """
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
# 演示入口
# ---------------------------------------------------------------------------


async def run_demo() -> None:
    """运行完整欺诈处理 Journey 对话演示。"""
    print("=" * 65)
    print(f"  Journey 对话演示：{fraud_config.title}")
    print("=" * 65)
    print(f"  触发条件：{' | '.join(fraud_config.triggers)}")
    print(f"  状态数量：{len(fraud_config.states)}")
    print(f"  转移规则：{len(fraud_config.transitions)}")
    print(f"  全局规则：{len(fraud_config.global_guidelines)}")
    print()

    # 初始化 LLM 客户端（从 third_party/.env 读取配置）
    config = LLMConfig.from_env()
    print(f"  LLM 模型：{config.model}")
    print(f"  API 地址：{config.base_url}")
    print()

    llm = LLMClient(config=config)
    sim = JourneySimulator(fraud_config, llm)

    print("─" * 65)
    print("  对话开始")
    print("─" * 65)

    for i, user_msg in enumerate(DEMO_SCRIPT, 1):
        print(f"\n[Turn {i}] 状态：{sim.current_state}")
        print(f"  👤 用户：{user_msg}")

        agent_reply = await sim.reply(user_msg)
        print(f"  🤖 Agent：{agent_reply}")

        if sim.done:
            print(f"\n  ✅ Journey 已到达终止状态：{sim.current_state}")
            break

    if not sim.done:
        print(f"\n  ℹ️  演示脚本结束，当前状态：{sim.current_state}")

    print("\n" + "=" * 65)
    print("  演示完成")
    print("=" * 65)


# ---------------------------------------------------------------------------
# pytest 测试入口（自动跳过无法连接的情况）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _LLM_AVAILABLE, reason="model_client 未安装（pip install litellm openai）")
async def test_fraud_demo_runs_full_conversation() -> None:
    """
    集成测试：验证 JourneySimulator 能走完完整的欺诈处理 Journey 对话。

    需要可用的 LLM 服务（third_party/.env 中配置），否则自动跳过。
    """
    try:
        config = LLMConfig.from_env()
    except (ValueError, Exception) as e:
        pytest.skip(f"LLM 配置不可用，跳过集成测试：{e}")

    try:
        llm = LLMClient(config=config)
        # 快速连通性检查
        await llm.chat_async([{"role": "user", "content": "你好"}])
    except Exception as e:
        pytest.skip(f"LLM 服务不可达，跳过集成测试：{e}")

    sim = JourneySimulator(fraud_config, llm)

    for user_msg in DEMO_SCRIPT:
        reply = await sim.reply(user_msg)
        assert isinstance(reply, str) and len(reply) > 0, "Agent 回复不应为空"
        if sim.done:
            break

    # 验证走完了至少 3 个状态节点
    assert sim._state_first_reply, "至少应有 1 个状态产生过回复"


# ---------------------------------------------------------------------------
# 直接运行
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if not _LLM_AVAILABLE:
        print("❌ 缺少依赖：请先安装 litellm 和 openai")
        print("   pip install litellm openai python-dotenv")
        sys.exit(1)

    asyncio.run(run_demo())
