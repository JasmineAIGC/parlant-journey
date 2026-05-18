"""
examples/fraud.py
~~~~~~~~~~~~~~~~~

欺诈处理 Journey 示例 — 演示 parlant-journey 的完整用法。

场景：银行客服 Agent 处理用户上报的可疑交易，依次完成：
  身份验证 → 信息采集 → 风险评估（工具调用）→ 针对性处置 → 流程收尾

三层控制同时启用：
  流程（路径）  : states + transitions
  行为（规则）  : guidelines（节点级 + 全局）
  话术（合规）  : canned_response（identity_verify 节点）

运行：
    python examples/fraud.py

注意：运行前需启动 Parlant Server 并按实际配置修改 host / port / nlp_service。
"""

from __future__ import annotations

import asyncio

import parlant.sdk as p

from parlant_journey import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
    create_journey,
)

# ---------------------------------------------------------------------------
# Journey 配置声明
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
    # ------------------------------------------------------------------
    # 状态定义（顺序无强制要求，但须覆盖 transitions 中的所有目标状态）
    # ------------------------------------------------------------------
    states=[
        StateConfig(
            name="identity_verify",
            description="完成用户身份验证（KYC）",
            chat_state=(
                "严格进行身份验证，要求用户提供姓名、身份证后四位或银行卡后四位。"
                "验证失败时给予明确提示，最多允许 3 次重试。"
            ),
            # 关键合规节点：锁定开场话术，零幻觉
            canned_response=CannedResponseConfig(
                template=(
                    "为了您的账户安全，我需要先验证您的身份。"
                    "请提供您的姓名、身份证后四位或银行卡后四位。"
                ),
                strict=True,
            ),
        ),
        StateConfig(
            name="collect_info",
            description="收集可疑交易的关键信息",
            chat_state=(
                "引导用户详细描述可疑交易：发生时间、金额、商户名称、是否本人操作。"
                "记录所有细节，为后续风险评估做准备。"
            ),
        ),
        StateConfig(
            name="risk_assess",
            description="评估交易风险等级",
            # 工具状态：调用风险评估工具，返回 HIGH / MEDIUM / LOW
            tool_state=None,  # 替换为实际工具函数，例如 assess_fraud_risk
            tool_instruction=(
                "根据收集到的交易信息（时间、金额、商户、用户行为），"
                "调用风险评估工具，返回风险等级：HIGH / MEDIUM / LOW。"
            ),
        ),
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
        StateConfig(
            name="medium_risk",
            description="执行中风险处置措施",
            chat_state=(
                "建议用户：1) 修改账户密码；2) 设置单日交易限额；"
                "3) 提交异议申诉，银行将在 3 个工作日内处理。"
            ),
        ),
        StateConfig(
            name="low_risk",
            description="执行低风险处置措施",
            chat_state=(
                "告知用户当前交易风险较低，未发现明确欺诈迹象。"
                "建议保持警惕，避免点击陌生链接或向陌生人透露验证码。"
            ),
        ),
        StateConfig(
            name="closing",
            description="完成流程收尾与客户确认",
            chat_state=(
                "总结本次处置结果，确认用户已完成所需操作，"
                "感谢客户的配合，并告知后续跟进方式。邀请客户对本次服务进行评价。"
            ),
        ),
    ],
    # ------------------------------------------------------------------
    # 状态迁移（拓扑顺序：父节点的边须先于子节点的边出现）
    # ------------------------------------------------------------------
    transitions=[
        # identity_verify 是根节点（无入边），由 journey.initial_state 直接创建
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
        # 三路汇聚到 closing（closing 已由第二条 identity_verify 边创建，后续复用）
        TransitionConfig(source="high_risk", target="closing"),
        TransitionConfig(source="medium_risk", target="closing"),
        TransitionConfig(source="low_risk", target="closing"),
    ],
    # ------------------------------------------------------------------
    # 全局 Guidelines（整个 Journey 生效）
    # ------------------------------------------------------------------
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
# 入口
# ---------------------------------------------------------------------------


async def main() -> None:
    async with p.Server(
        host="localhost",
        port=8800,
        nlp_service="<your-nlp-service-url>",  # 替换为实际 NLP 服务地址
    ) as server:
        agent = await server.create_agent(
            name="FraudHandlingAgent",
            description="专注于欺诈交易处理的银行客服 Agent，确保流程合规、可审计。",
        )

        journey = await create_journey(agent, fraud_config)
        print(f"✅ Journey '{fraud_config.title}' 创建成功")
        return journey


if __name__ == "__main__":
    asyncio.run(main())
