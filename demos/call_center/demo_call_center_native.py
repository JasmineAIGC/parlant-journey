"""
demos/call_center/demo_call_center_native.py  —  银行智能呼叫中心（Parlant 原生 SDK）

直接使用 parlant.sdk 的原生 API 构建 7 个呼叫中心决策 Journey：
  • parlant.sdk.Server       —— 启动带 UI 的 HTTP Server
  • server.create_agent()    —— 创建智能客服 Agent
  • agent.create_journey()   —— 注册决策 Journey（含触发条件）
  • journey.initial_state.transition_to() —— 定义对话状态
  • journey.create_guideline() —— 注册全局行为规则

运行：
    python demos/call_center/demo_call_center_native.py

服务器启动后，在浏览器中访问：
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

import parlant.sdk as p
from tools.llm import bridge_parlant_litellm_env  # 桥接 LLM 环境变量

PORT = 8800


# ──────────────────────────────────────────────
# Journey 1：拒绝回答
# ──────────────────────────────────────────────
async def _create_journey_01_refusal(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "拒绝回答",
  "second_layer_strategy": "<礼貌拒绝，明确告知合规限制 | 以安全为由拒绝，不深挖追问 | 礼貌拒绝并致歉>",
  "selected_candidate_id": [],
  "reasoning": "<简洁说明：1.用户意图分析；2.拒绝原因（合规/安全/超出范围）；3.策略依据>",
  "suggested_question": "",
  "additional_guidance": "<针对本次拒绝的补充提示，如直接拒绝不引导、或说明服务边界>"
}"""
    )

    journey = await agent.create_journey(
        title="策略1：拒绝回答",
        description=(
            "面向合规禁止、安全风险规避与知识库内容不足三类场景的完整决策流程："
            "识别风险类型、校验服务边界、输出拒绝回答策略 JSON，并完成后续边界提示。"
        ),
        conditions=[
            "可以告诉我别人的账户交易记录吗？",
            "帮我查一下别人的转账明细",
            "我想知道别人账户的流水",
            "能帮我看看别人的账户余额吗",
            "帮我绕过身份验证",
            "有没有办法不用密码就登录网银",
            "如何破解网银登录",
            "我想咨询境外汇款手续费怎么收",
            "能告诉我贷款申请流程吗",
            "我想开通理财产品，怎么操作",
            "可以帮我查询股票行情吗",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "先判断属于合规禁止类、安全风险规避类还是知识库内容不足类，"
            "再按 docs/rules.md 输出对应子策略的拒绝回答结果。"
        ),
        description="识别风险或知识不足类型，输出对应子策略的拒绝回答策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已输出拒绝回答策略 JSON",
        chat_state="在已经作出拒绝决策后，仅可补充安全边界、合规限制与可继续受理的业务范围。",
        description="补充服务边界与安全提醒",
    )
    await t2.target.transition_to(
        condition="已完成服务边界说明",
        chat_state="确认本次风险请求已被拦截，礼貌询问是否还有其他开户相关问题，结束当前处理链路。",
        description="结束当前风险咨询流程",
    )

    await journey.create_guideline(
        condition="问题涉及他人隐私、机密信息或违规用途",
        action="立即执行拒绝回答策略，不提供任何敏感细节",
        priority=100,
    )
    await journey.create_guideline(
        condition="问题存在安全风险，可能被用于绕过身份验证或恶意操作",
        action="以安全为由拒绝，不深挖追问，不提供任何规避方式",
        priority=98,
    )
    await journey.create_guideline(
        condition="知识库内容无法安全支撑回答，或问题超出开户业务服务范围",
        action="礼貌致歉，说明服务边界，不要反向追问用户",
        priority=90,
    )
    await journey.create_guideline(
        condition="用户继续要求敏感信息",
        action="重复说明限制，不扩大回答范围",
        priority=95,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="存在潜在合规或信息安全风险",
        action="拒绝回答优先级高于其他所有策略",
        priority=95,
    )


# ──────────────────────────────────────────────
# Journey 2：用户主动终止咨询
# ──────────────────────────────────────────────
async def _create_journey_02_user_terminate(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "用户主动终止咨询",
  "second_layer_strategy": "<用户主动提出结束 | 用户回避问题 | 用户切换话题>",
  "selected_candidate_id": [],
  "reasoning": "<简洁说明：1.用户意图分析；2.判断依据（明确结束/回避/切换）；3.策略匹配>",
  "suggested_question": "",
  "additional_guidance": "<针对本次终止场景的补充提示，如礼貌询问是否有其他需求>"
}"""
    )

    journey = await agent.create_journey(
        title="策略2：用户主动终止咨询",
        description=(
            "面向用户主动结束咨询、回避问题或切换话题三类场景的完整流程："
            "识别终止/回避/切换意愿、确认是否仍有其他需求、礼貌收尾。"
        ),
        conditions=[
            "我先不办了，就这样吧",
            "先到这里吧",
            "这个问题先不用继续了",
            "这个我不太方便讲",
            "算了，这个先不说了",
            "换个话题吧",
            "我换一个问题问你",
            "那个先放一边，我想问别的",
            "不用了，我自己再想想",
            "先这样吧，我回去再看看",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "结合用户当前表述与 intent_status，判断是主动提出结束、回避问题还是切换话题，"
            "选择对应子策略。"
        ),
        description="识别终止意愿、回避行为或话题切换，输出用户主动终止咨询策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已识别用户主动结束、回避或切换话题",
        chat_state="如果用户结束当前咨询，应以低打扰方式确认是否还有其他问题。",
        description="确认用户是否还需其他帮助",
    )
    await t2.target.transition_to(
        condition="已完成其他需求确认",
        chat_state="确认用户无其他需求后，完成礼貌收尾与会话结束。",
        description="礼貌收尾并关闭本次咨询",
    )

    await journey.create_guideline(
        condition="用户明确表达结束、暂停或放弃办理",
        action="不要继续推进原业务流程，优先执行终止咨询策略",
        priority=100,
    )
    await journey.create_guideline(
        condition="用户回避问题或表示不方便透露信息",
        action="表示理解，不追问敏感信息，礼貌询问是否有其他问题",
        priority=90,
    )
    await journey.create_guideline(
        condition="用户突然切换话题",
        action="顺势承接新话题，自然过渡对话，不强行拉回原话题",
        priority=80,
    )
    await journey.create_guideline(
        condition="用户虽然结束当前问题，但仍可能有其他需求",
        action="礼貌询问是否需要其他帮助",
        priority=85,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="用户主动终止当前咨询",
        action="尊重用户意愿，不强行追问或继续推销其他流程",
        priority=90,
    )


# ──────────────────────────────────────────────
# Journey 3：确认/追问
# ──────────────────────────────────────────────
async def _create_journey_03_clarify(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "确认/追问",
  "second_layer_strategy": "核实具体业务关键信息",
  "selected_candidate_id": [],
  "reasoning": "<简洁说明：1.用户意图分析；2.缺失的关键字段及原因；3.追问优先级判断>",
  "suggested_question": "<按优先级仅写单个追问问题，如：请问您是日本国籍吗？>",
  "additional_guidance": "单次只追问一个问题，不要同时提出多个问题。"
}"""
    )

    journey = await agent.create_journey(
        title="策略3：确认/追问",
        description=(
            "面向开户资质、开户所需资料、开户流程及密码相关四类意图下关键信息缺失场景的完整流程："
            "识别核心意图、定位缺失字段、单轮只追问一个最高优先级问题、等待后续补充。"
        ),
        conditions=[
            "我想了解开户条件",
            "开户需要什么条件",
            "我可以开户吗",
            "开户需要准备哪些材料",
            "需要带什么证件来开户",
            "开户流程是什么",
            "怎么开一个账户",
            "密码忘了怎么办",
            "如何修改我的密码",
            "我想开 Star One 账户，需要什么条件",
            "普通账户和 Star One 账户开户条件有什么区别",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "结合用户意图、当前 user_space 与知识前提，判断是否必须先追问关键信息，"
            "依据意图类型确定追问优先级最高的单个字段。"
        ),
        description="识别核心意图与缺失字段，输出确认/追问策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已提出单个最高优先级追问",
        chat_state="在提出单个关键问题后，等待用户补充信息，不要连续追问多个字段。",
        description="等待用户补充缺失信息",
    )
    await t2.target.transition_to(
        condition="已收到用户补充信息",
        chat_state="用户补充完关键信息后，再决定进入直接作答、拒绝回答或继续追问。",
        description="收到补充信息后恢复后续策略判断",
    )

    await journey.create_guideline(
        condition="用户核心意图明确，但办理所需关键信息缺失",
        action="只追问一个最高优先级的问题",
        priority=100,
    )
    await journey.create_guideline(
        condition="核心意图为开户资质，用户国籍或办理网点信息缺失",
        action="优先核实国籍，再确认办理网点",
        priority=92,
    )
    await journey.create_guideline(
        condition="核心意图为开户所需资料，用户国籍或永住者状态缺失",
        action="先追问国籍，非日本籍需额外确认是否为永住者",
        priority=90,
    )
    await journey.create_guideline(
        condition="核心意图为开户流程，办理网点不明或账户类型未确认",
        action="先核实办理网点，再确认账户类型",
        priority=88,
    )
    await journey.create_guideline(
        condition="核心意图为密码相关，密码类型未知",
        action="先追问密码类型（网银密码 / 卡密码 / 其他）",
        priority=85,
    )
    await journey.create_guideline(
        condition="现有知识不足以直接回答",
        action="禁止直接作答或臆测用户情况",
        priority=80,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="需要追问时",
        action=(
            "单轮只允许一个问题，问题必须指向用户自身缺失的关键信息"
            "（如国籍、年龄、网点等），禁止将知识库产品内容本身作为追问依据"
        ),
        priority=95,
    )


# ──────────────────────────────────────────────
# Journey 4：直接作答
# ──────────────────────────────────────────────
async def _create_journey_04_direct_answer(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "直接作答",
  "second_layer_strategy": "<常规问题完整解答 | 多问题拆分解答 | 产品/服务推荐 | 操作指引说明 | 告知开户资质不符 | 告知符合开户条件 | 用户主动要求转接人工>",
  "selected_candidate_id": ["<高度匹配的知识ID，无则填空数组>"],
  "reasoning": "<简洁说明：1.用户意图分析；2.知识筛选逻辑（可用知识及选用原因）；3.策略匹配依据>",
  "suggested_question": "",
  "additional_guidance": "<针对本次作答的补充说明或注意事项>"
}"""
    )

    journey = await agent.create_journey(
        title="策略4：直接作答",
        description=(
            "面向知识充分、问题清晰场景的完整流程：覆盖常规问题解答、多问题拆分、产品推荐、"
            "操作指引、资质告知及转接人工七类子场景；确认问题范围、选择高度相关知识、"
            "输出直接作答策略 JSON，并补充结构化答复要点。"
        ),
        conditions=[
            "哪一种办理速度最快？",
            "线下、邮寄和线上开户哪个最快",
            "请直接告诉我最快的开户方式",
            "Star One账户和普通账户有什么区别",
            "哪款账户更适合我",
            "怎么使用网上银行查询余额",
            "如何设置每日交易限额",
            "我已满18岁且是日本国籍，可以开户吗",
            "我是日本国籍，年满18岁，符合开户条件吗",
            "帮我接人工客服",
            "我要转人工处理",
            "我想要人工服务",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "先确认用户问题是否清晰、知识是否完整，再按 docs/rules.md 判断属于哪个子场景"
            "（常规解答、产品推荐、操作指引、资质告知或转接人工），输出直接作答策略。"
        ),
        description="确认问题清晰且知识充足，输出对应子策略的直接作答策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已完成直接作答策略判定",
        chat_state="在直接作答后，应继续保持答案结构化，避免遗漏多项信息。",
        description="整理多知识点输出结构",
    )
    await t2.target.transition_to(
        condition="已完成结构化答案整理",
        chat_state="根据已输出的结论补充必要的比较说明与下一步建议。",
        description="补充推荐与注意事项",
    )

    await journey.create_guideline(
        condition="用户问题清晰且已有完整知识支持",
        action="直接作答，并按逻辑顺序组织结构化信息",
        priority=100,
    )
    await journey.create_guideline(
        condition="用户需要多问题或多选项的对比说明",
        action="按逻辑顺序逐一回复所有问题，不得遗漏",
        priority=95,
    )
    await journey.create_guideline(
        condition="用户需要产品或服务的对比建议",
        action="结合用户需求对比选项，给出合理建议",
        priority=90,
    )
    await journey.create_guideline(
        condition="用户需要操作指引或详细步骤",
        action="提供详细操作步骤与注意事项",
        priority=88,
    )
    await journey.create_guideline(
        condition="用户询问是否符合开户资质",
        action="根据知识库内容明确告知是否符合条件，符合时推荐最优办理方式",
        priority=85,
    )
    await journey.create_guideline(
        condition="用户主动要求转接人工",
        action=(
            "执行 docs/rules.md 规定的固定话术（业务要求不可修改）："
            "您好，非常抱歉。您咨询的内容无法即时解答，即将为您转接人工客服，请您耐心等待。"
        ),
        priority=82,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="选用直接作答策略",
        action="仅引用高度相关知识，不得扩展到无关业务",
        priority=95,
    )


# ──────────────────────────────────────────────
# Journey 5：重复说明
# ──────────────────────────────────────────────
async def _create_journey_05_repeat_explain(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "重复说明",
  "second_layer_strategy": "<用户表示无法理解 | 用户重复提问 | 内容复杂需简化 | 重点内容强化>",
  "selected_candidate_id": ["<沿用上一轮已使用的知识ID，无则填空数组>"],
  "reasoning": "<简洁说明：1.用户反馈分析（无法理解/重复/要求简化/强化）；2.沿用原知识的说明；3.策略依据>",
  "suggested_question": "",
  "additional_guidance": "<重述方式调整说明，如：改用更口语简短表述、或重点强调关键结论>"
}"""
    )

    journey = await agent.create_journey(
        title="策略5：重复说明",
        description=(
            "面向用户未理解、重复提问或要求强化关键信息四类场景的完整流程："
            "识别理解障碍类型、沿用原知识重述、简化表达并强化关键点。"
        ),
        conditions=[
            "你再说简单一点，我还是没听懂哪一种办理速度最快",
            "这个我没听懂，你再解释一次",
            "你刚才说得太复杂了",
            "你再说一遍",
            "刚才那个我没听清楚",
            "最重要的地方是什么",
            "你帮我重点强调一下",
            "刚才说的我还是不明白",
            "能不能说得更简单一点",
            "我没太理解，能再说一次吗",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "判断用户是表示无法理解、重复提问、要求简化还是要求强化关键点，"
            "选择对应子策略。"
        ),
        description="识别用户理解障碍类型并输出重复说明策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已识别需要重复说明",
        chat_state="将复杂说明拆成更短、更易理解的句子，保持信息准确。",
        description="简化原有说明话术",
    )
    await t2.target.transition_to(
        condition="已完成简化表达",
        chat_state="在完成简化说明后，重复最关键的结论与对比结果。",
        description="强化用户最容易忽略的关键结论",
    )

    await journey.create_guideline(
        condition="用户表示没有理解或要求重新解释",
        action="继续使用原有知识，但要更换表达方式",
        priority=100,
    )
    await journey.create_guideline(
        condition="用户重复相同问题",
        action="提示已作答，更换角度补充说明，避免原文照抄",
        priority=92,
    )
    await journey.create_guideline(
        condition="复杂信息需要简化",
        action="拆分要点、突出结论、避免引入新概念",
        priority=90,
    )
    await journey.create_guideline(
        condition="用户要求强调或重申关键结论",
        action="重申用户遗漏的关键信息，突出最重要的结论",
        priority=85,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="进入重复说明策略",
        action="只能重述已有知识，不能编造新的解释内容",
        priority=95,
    )


# ──────────────────────────────────────────────
# Journey 6：问候/话题过渡
# ──────────────────────────────────────────────
async def _create_journey_06_greeting_transition(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "问候/话题过渡",
  "second_layer_strategy": "<简单问候回应 | 安抚负面情绪 | 用户表达感谢 | 闲聊话题应对 | 用户表达不满>",
  "selected_candidate_id": [],
  "reasoning": "<简洁说明：1.用户情绪/话题类型识别；2.子策略选择依据；3.过渡方式>",
  "suggested_question": "",
  "additional_guidance": "<针对本次情绪或话题的补充提示，如致歉安抚、引导业务或礼貌回应>"
}"""
    )

    journey = await agent.create_journey(
        title="策略6：问候/话题过渡",
        description=(
            "面向简单问候、负面情绪安抚、用户感谢（非办结场景）、闲聊话题及用户表达不满五类场景的完整流程："
            "先识别情绪或话题类型，再执行安抚或礼貌回应，最后过渡回业务处理。"
        ),
        conditions=[
            "回复也太慢了，体验太差！",
            "你们这服务怎么这么差",
            "你好，我有点着急",
            "你好",
            "您好，请问有人吗",
            "谢谢你",
            "感谢你的帮助",
            "今天天气真好啊",
            "你是AI吗",
            "你会说中文吗",
            "等了好久了，什么时候能答复我",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "先判断用户是问候、致谢（非办结场景）、闲聊还是表达不满，"
            "选择对应子策略后按 docs/rules.md 输出问候/话题过渡策略。"
        ),
        description="识别情绪或话题类型，输出对应子策略的问候/话题过渡策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已确定情绪安抚、问候回应或感谢回应类型",
        chat_state="在已经识别情绪或话题类型的前提下，优先给出共情回应、礼貌问候或感谢回应。",
        description="完成情绪安抚、礼貌问候或感谢回应",
    )
    await t2.target.transition_to(
        condition="已完成情绪安抚或话题回应",
        chat_state="安抚或回应完成后，平滑引导回业务咨询或人工处理流程。",
        description="将对话过渡回业务处理",
    )

    await journey.create_guideline(
        condition="用户表达不满、抱怨或负面情绪",
        action="先共情安抚，诚恳致歉，再引导下一步处理或转接人工",
        priority=100,
    )
    await journey.create_guideline(
        condition="用户发起简单问候或询问是否有人",
        action="礼貌友好回复，恪守礼仪，引导至业务咨询",
        priority=90,
    )
    await journey.create_guideline(
        condition="用户表达感谢但当前问题尚未办结",
        action="礼貌回应，维持良好沟通氛围，告知可继续提供帮助",
        priority=85,
    )
    await journey.create_guideline(
        condition="用户发起闲聊或与业务无关的话题",
        action="适度回应，引导回归业务咨询，不过度闲聊",
        priority=80,
    )
    await journey.create_guideline(
        condition="当前轮次属于礼貌过渡或缓和氛围",
        action="保持友好，不与用户争辩，不直接忽略情绪",
        priority=88,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="用户情绪明显波动",
        action="安抚动作优先于业务推进动作",
        priority=95,
    )


# ──────────────────────────────────────────────
# Journey 7：问题已解决
# ──────────────────────────────────────────────
async def _create_journey_07_resolved(agent: p.Agent) -> None:
    canned = await agent.create_canned_response(
        template="""{
  "first_layer_strategy": "问题已解决",
  "second_layer_strategy": "<用户明确表示问题解决 | 用户表达感谢 | 长时间无回复且问题办结 | 对话目标达成 | 用户主动提出结束/无其他问题>",
  "selected_candidate_id": [],
  "reasoning": "<简洁说明：1.用户办结信号分析；2.intent_status 与用户表述的一致性；3.策略匹配依据>",
  "suggested_question": "",
  "additional_guidance": "<简洁致谢，告知可随时继续咨询，礼貌结束对话>"
}"""
    )

    journey = await agent.create_journey(
        title="策略7：问题已解决",
        description=(
            "面向用户明确表示解决、表达感谢（办结场景）、对话目标达成或主动提出结束五类办结场景的完整流程："
            "确认问题已解决、补充可继续协助提示、礼貌结束会话。"
        ),
        conditions=[
            "没有其他问题了，谢谢！",
            "好的，问题解决了",
            "谢谢你，已经可以了",
            "好了，我知道了",
            "明白了，就这样吧",
            "我的问题都解决了",
            "好的就这样吧，谢谢",
            "非常感谢，这次解答很满意",
            "好，我懂了，不用了",
            "已经够了，谢谢你的解释",
        ],
    )

    t1 = await journey.initial_state.transition_to(
        chat_state=(
            "结合 intent_status 与用户话术，判断属于哪类办结子场景"
            "（明确解决、感谢并结束、目标达成或主动结束），输出对应子策略。"
        ),
        description="识别办结信号并输出问题已解决策略 JSON",
        canned_responses=[canned],
    )
    t2 = await t1.target.transition_to(
        condition="已识别问题办结信号",
        chat_state="在办结前简要确认用户当前需求已满足。",
        description="确认需求已满足",
    )
    await t2.target.transition_to(
        condition="已确认用户需求满足",
        chat_state="完成确认后，礼貌道别并关闭当前服务流程。",
        description="礼貌收尾并结束会话",
    )

    await journey.create_guideline(
        condition="用户明确表示问题已解决、没有其他问题或表达感谢并结束",
        action="切换到办结收尾流程，不再追加无关追问",
        priority=100,
    )
    await journey.create_guideline(
        condition="当前咨询目标已明确达成",
        action="简洁总结服务内容，礼貌道别",
        priority=92,
    )
    await journey.create_guideline(
        condition="用户主动提出结束或无其他问题",
        action="致谢并礼貌道别，告知可随时继续咨询",
        priority=90,
    )
    await journey.create_guideline(
        condition="对话长时间无回复且问题已办结",
        action="主动询问是否有其他需求，无需求则礼貌结束",
        priority=82,
    )
    await journey.create_guideline(
        condition="输出策略结果",
        action="严格输出 JSON，不附加任何额外说明文字",
        priority=100,
    )
    await journey.create_guideline(
        condition="进入问题已解决策略",
        action="保持收尾简洁自然，同时提示仍可提供帮助",
        priority=95,
    )


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
async def main() -> None:
    """使用 Parlant 原生 SDK 初始化 7 个 Journey，启动 HTTP Server。"""
    bridge_parlant_litellm_env()

    print("=" * 72)
    print("  银行智能呼叫中心  —  Parlant 原生 SDK 演示")
    print("=" * 72)
    print("\n正在初始化 Agent 与 7 个决策 Journey，请稍候...")

    async with p.Server(
        nlp_service=p.NLPServices.litellm,
        port=PORT,
    ) as server:
        agent = await server.create_agent(
            name="银行呼叫中心智能客服",
            description="依据 docs/rules.md 输出银行呼叫中心应答策略 JSON",
        )

        await _create_journey_01_refusal(agent)
        print("  ✓ 策略1：拒绝回答")
        await _create_journey_02_user_terminate(agent)
        print("  ✓ 策略2：用户主动终止咨询")
        await _create_journey_03_clarify(agent)
        print("  ✓ 策略3：确认/追问")
        await _create_journey_04_direct_answer(agent)
        print("  ✓ 策略4：直接作答")
        await _create_journey_05_repeat_explain(agent)
        print("  ✓ 策略5：重复说明")
        await _create_journey_06_greeting_transition(agent)
        print("  ✓ 策略6：问候/话题过渡")
        await _create_journey_07_resolved(agent)
        print("  ✓ 策略7：问题已解决")

        print(f"\n✅ {agent.name} 配置完成（7 个 Journey 已注册）")
        print(f"🌐 服务器即将启动，请在浏览器中访问：http://localhost:{PORT}")
        print("   在对话框中输入消息即可与 Agent 交互。")
        print("   按 Ctrl+C 停止服务器。\n")
    # async with 块退出时，Parlant HTTP Server 在此处启动并阻塞，直到 Ctrl+C


if __name__ == "__main__":
    asyncio.run(main())
