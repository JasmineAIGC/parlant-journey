在我们的 Journey 框架中增加 **Guidelines** 和 **Canned Responses** 来大幅强化控制力。

---

### **一、为什么需要 Guidelines 和 Canned Responses？**

- **Guidelines**：定义“在什么条件下，应该做什么行为”（动态注入）。
- **Canned Responses**：关键合规节点**强制使用预设话术**，彻底杜绝幻觉（金融场景极其重要）。
- 二者结合可以实现**“流程 + 行为 + 话术”** 三层强控制。

---

### **二、扩展后的配置模型**

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class JourneyStateConfig:
    """增强版状态配置"""
    name: str
    goal: str
    chat_state: Optional[str] = None
    tool_state: Optional[Any] = None
    tool_instruction: Optional[str] = None
    description: str = ""
    policy: Dict[str, Any] = field(default_factory=dict)
    
    # 新增：Guidelines 和 Canned Responses
    guidelines: List[Dict[str, Any]] = field(default_factory=list)   # 本状态专属规则
    canned_response: Optional[str] = None                            # 强制使用预设回复


@dataclass
class JourneyConfig:
    """增强版 Journey 配置"""
    title: str
    conditions: List[str]
    description: str
    states: List[JourneyStateConfig]
    transitions: List[JourneyTransitionConfig]
    scenario: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Journey 级别全局 Guidelines
    global_guidelines: List[Dict[str, Any]] = field(default_factory=list)
```

---

### **三、完整实现代码（journey_framework.py）**

```python
import parlant.sdk as p
from typing import Dict

async def create_configurable_journey(agent: p.Agent, config: JourneyConfig) -> p.Journey:
    journey = await agent.create_journey(
        title=config.title,
        conditions=config.conditions,
        description=config.description
    )

    # ==================== 1. 创建 Journey 级别 Guidelines ====================
    for g in config.global_guidelines:
        await journey.create_guideline(
            condition=g.get("condition"),
            action=g.get("action"),
            priority=g.get("priority", 50)
        )

    # ==================== 2. 创建所有状态 ====================
    state_map = {}
    current = journey.initial_state

    for state_cfg in config.states:
        desc = state_cfg.description or f"{state_cfg.name} - {state_cfg.goal}"

        if state_cfg.tool_state:
            transition = await current.transition_to(
                tool_state=state_cfg.tool_state,
                tool_instruction=state_cfg.tool_instruction or "",
                description=desc
            )
        else:
            chat_instruction = state_cfg.chat_state or f"当前目标：{state_cfg.goal}"
            transition = await current.transition_to(
                chat_state=chat_instruction,
                description=desc
            )

        state_map[state_cfg.name] = transition
        current = transition.target

        # ==================== 3. 为当前状态绑定 Guidelines 和 Canned Responses ====================
        # 创建 Canned Response（关键合规节点）
        if state_cfg.canned_response:
            canned = await agent.create_canned_response(state_cfg.canned_response)
            await journey.create_guideline(
                condition=f"处于 {state_cfg.name} 状态",
                action="使用预设回复",
                canned_responses=[canned],
                composition_mode=p.CompositionMode.STRICT,   # 严格模式
                priority=100
            )

        # 创建 State 专属 Guidelines
        for g in state_cfg.guidelines:
            await journey.create_guideline(
                condition=g.get("condition", f"处于 {state_cfg.name} 状态"),
                action=g.get("action"),
                priority=g.get("priority", 80)
            )

    # ==================== 4. 创建 Transitions ====================
    for trans in config.transitions:
        if trans.source not in state_map or trans.target not in state_map:
            continue
        source = state_map[trans.source]
        if trans.condition:
            await source.target.transition_to(condition=trans.condition, state=state_map[trans.target].target)
        else:
            await source.target.transition_to(state=state_map[trans.target].target)

    # 生成流程图
    mermaid = generate_mermaid(config)
    print(f"✅ Journey '{config.title}' 创建成功（含 Guidelines & Canned Responses）")
    print(mermaid)
    
    return journey
```

---

### **四、实际使用示例（欺诈处理 Journey）**

```python
fraud_config = JourneyConfig(
    title="处理潜在欺诈交易",
    conditions=["客户报告可疑交易", "被盗刷", "欺诈预警"],
    description="...",

    states=[
        JourneyStateConfig(
            name="identity_verify",
            goal="完成用户身份验证",
            chat_state="严格进行身份验证",
            canned_response="为了您的账户安全，我需要先验证您的身份。请提供您的姓名、身份证后四位或银行卡后四位。",
            policy={"strict_mode": True}
        ),
        JourneyStateConfig(
            name="high_risk",
            goal="执行高风险处置",
            chat_state="告知客户风险较高，立即引导冻结账户和挂失",
            guidelines=[
                {
                    "condition": "处于 high_risk 状态",
                    "action": "必须强调立即操作的重要性，并提供操作指引",
                    "priority": 95
                }
            ]
        ),
        # 其他状态...
    ],

    global_guidelines=[
        {
            "condition": "用户情绪激动",
            "action": "先安抚情绪，再继续流程",
            "priority": 90
        }
    ]
)
```

---

### **五、强化控制效果**

- **Canned Responses**：在 `identity_verify` 等关键节点，**强制使用固定话术**，几乎零幻觉。
- **Guidelines**：可以针对不同状态设置不同行为规则。
- **STRICT 模式**：在合规节点使用 `CompositionMode.STRICT`，进一步加强控制。
- **优先级**：可以通过 `priority` 控制规则生效顺序。

---

**总结**：  
通过这种方式，你可以实现**流程（Journey） + 行为（Guidelines） + 话术（Canned Responses）** 三层精细控制，完美满足银行等强监管场景的需求。


```
flowchart TD
    title[处理潜在欺诈交易]
    style title fill:#f0f0f0,stroke:#333,stroke-width:2px

    start[start<br/><small>安抚客户情绪并启动标准化流程</small>]
    identity_verify[identity_verify<br/><small>完成用户身份验证（KYC）</small>]
    collect_info[collect_info<br/><small>收集可疑交易的关键信息</small>]
    risk_assess[/risk_assess<br/><small>评估交易风险等级</small>/]
    high_risk[high_risk<br/><small>执行高风险处置措施</small>]
    medium_risk[medium_risk<br/><small>执行中风险处置措施</small>]
    low_risk[low_risk<br/><small>执行低风险处置措施</small>]
    closing[closing<br/><small>完成流程收尾与客户确认</small>]
    endNode((流程结束))

    start --> identity_verify
    identity_verify -->|身份验证成功| collect_info
    identity_verify -->|验证失败达到3次| closing
    collect_info --> risk_assess
    risk_assess -->|风险等级为 HIGH| high_risk
    risk_assess -->|风险等级为 MEDIUM| medium_risk
    risk_assess -->|风险等级为 LOW| low_risk
    high_risk --> closing
    medium_risk --> closing
    low_risk --> closing
    closing --> endNode

    style risk_assess fill:#e0f2fe,stroke:#0369a1
    style endNode fill:#fee2e2,stroke:#b91c1c,color:white,stroke-width:3px
```

### **一、整体 Journey 的输入与输出**

#### **1. 输入（Inputs）**

**触发输入（激活 Journey）**：
- 用户消息中包含以下任意关键词/意图（由 `conditions` 定义）：
  - “我卡被盗刷了”
  - “可疑交易”
  - “账户异常”
  - “系统发出欺诈预警”等

**运行时每轮输入**：
- 用户的自然语言消息
- 当前 Session 上下文（历史对话、客户基本信息等）
- 当前 Journey 状态（系统自动维护）

**配置层输入**（开发者提供）：
- `JourneyConfig`（标题、states、transitions 等配置）

#### **2. 输出（Outputs）**

**每轮对话输出**：
- Agent 回复给用户的消息（自然语言）
- 状态迁移结果（当前处于哪个 State）
- 如果是 Tool State，则输出 Tool 调用结果（例如风险等级）

**整体 Journey 输出**：
- 完整的对话轨迹（走了哪条路径：高风险 / 中风险 / 低风险）
- 审计日志（每个状态的决策依据、可追溯）
- Journey 结束信号（到达 `closing` → `endNode`）

---

### **二、流程中每个主要节点的输入输出**

| 节点              | 输入（什么触发）                          | 输出（Agent 行为 + 下一状态）                     |
|-------------------|------------------------------------------|------------------------------------------------|
| **start**         | 用户报告异常                             | 安抚情绪 + 进入身份验证                        |
| **identity_verify** | 上一状态完成                             | 要求用户提供身份信息 → 验证成功/失败           |
| **collect_info**  | 身份验证通过                             | 引导用户提供交易细节                           |
| **risk_assess**   | 信息收集完成                             | **调用工具** → 返回风险等级（HIGH/MEDIUM/LOW） |
| **high_risk**     | 风险等级=HIGH                            | 引导冻结账户、挂失                             |
| **medium_risk**   | 风险等级=MEDIUM                          | 建议改密码、设限额                             |
| **low_risk**      | 风险等级=LOW                             | 告知安全 + 提醒警惕                           |
| **closing**       | 处置完成                                 | 总结 + 感谢 + 邀请评价 → 结束 Journey         |

---

### **三、如何被其他人使用（集成方式）**

#### **1. 开发者 / 工程师使用方式**

```python
# 1. 定义配置（推荐放在单独文件，如 journeys/fraud.py）
from journey_framework import JourneyConfig, JourneyStateConfig, JourneyTransitionConfig

fraud_config = JourneyConfig(
    title="处理潜在欺诈交易",
    # ... 其他配置
)

# 2. 创建 Journey（初始化时执行一次）
journey = await create_configurable_journey(agent, fraud_config)

# 3. Parlant 会自动管理后续所有对话
```

#### **2. 业务 / 产品人员使用方式**
- 只需修改 `JourneyConfig` 中的 `goal`、`chat_state`、`conditions` 等文字内容即可调整流程。
- 修改后重新运行 `create_configurable_journey()` 即可生效。
- 可通过生成的 **Mermaid 流程图** 直观评审流程是否合理。

#### **3. 在完整系统中的使用**

```python
# 在 Agent 初始化完成后统一注册多个 Journey
await create_configurable_journey(agent, fraud_config)
await create_configurable_journey(agent, account_opening_config)
await create_configurable_journey(agent, loan_consult_config)
```

Parlant 会根据用户输入**自动匹配并激活**对应的 Journey。

---

### **四、总结：输入输出全景**

- **输入** = 用户消息 + 当前状态 + 配置定义
- **输出** = 受控的 Agent 回复 + 状态迁移 + 执行轨迹
- **核心价值** = 把原本不可控的多轮对话，变成了**结构清晰、可配置、可审计**的标准流程

这个 Journey 就像给 Agent 安装了一个**业务 autopilot（自动驾驶）**，在不同阶段给予不同指令，确保既专业又合规。
