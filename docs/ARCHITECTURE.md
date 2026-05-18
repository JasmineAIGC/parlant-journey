# parlant-journey — Architecture

> 本文档是 parlant-journey 的**权威技术方案文档**，综合自 `docs/` 下的两份设计文档，
> 适用于所有贡献者和调用方参考。

---

## 一、定位与边界

### 1.1 唯一职责

```
输入：p.Agent  +  JourneyConfig
输出：p.Journey
```

本库**只做一件事**：将声明式的 `JourneyConfig` 构建为 Parlant 的 `Journey` 对象。

**不在本库职责范围内**：Server 创建、Agent 创建、NLP 服务配置、Session 管理、Tool 定义。

### 1.2 与调用方的关系

```
┌─────────────────────────────────────────┐
│        调用方（Parlant 对话系统）         │
│                                         │
│   server = p.Server(...)                │
│   agent  = await server.create_agent()  │
│                                         │
│   from parlant_journey import           │
│       create_journey, JourneyConfig     │
│                                         │
│   await create_journey(agent, config)   │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│           parlant-journey               │
│                                         │
│   唯一职责：将 JourneyConfig            │
│   翻译为 Parlant Journey API 调用       │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│           Parlant SDK                   │
│   agent.create_journey()                │
│   agent.create_canned_response()        │
│   state.transition_to()                 │
│   journey.create_guideline()            │
└─────────────────────────────────────────┘
```

### 1.3 职责边界表

| 职责 | 归属 |
|------|------|
| Server / Agent / Session 管理 | 调用方 |
| NLP 服务配置 | 调用方 |
| `@tool` 函数定义 | 调用方 |
| Journey 构建（Config → Parlant Journey） | **本库** |
| 三层控制封装（流程 / 行为 / 话术） | **本库** |
| Parlant Journey API 的正确调用顺序 | **本库** |

---

## 二、目录结构

```
parlant-journey/
│
├── parlant_journey/          # 可安装包，对外暴露的唯一命名空间
│   ├── __init__.py           # 公开 API 入口（re-export）
│   ├── models.py             # 纯数据层（Config Dataclasses）
│   ├── builder.py            # 翻译层（JourneyBuilder，内部私有）
│   └── factory.py            # 公开入口层（create_journey / register_journeys）
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py        # 数据模型单元测试
│   ├── test_builder.py       # 构建器测试（mock Parlant SDK）
│   └── test_factory.py       # 工厂函数测试
│
├── examples/
│   └── fraud.py              # 欺诈处理 Journey 完整示例
│
├── docs/
│   ├── ARCHITECTURE_Version2.md   # 架构设计草稿（参考）
│   └── parlant-journey.md         # Guidelines & Canned Responses 设计（参考）
│
├── ARCHITECTURE.md           # ← 本文档（权威技术方案）
├── pyproject.toml
└── README.md
```

---

## 三、内部分层

### 3.1 层次图

```
parlant_journey/
│
├── models.py          纯数据层
│   │
│   │  @dataclass CannedResponseConfig   template, strict
│   │  @dataclass GuidelineConfig        condition, action, priority
│   │  @dataclass StateConfig            name, chat/tool, canned, guidelines
│   │  @dataclass TransitionConfig       source, target, condition
│   └─ @dataclass JourneyConfig          title, triggers, states, transitions, global_guidelines
│
├── builder.py         翻译层（Config → Parlant API）
│   │
│   │  class JourneyBuilder                     [内部私有，不对外暴露]
│   │    ├── build()
│   │    ├── _create_journey()
│   │    ├── _create_canned_responses()
│   │    ├── _build_dag()
│   │    ├── _build_state_kwargs()
│   │    ├── _register_node_guidelines()
│   │    └── _register_global_guidelines()
│   └─
│
└── factory.py         公开入口层
    │
    │  async create_journey(agent, config) → Journey
    └─ async register_journeys(agent, configs) → list[Journey]
```

### 3.2 分层职责表

| 文件 | 职责 | 依赖 |
|------|------|------|
| `models.py` | 定义纯数据结构，无任何副作用 | 无（仅 dataclasses / typing） |
| `builder.py` | 将 Config 翻译为 Parlant API 调用序列 | `models.py` + `parlant.sdk` |
| `factory.py` | 对外暴露简洁入口，实例化并驱动 Builder | `builder.py` |
| `__init__.py` | 控制公开 API 边界，re-export 必要符号 | `models.py` + `factory.py` |

---

## 四、公开 API

调用方**只需从一个命名空间导入**：

```python
from parlant_journey import (
    # 工厂函数（主入口）
    create_journey,
    register_journeys,

    # 数据模型（构造配置用）
    JourneyConfig,
    StateConfig,
    TransitionConfig,
    GuidelineConfig,
    CannedResponseConfig,
)
```

`JourneyBuilder` 是内部实现细节，不对外暴露。

---

## 五、三层控制模型

本库将 Journey 的控制能力封装为三个正交维度，调用方按需组合：

```
维度               配置字段                        Parlant 映射
─────────────────────────────────────────────────────────────────
流程（路径）       states + transitions            transition_to()
行为（规则）       guidelines（节点级 + 全局）     journey.create_guideline()
话术（合规）       canned_response（节点级）       canned_responses= + CompositionMode
```

三个维度互相独立，可单独使用，也可任意组合：

- 只需流程控制 → 仅填 `states` + `transitions`
- 需要行为约束 → 追加 `guidelines`
- 需要合规话术 → 追加 `canned_response`（`strict=True` → `STRICT` 模式，零幻觉）

---

## 六、数据流

```
调用方
  │  from parlant_journey import create_journey, JourneyConfig
  │  await create_journey(agent, config)
  │
  ▼
factory.create_journey(agent, config)
  │
  ▼
JourneyBuilder(agent, config).build()
  │
  ├─ ① _create_journey()
  │       agent.create_journey(title, triggers, description)
  │
  ├─ ② _create_canned_responses()
  │       agent.create_canned_response(template=cfg.template)
  │       → 内部维护 canrep_map: {state_name → CannedResponseId}
  │
  ├─ ③ _build_dag()
  │       确定根节点（无入边的 source 状态）
  │       根节点 → journey.initial_state.transition_to(chat/tool, canned, mode)
  │       遍历 transitions（拓扑顺序）:
  │         新节点 → source_state.transition_to(chat/tool, canned, mode, condition)
  │         已有节点 → source_state.transition_to(state=existing)   ← 汇聚
  │
  ├─ ④ _register_node_guidelines()
  │       journey.create_guideline(condition, action, priority)   × 每个节点
  │
  └─ ⑤ _register_global_guidelines()
          journey.create_guideline(condition, action, priority)   × 全局
```

---

## 七、DAG 拓扑支持

`transitions` 字段描述有向图的边集合，Builder 自动处理三种拓扑：

```
线性    A → B → C

分支    A → B (condition="X")
        A → C (condition="Y")
        A → D (condition="Z")

汇聚    B → E
        C → E   ← 自动复用同一 JourneyState 对象
        D → E
```

**约定**：`transitions` 列表须满足**拓扑顺序**（父节点的边先于子节点的边出现）。

根节点（无入边的 source 状态）由 Builder 自动检测，并通过 `journey.initial_state.transition_to()` 创建。

---

## 八、快速开始

### 安装

```bash
pip install parlant-journey
```

### 最简调用

```python
import parlant.sdk as p
from parlant_journey import create_journey, JourneyConfig, StateConfig, TransitionConfig

async with p.Server(...) as server:
    agent = await server.create_agent(...)

    config = JourneyConfig(
        title="欺诈处理",
        triggers=["我卡被盗刷了", "可疑交易"],
        description="处理欺诈交易的标准流程",
        states=[
            StateConfig(name="verify", chat_state="请验证您的身份"),
            StateConfig(name="closing", chat_state="感谢您的配合"),
        ],
        transitions=[
            TransitionConfig(source="verify", target="closing", condition="验证成功"),
        ],
    )

    journey = await create_journey(agent, config)
```

完整示例见 [`examples/fraud.py`](examples/fraud.py)。

### 批量注册

```python
from parlant_journey import register_journeys

journeys = await register_journeys(agent, [
    fraud_config,
    account_opening_config,
    loan_consult_config,
])
```

---

## 九、依赖声明

```toml
[project]
name = "parlant-journey"
dependencies = [
    "parlant",   # 唯一运行时依赖
]
```

本库不引入任何其他运行时依赖。

---

## 十、测试

```bash
pip install -e ".[dev]"
pytest
```

测试使用 `unittest.mock` 完整模拟 Parlant SDK，无需启动真实服务器。

---

## 参考文档

| 文档 | 说明 |
|------|------|
| [`docs/ARCHITECTURE_Version2.md`](docs/ARCHITECTURE_Version2.md) | 架构设计草稿（数据流、分层设计） |
| [`docs/parlant-journey.md`](docs/parlant-journey.md) | Guidelines & Canned Responses 功能设计 |
| [`examples/fraud.py`](examples/fraud.py) | 欺诈处理完整示例（三层控制综合演示） |
