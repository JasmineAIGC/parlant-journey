# parlant-journey

> 基于 [Parlant](https://github.com/emcie-co/parlant) 的**独立 Journey 模块**。
> 专注且仅专注于 Journey 的声明式构建，可被任意 Parlant 对话系统作为依赖库调用。

---

## 仓库名称

```
parlant-journey
```

命名逻辑：`parlant`（宿主框架）+ `journey`（本库唯一职责），语义精确，无歧义。

---

## 一、定位

```
┌─────────────────────────────────────────┐
│        你的 Parlant 对话系统            │
│                                         │
│   server = p.Server(...)                │
│   agent  = await server.create_agent()  │
│                                         │
│   ← 调用 parlant-journey ──────────────►│
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
│   agent.create_journey()               │
│   transition_to()                       │
│   create_guideline()                    │
│   create_canned_response()              │
└─────────────────────────────────────────┘
```

**本库不包含**：Server 创建、Agent 创建、NLP 服务配置、Session 管理、Tool 定义——这些均属于调用方的职责。

---

## 二、代码架构

### 2.1 目录结构

```
parlant-journey/
│
├── parlant_journey/               # 可安装包，对外暴露的唯一命名空间
│   ├── __init__.py                # 公开 API 入口（re-export）
│   ├── models.py                  # 数据模型（Config Dataclasses）
│   ├── builder.py                 # 构建器（JourneyBuilder）
│   └── factory.py                 # 工厂函数（create_journey / register_journeys）
│
├── tests/
│   ├── test_models.py             # 数据模型单元测试
│   ├── test_builder.py            # 构建器单元测试
│   └── test_factory.py            # 工厂函数集成测试
│
├── examples/
│   └── fraud.py                   # 调用示例（欺诈场景）
│
├── pyproject.toml                 # 包配置，依赖声明（parlant）
└── ARCHITECTURE.md
```

---

### 2.2 公开 API（`parlant_journey/__init__.py`）

调用方只需从这一个命名空间导入：

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

库内部的 `JourneyBuilder` 不对外暴露，属于实现细节。

---

### 2.3 内部分层

```
parlant_journey/
│
├── models.py          纯数据层
│   │
│   │  @dataclass CannedResponseConfig   template, strict
│   │  @dataclass GuidelineConfig        condition, action, priority
│   │  @dataclass StateConfig            name, chat/tool, canned, guidelines
│   │  @dataclass TransitionConfig       source, target, condition
│   └─ @dataclass JourneyConfig          title, triggers, states, transitions, guidelines
│
├── builder.py         翻译层（Config → Parlant API）
│   │
│   │  class JourneyBuilder
│   │    ├── build()
│   │    ├── _create_journey()
│   │    ├── _create_canned_responses()
│   │    ├── _build_dag()
│   │    ├── _register_node_guidelines()
│   │    └── _register_global_guidelines()
│   └─  （内部私有，不对外暴露）
│
└── factory.py         公开入口层
    │
    │  async create_journey(agent, config) → Journey
    └─ async register_journeys(agent, configs) → list[Journey]
```

---

### 2.4 分层职责

| 文件 | 职责 | 依赖 |
|------|------|------|
| `models.py` | 定义纯数据结构，无任何副作用 | 无（仅 dataclasses / typing） |
| `builder.py` | 将 Config 翻译为 Parlant API 调用序列 | `models.py` + `parlant.sdk` |
| `factory.py` | 对外暴露简洁入口，实例化并驱动 Builder | `builder.py` |
| `__init__.py` | 控制公开 API 边界，re-export 必要符号 | `models.py` + `factory.py` |

---

### 2.5 数据流

```
调用方代码
  │
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
  │        agent.create_journey(triggers=config.triggers)
  │
  ├─ ② _create_canned_responses()
  │        agent.create_canned_response(template=cfg.template)
  │        → 内部维护 canrep_map: {state_name → CannedResponseId}
  │
  ├─ ③ _build_dag()
  │        遍历 config.transitions（拓扑顺序）
  │          新节点 → transition_to(chat/tool, canned_responses, composition_mode)
  │          已有节点 → transition_to(state=existing)   ← 汇聚
  │
  ├─ ④ _register_node_guidelines()
  │        journey.create_guideline(condition, action, priority)
  │
  └─ ⑤ _register_global_guidelines()
           journey.create_guideline(condition, action, priority)
```

---

## 三、架构说明

### 3.1 单一职责边界

本库**只做一件事**：将声明式的 `JourneyConfig` 构建为 Parlant 的 `Journey` 对象。

```
输入：p.Agent + JourneyConfig
输出：p.Journey
```

Server、Agent、Session、Tool、NLP 服务等概念均不在本库的职责范围内。

---

### 3.2 调用方集成方式

**安装**：

```bash
pip install parlant-journey
```

**最简调用**：

```python
import parlant.sdk as p
from parlant_journey import create_journey, JourneyConfig, StateConfig, TransitionConfig

async with p.Server(...) as server:
    agent = await server.create_agent(...)

    config = JourneyConfig(
        title="...",
        triggers=["..."],
        description="...",
        states=[StateConfig(...)],
        transitions=[TransitionConfig(...)],
    )

    journey = await create_journey(agent, config)
```

调用方**不需要**了解 `JourneyBuilder`、`transition_to`、`create_canned_response` 等 Parlant 内部 API。

---

### 3.3 三层控制模型

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
- 需要合规话术 → 追加 `canned_response`

---

### 3.4 DAG 拓扑支持

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

**约定**：`transitions` 列表须满足拓扑顺序（父节点的边先于子节点的边）。

---

### 3.5 包依赖声明（pyproject.toml）

```toml
[project]
name = "parlant-journey"
dependencies = [
    "parlant",      # 唯一外部依赖
]
```

本库不引入任何其他运行时依赖。

---

### 3.6 与调用方的职责边界

| 职责 | 归属 |
|------|------|
| Server / Agent / Session 管理 | 调用方 |
| NLP 服务配置 | 调用方 |
| `@tool` 函数定义 | 调用方 |
| Journey 构建（Config → Parlant Journey） | **本库** |
| 三层控制封装（流程/行为/话术） | **本库** |
| Parlant Journey API 的正确调用姿势 | **本库** |