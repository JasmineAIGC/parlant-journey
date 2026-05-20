# parlant-journey

> 基于 [Parlant](https://github.com/emcie-co/parlant) 的**独立 Journey 模块**。
> 专注且仅专注于 Journey 的声明式构建，可被任意 Parlant 对话系统作为依赖库调用。

## 安装

```bash
pip install -r requirements.txt
# 或仅安装核心
pip install parlant
```

## 快速开始

```python
import parlant.sdk as p
from journey import (
    create_journey,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
    GuidelineConfig,
    CannedResponseConfig,
)

async with p.Server(...) as server:
    agent = await server.create_agent(...)

    config = JourneyConfig(
        title="欺诈处理",
        conditions=["我卡被盗刷了", "可疑交易"],
        description="处理欺诈交易的标准流程",
        states=[
            StateConfig(
                name="verify",
                chat_state="请验证您的身份",
                canned_response=CannedResponseConfig(
                    template="为了账户安全，请提供您的姓名和身份证后四位。",
                    strict=True,
                ),
            ),
            StateConfig(name="closing", chat_state="感谢您的配合"),
        ],
        transitions=[
            TransitionConfig(source="verify", target="closing", condition="验证成功"),
        ],
        global_guidelines=[
            GuidelineConfig(condition="用户情绪激动", action="先安抚情绪", priority=90),
        ],
    )

    journey = await create_journey(agent, config)
```

## 三层控制模型

| 维度 | 配置字段 | 效果 |
|------|---------|------|
| 流程（路径） | `states` + `transitions` | 声明式 DAG，自动处理分支与汇聚 |
| 行为（规则） | `guidelines`（节点级 + 全局） | 条件触发的行为约束 |
| 话术（合规） | `canned_response`（节点级） | 关键节点锁定话术，零幻觉 |

## 完整对话演示

可直接运行独立的欺诈处理全流程演示：

```bash
# 1. 在项目根目录 .env 中配置 LLM 服务
# 2. 安装依赖
pip install -r requirements.txt
# 3. 运行演示
python demos/fraud/demo_fraud.py
python demos/rules/demo_rules.py
```

## 目录说明

```
journey/              核心包（Python import: from journey import ...）
tests/
  01_test_models.py   数据模型单元测试
  02_test_builder.py  构建器单元测试（mock SDK）
  03_test_factory.py  工厂函数单元测试
demos/fraud/          独立欺诈处理演示及其配置文件
demos/rules/          docs/rules.md 对应的多 Journey 策略演示
docs/ARCHITECTURE.md  完整技术方案（权威文档）
```

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 完整技术方案（推荐阅读）
- [demos/fraud/demo_fraud.py](demos/fraud/demo_fraud.py) — 欺诈处理完整对话演示
- [demos/rules/demo_rules.py](demos/rules/demo_rules.py) — docs/rules.md 多 Journey 策略演示

## 测试

```bash
pip install -e ".[dev]"
pytest tests/01_test_models.py tests/02_test_builder.py tests/03_test_factory.py -v
```
