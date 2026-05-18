# parlant-journey

> 基于 [Parlant](https://github.com/emcie-co/parlant) 的**独立 Journey 模块**。
> 专注且仅专注于 Journey 的声明式构建，可被任意 Parlant 对话系统作为依赖库调用。

## 安装

```bash
pip install parlant-journey
```

## 快速开始

```python
from parlant_journey import create_journey, JourneyConfig, StateConfig, TransitionConfig

journey = await create_journey(agent, config)
```

详见 `examples/fraud.py` 和 `ARCHITECTURE.md`。
