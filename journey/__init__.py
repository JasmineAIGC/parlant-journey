"""
journey
~~~~~~~

基于 Parlant 的声明式 Journey 构建框架。

公开 API — 调用方只需从这里导入::

    from journey import (
        create_journey,           # 创建单个 Journey
        register_journeys,        # 批量注册 Journey
        JourneyConfig,            # 顶层配置
        StateConfig,              # 状态节点配置
        TransitionConfig,         # 状态转移配置
        GuidelineConfig,          # 行为规则配置
        CannedResponseConfig,     # 预设话术配置
        JourneyDSL,               # 流式构建 API（快速上手）
        load_journey,             # 从 YAML/JSON 文件加载配置
        load_journey_dict,        # 从字典对象加载配置
        JourneyGenerator,         # LLM 驱动的配置自动生成器
        generate_journey_config,  # 一行代码生成配置（便捷函数）
        LLMConfig,                # LLM 连接配置（来源：tools.llm）
        LLMClient,                # 基于 LiteLLM 的异步 LLM 客户端（来源：tools.llm）
    )

LLMClient / LLMConfig 的规范来源是 ``tools.llm``，此处 re-export 保持向后兼容。
内部实现细节（``JourneyBuilder`` 等子模块）**不属于公开 API**，可能随版本变化。
"""

from .factory import create_journey, register_journeys
from .models import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)
from .dsl import JourneyDSL
from .loader import load_journey, load_journey_dict
from .generator import JourneyGenerator, LLMClient, LLMConfig, generate_journey_config

__all__ = [
    # 工厂函数（主入口）
    "create_journey",
    "register_journeys",
    # 数据模型（构造配置用）
    "JourneyConfig",
    "StateConfig",
    "TransitionConfig",
    "GuidelineConfig",
    "CannedResponseConfig",
    # 快速构建工具（无 LLM）
    "JourneyDSL",
    "load_journey",
    "load_journey_dict",
    # LLM 自动生成工具
    "JourneyGenerator",
    "generate_journey_config",
    "LLMConfig",
    "LLMClient",
]

