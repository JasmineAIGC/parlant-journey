"""
journey
~~~~~~~

基于 Parlant 的声明式 Journey 构建框架。

本模块导出公开 API（create_journey / register_journeys / JourneyDSL / load_journey 等）。
LLMClient / LLMConfig 从 ``tools.llm`` re-export，用于兼容旧调用路径。
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
