"""
journey.loader
~~~~~~~~~~~~~~

从 YAML / JSON 文件（或字典）加载 :class:`~journey.models.JourneyConfig`，
实现"零 Python 代码"的场景迁移：只需维护一个配置文件即可切换业务场景。

**格式选择建议**：
  - JSON — 无额外依赖（Python 内置），适合程序生成的配置。
  - YAML — 需要安装 pyyaml（``pip install pyyaml``），人类可读性更佳，
    适合手动维护、含大量中文的配置文件。

YAML 格式示例（参见 ``tests/fraud.yaml``）::

    title: "处理潜在欺诈交易"
    description: "..."
    conditions:
      - "我卡被盗刷了"
    states:
      - name: verify
        description: "身份验证"
        chat_state: "请验证您的身份"
        canned_response:
          template: "为了账户安全，请提供..."
          strict: true
    transitions:
      - source: verify
        target: closing
        condition: "验证成功"
    global_guidelines:
      - condition: "用户情绪激动"
        action: "先安抚情绪"
        priority: 90
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .models import (
    CannedResponseConfig,
    GuidelineConfig,
    JourneyConfig,
    StateConfig,
    TransitionConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------


def load_journey(path: str) -> JourneyConfig:
    """从 YAML 或 JSON 文件加载并返回 :class:`~journey.models.JourneyConfig`。

    文件格式由扩展名自动识别：``.yaml`` / ``.yml`` → YAML；其余 → JSON。

    Args:
        path: 配置文件的路径（绝对路径或相对于当前工作目录的路径）。

    Returns:
        解析完成的 :class:`~journey.models.JourneyConfig` 对象。

    Raises:
        FileNotFoundError: 文件不存在。
        ImportError:        读取 YAML 文件时未安装 pyyaml。
        ValueError:         配置数据格式不合法。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在：{path}")

    ext = os.path.splitext(path)[1].lower()
    logger.debug("[load_journey] 加载配置文件  path=%r  format=%s", path, "yaml" if ext in (".yaml", ".yml") else "json")
    with open(path, encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            data = _load_yaml(fh)
        else:
            data = json.load(fh)

    config = _parse(data)
    logger.debug("[load_journey] 加载完成  title=%r  states=%d", config.title, len(config.states))
    return config


def load_journey_dict(data: Dict[str, Any]) -> JourneyConfig:
    """从已解析的字典对象构建 :class:`~journey.models.JourneyConfig`。

    适用于配置数据已经从其他渠道（数据库、环境变量等）加载的场景。

    Args:
        data: 符合 Journey 配置格式的字典。

    Returns:
        解析完成的 :class:`~journey.models.JourneyConfig` 对象。

    Raises:
        ValueError: 配置数据格式不合法。
    """
    logger.debug("[load_journey_dict] 从字典解析配置")
    config = _parse(data)
    logger.debug("[load_journey_dict] 解析完成  title=%r  states=%d", config.title, len(config.states))
    return config


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------


def _load_yaml(fh: Any) -> Dict[str, Any]:
    """读取 YAML 文件句柄，返回 Python 字典。"""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "读取 YAML 配置文件需要安装 pyyaml：\n"
            "    pip install pyyaml"
        )
    return yaml.safe_load(fh) or {}


def _parse(data: Dict[str, Any]) -> JourneyConfig:
    """将字典数据解析为 JourneyConfig。"""
    _require_key(data, "title")
    _require_key(data, "states")

    return JourneyConfig(
        title=data["title"],
        description=data.get("description", ""),
        conditions=data.get("conditions") or data.get("triggers", []),
        states=[_parse_state(s) for s in data["states"]],
        transitions=[_parse_transition(t) for t in data.get("transitions", [])],
        global_guidelines=[
            _parse_guideline(g) for g in data.get("global_guidelines", [])
        ],
    )


def _parse_state(data: Dict[str, Any]) -> StateConfig:
    """将单条状态字典解析为 StateConfig。"""
    _require_key(data, "name", context="states 条目")

    # 解析预设话术
    cr_data = data.get("canned_response")
    canned = None
    if cr_data:
        _require_key(cr_data, "template", context="canned_response")
        canned = CannedResponseConfig(
            template=cr_data["template"],
            strict=cr_data.get("strict", True),
        )

    return StateConfig(
        name=data["name"],
        description=data.get("description", ""),
        chat_state=data.get("chat_state"),
        tool_instruction=data.get("tool_instruction"),
        guidelines=[_parse_guideline(g) for g in data.get("guidelines", [])],
        canned_response=canned,
    )


def _parse_transition(data: Dict[str, Any]) -> TransitionConfig:
    """将单条转移字典解析为 TransitionConfig。"""
    _require_key(data, "source", context="transitions 条目")
    _require_key(data, "target", context="transitions 条目")
    return TransitionConfig(
        source=data["source"],
        target=data["target"],
        condition=data.get("condition"),
    )


def _parse_guideline(data: Dict[str, Any]) -> GuidelineConfig:
    """将单条规则字典解析为 GuidelineConfig。"""
    _require_key(data, "condition", context="guidelines 条目")
    _require_key(data, "action", context="guidelines 条目")
    return GuidelineConfig(
        condition=data["condition"],
        action=data["action"],
        priority=data.get("priority", 50),
    )


def _require_key(data: Dict[str, Any], key: str, context: str = "配置") -> None:
    """断言 data 中存在 key，否则抛出 ValueError。"""
    if key not in data or data[key] is None:
        raise ValueError(f"{context} 中缺少必填字段：'{key}'")
