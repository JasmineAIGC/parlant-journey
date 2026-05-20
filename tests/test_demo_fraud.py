from __future__ import annotations

import os
import sys
from typing import List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

try:
    from tools.llm import LLMClient, LLMConfig
    from tools.parlant_server import ParlantServerContext
    from tools.parlant_server.session import run_conversation
    from journey import create_journey
    from demos.fraud.demo_fraud import DEMO_SCRIPT, fraud_config
    _PARLANT_AVAILABLE = True
except ImportError:
    _PARLANT_AVAILABLE = False


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _PARLANT_AVAILABLE,
    reason="parlant 未安装（pip install parlant）",
)
async def test_fraud_demo_runs_full_conversation() -> None:
    if _PARLANT_AVAILABLE:
        try:
            await LLMClient(config=LLMConfig.from_env()).chat_async([{"role": "user", "content": "你好"}])
        except Exception as exc:
            pytest.skip(f"LLM 服务不可达：{exc}")

    replies: List[str] = []
    try:
        async with ParlantServerContext() as ctx:
            agent = await ctx.create_agent("银行客服", "专业银行客服 Agent")
            await create_journey(agent, fraud_config)
            async for _user_msg, reply in run_conversation(ctx, agent, DEMO_SCRIPT[:2]):
                replies.append(reply)
    except RuntimeError as exc:
        pytest.skip(f"Parlant Server 启动失败：{exc}")

    assert replies, "至少应收到 1 条 Agent 回复"
