"""
05_test_generator.py — journey.generator 单元测试

通过 mock AsyncOpenAI 客户端模拟 LLM 响应，无需真实 LLM 服务即可运行。
覆盖：LLMConfig 构建、JSON 提取、序列化、JourneyGenerator 主流程、CLI 参数解析。
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from journey.generator import (
    JourneyGenerator,
    LLMConfig,
    _extract_json,
    _serialize,
    generate_journey_config,
)
from journey.models import JourneyConfig


# ---------------------------------------------------------------------------
# 测试用 LLM 响应数据
# ---------------------------------------------------------------------------

# 标准的、LLM 正确输出 JSON 的场景
_VALID_LLM_JSON = {
    "title": "贷款申请流程",
    "description": "引导用户完成贷款申请的全流程",
    "triggers": ["我想申请贷款", "贷款咨询"],
    "states": [
        {"name": "collect_info",  "description": "收集资料", "chat_state": "请提供您的基本信息"},
        {"name": "credit_review", "description": "信用审核", "chat_state": "正在审核您的信用"},
        {"name": "closing",       "description": "流程收尾", "chat_state": "感谢您的申请"},
    ],
    "transitions": [
        {"source": "collect_info",  "target": "credit_review"},
        {"source": "credit_review", "target": "closing", "condition": "审核通过"},
    ],
    "global_guidelines": [
        {"condition": "用户询问利率", "action": "提供最新利率表", "priority": 70}
    ],
}

_VALID_LLM_RESPONSE = json.dumps(_VALID_LLM_JSON, ensure_ascii=False)


def _mock_openai_client(response_text: str) -> MagicMock:
    """构造一个模拟 AsyncOpenAI 客户端，使 chat.completions.create 返回指定文本。"""
    choice = MagicMock()
    choice.message.content = response_text

    completion = MagicMock()
    completion.choices = [choice]

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


# ---------------------------------------------------------------------------
# LLMConfig 构建
# ---------------------------------------------------------------------------

class TestLLMConfig构建:
    def test_直接构造(self):
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini")
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.base_url is None
        assert cfg.temperature == 0.0

    def test_完整参数(self):
        cfg = LLMConfig(
            api_key="sk-xxx",
            base_url="http://localhost:8000/v1",
            model="Qwen2.5-7B-Instruct",
            temperature=0.1,
            max_tokens=2048,
        )
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.max_tokens == 2048

    def test_from_env读取环境变量(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_BASE_URL", "http://my-server/v1")

        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-from-env"
        assert cfg.model == "gpt-4o"
        assert cfg.base_url == "http://my-server/v1"

    def test_from_env兼容QWEN3前缀(self, monkeypatch):
        """验证兼容 third_party/.env 中的 QWEN3_* 命名。"""
        monkeypatch.setenv("QWEN3_API_KEY", "sk-qwen3")
        monkeypatch.setenv("QWEN3_MODEL", "Qwen3-7B")
        # 清除 LLM_* 以确保 QWEN3_* 生效
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-qwen3"
        assert cfg.model == "Qwen3-7B"

    def test_from_env缺少API_KEY时抛出ValueError(self, monkeypatch):
        # 清除所有已知 key 的环境变量
        for key in ["LLM_API_KEY", "QWEN3_API_KEY", "OPENAI_API_KEY",
                    "DEEPSEEK_API_KEY", "QWEN_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        with pytest.raises(ValueError, match="API Key"):
            LLMConfig.from_env()

    def test_from_env_temperature被正确解析(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
        cfg = LLMConfig.from_env()
        assert cfg.temperature == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _extract_json — JSON 提取
# ---------------------------------------------------------------------------

class TestExtractJSON:
    def test_纯JSON被正确解析(self):
        text = json.dumps(_VALID_LLM_JSON, ensure_ascii=False)
        result = _extract_json(text)
        assert result["title"] == "贷款申请流程"

    def test_markdown代码块被自动剥除(self):
        text = f"```json\n{json.dumps(_VALID_LLM_JSON)}\n```"
        result = _extract_json(text)
        assert result["title"] == "贷款申请流程"

    def test_前后有多余文字时仍能提取(self):
        """LLM 偶尔会在 JSON 前后附加说明文字。"""
        text = f"以下是生成的配置：\n{json.dumps(_VALID_LLM_JSON)}\n请根据需要调整。"
        result = _extract_json(text)
        assert result["title"] == "贷款申请流程"

    def test_无JSON时抛出ValueError(self):
        with pytest.raises(ValueError, match="JSON"):
            _extract_json("这里没有任何 JSON 内容，只有纯文本。")

    def test_格式错误时抛出ValueError(self):
        with pytest.raises(ValueError):
            _extract_json("{invalid json}")


# ---------------------------------------------------------------------------
# _serialize — 序列化
# ---------------------------------------------------------------------------

class TestSerialize:
    def test_JSON序列化(self):
        data = {"title": "测试", "states": []}
        result = _serialize(data, "json")
        parsed = json.loads(result)
        assert parsed["title"] == "测试"

    def test_JSON保留中文字符(self):
        data = {"title": "中文标题", "states": []}
        result = _serialize(data, "json")
        assert "中文标题" in result  # 不应被转义为 \uXXXX

    def test_注释字段被过滤(self):
        """_comment 字段不应出现在序列化结果中。"""
        data = {"_comment": "注释", "title": "T", "states": []}
        result = _serialize(data, "json")
        parsed = json.loads(result)
        assert "_comment" not in parsed
        assert "title" in parsed

    def test_YAML序列化(self):
        pytest.importorskip("yaml", reason="需要 pyyaml")
        data = {"title": "测试", "states": []}
        result = _serialize(data, "yaml")
        assert "测试" in result

    def test_YAML未安装时抛出ImportError(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyyaml"):
            _serialize({"title": "T", "states": []}, "yaml")


# ---------------------------------------------------------------------------
# JourneyGenerator — 核心生成流程
# ---------------------------------------------------------------------------

class TestJourneyGenerator:
    def _make_generator(self, response_text: str = _VALID_LLM_RESPONSE) -> JourneyGenerator:
        """构造一个使用 mock 客户端的 JourneyGenerator。"""
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini")
        gen = JourneyGenerator.__new__(JourneyGenerator)
        gen._config = cfg
        gen._client = _mock_openai_client(response_text)
        return gen

    @pytest.mark.asyncio
    async def test_generate返回JourneyConfig和文本(self):
        gen = self._make_generator()
        config, text = await gen.generate("贷款申请流程")

        assert isinstance(config, JourneyConfig)
        assert config.title == "贷款申请流程"
        assert len(config.states) == 3
        assert len(config.transitions) == 2

    @pytest.mark.asyncio
    async def test_generate_json格式(self):
        gen = self._make_generator()
        config, text = await gen.generate("贷款申请流程", fmt="json")

        parsed = json.loads(text)
        assert parsed["title"] == "贷款申请流程"

    @pytest.mark.asyncio
    async def test_generate_yaml格式(self):
        pytest.importorskip("yaml", reason="需要 pyyaml")
        gen = self._make_generator()
        config, text = await gen.generate("贷款申请流程", fmt="yaml")

        import yaml
        parsed = yaml.safe_load(text)
        assert parsed["title"] == "贷款申请流程"

    @pytest.mark.asyncio
    async def test_generate携带extra_instruction(self):
        """extra_instruction 应被追加到 system prompt 中（通过 LLM 调用参数验证）。"""
        gen = self._make_generator()
        await gen.generate("流程", extra_instruction="状态不超过3个")

        call_args = gen._client.chat.completions.create.call_args
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "状态不超过3个" in system_msg

    @pytest.mark.asyncio
    async def test_generate_to_file写入文件(self, tmp_path):
        gen = self._make_generator()
        output = str(tmp_path / "out.json")
        config = await gen.generate_to_file("贷款申请流程", output)

        assert os.path.exists(output)
        with open(output, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["title"] == "贷款申请流程"
        assert isinstance(config, JourneyConfig)

    @pytest.mark.asyncio
    async def test_generate_to_file_yaml扩展名写yaml(self, tmp_path):
        pytest.importorskip("yaml", reason="需要 pyyaml")
        gen = self._make_generator()
        output = str(tmp_path / "out.yaml")
        await gen.generate_to_file("贷款申请流程", output)

        assert os.path.exists(output)
        import yaml
        with open(output, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["title"] == "贷款申请流程"

    def test_save写入文件(self, tmp_path):
        gen = self._make_generator()
        path = str(tmp_path / "config.json")
        gen.save(path, '{"title": "test"}')

        with open(path, encoding="utf-8") as fh:
            assert '"title"' in fh.read()

    @pytest.mark.asyncio
    async def test_llm返回非法JSON时抛出ValueError(self):
        gen = self._make_generator(response_text="这不是 JSON 内容，无法解析。")
        with pytest.raises(ValueError, match="JSON"):
            await gen.generate("任意输入")

    @pytest.mark.asyncio
    async def test_llm连接失败时抛出RuntimeError(self):
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini")
        gen = JourneyGenerator.__new__(JourneyGenerator)
        gen._config = cfg
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=Exception("connection refused to localhost:9999")
        )
        gen._client = client

        with pytest.raises(RuntimeError, match="LLM 服务"):
            await gen.generate("任意输入")

    @pytest.mark.asyncio
    async def test_llm其他异常直接向上传递(self):
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini")
        gen = JourneyGenerator.__new__(JourneyGenerator)
        gen._config = cfg
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=ValueError("模型不存在")
        )
        gen._client = client

        with pytest.raises(ValueError, match="模型不存在"):
            await gen.generate("任意输入")


# ---------------------------------------------------------------------------
# generate_journey_config — 便捷函数
# ---------------------------------------------------------------------------

class TestGenerateJourneyConfig便捷函数:
    @pytest.mark.asyncio
    async def test_返回config和文本(self, monkeypatch):
        """通过 patch JourneyGenerator 验证 generate_journey_config 的委托行为。"""
        import journey.generator as gen_mod

        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(
            return_value=(MagicMock(spec=JourneyConfig), '{"title": "T"}')
        )
        mock_gen.save = MagicMock()

        with patch.object(gen_mod, "JourneyGenerator", return_value=mock_gen):
            config, text = await generate_journey_config(
                "贷款申请", fmt="json",
                llm_config=LLMConfig(api_key="sk-x")
            )

        mock_gen.generate.assert_awaited_once()
        assert text == '{"title": "T"}'

    @pytest.mark.asyncio
    async def test_指定output_path时保存文件(self, tmp_path, monkeypatch):
        import journey.generator as gen_mod

        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(
            return_value=(MagicMock(spec=JourneyConfig), '{"title": "T"}')
        )
        mock_gen.save = MagicMock()

        out = str(tmp_path / "out.json")
        with patch.object(gen_mod, "JourneyGenerator", return_value=mock_gen):
            await generate_journey_config(
                "贷款申请", output_path=out,
                llm_config=LLMConfig(api_key="sk-x")
            )

        mock_gen.save.assert_called_once_with(out, '{"title": "T"}')


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

class TestCLI参数解析:
    def test_位置参数被正确解析(self):
        from journey.generator import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["贷款申请流程"])
        assert args.description == "贷款申请流程"

    def test_output参数被正确解析(self):
        from journey.generator import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["desc", "--output", "out.yaml"])
        assert args.output == "out.yaml"

    def test_format参数被正确解析(self):
        from journey.generator import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["desc", "--format", "json"])
        assert args.format == "json"

    def test_extra参数被正确解析(self):
        from journey.generator import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["desc", "--extra", "状态不超过5个"])
        assert args.extra == "状态不超过5个"

    def test_model参数被正确解析(self):
        from journey.generator import _build_arg_parser
        parser = _build_arg_parser()
        args = parser.parse_args(["desc", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"
