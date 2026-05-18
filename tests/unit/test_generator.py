"""
tests/unit/test_generator.py — journey.generator 单元测试

通过 mock LiteLLM（litellm.acompletion）模拟 LLM 响应，无需真实 LLM 服务即可运行。
覆盖：LLMConfig 构建（含 enable_thinking）、LiteLLM 工具函数、JSON 提取、
      序列化、LLMClient、JourneyGenerator 主流程、CLI 参数解析。
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from journey.generator import (
    DEFAULT_MODEL,
    JourneyGenerator,
    LLMClient,
    LLMConfig,
    _build_thinking_extra_body,
    _extract_json,
    _is_thinking_model,
    _needs_openai_prefix,
    _serialize,
    generate_journey_config,
)
from journey.models import JourneyConfig


# ---------------------------------------------------------------------------
# 测试用 LLM 响应数据
# ---------------------------------------------------------------------------

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


def _make_litellm_mock(response_text: str) -> MagicMock:
    """构造一个模拟 litellm 模块，使 acompletion 返回指定文本。"""
    choice = MagicMock()
    choice.message.content = response_text

    completion = MagicMock()
    completion.choices = [choice]

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=completion)
    return mock_litellm


# ---------------------------------------------------------------------------
# LiteLLM 工具函数
# ---------------------------------------------------------------------------

class TestIsThinkingModel:
    def test_qwen3被识别为推理模型(self):
        assert _is_thinking_model("Qwen3-7B") is True
        assert _is_thinking_model("qwen3-14b-instruct") is True

    def test_deepseek_r1被识别为推理模型(self):
        assert _is_thinking_model("deepseek-r1-distill-7b") is True

    def test_普通模型不是推理模型(self):
        assert _is_thinking_model("Qwen2.5-7B-Instruct") is False
        assert _is_thinking_model("gpt-4o") is False
        assert _is_thinking_model("gpt-4o-mini") is False


class TestNeedsOpenAIPrefix:
    def test_无前缀模型需要加前缀(self):
        assert _needs_openai_prefix("Qwen2.5-7B-Instruct") is True
        assert _needs_openai_prefix("gpt-4o-mini") is True

    def test_已有提供商前缀的模型不需要(self):
        assert _needs_openai_prefix("openai/gpt-4o") is False
        assert _needs_openai_prefix("anthropic/claude-3") is False
        assert _needs_openai_prefix("deepseek/deepseek-chat") is False


class TestBuildThinkingExtraBody:
    def test_None时返回空字典(self):
        assert _build_thinking_extra_body(None, None) == {}
        assert _build_thinking_extra_body(None, "http://localhost") == {}

    def test_有base_url时嵌套在chat_template_kwargs(self):
        result = _build_thinking_extra_body(False, "http://vllm:8080/v1")
        assert result == {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }

    def test_无base_url时放顶层(self):
        result = _build_thinking_extra_body(True, None)
        assert result == {"extra_body": {"enable_thinking": True}}


# ---------------------------------------------------------------------------
# LLMConfig 构建
# ---------------------------------------------------------------------------

class TestLLMConfig构建:
    def test_直接构造最小参数(self):
        cfg = LLMConfig(api_key="sk-test")
        assert cfg.api_key == "sk-test"
        assert cfg.model == DEFAULT_MODEL
        assert cfg.base_url is None
        assert cfg.temperature == 0.0
        assert cfg.enable_thinking is None

    def test_直接构造完整参数(self):
        cfg = LLMConfig(
            api_key="sk-xxx",
            model="Qwen2.5-7B-Instruct",
            base_url="http://localhost:8000/v1",
            temperature=0.1,
            max_tokens=2048,
            enable_thinking=False,
        )
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.max_tokens == 2048
        assert cfg.enable_thinking is False

    def test_from_env读取LLM前缀(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_BASE_URL", "http://my-server/v1")
        for k in ["QWEN3_API_KEY", "QWEN3_MODEL"]:
            monkeypatch.delenv(k, raising=False)

        with patch("journey.generator._load_dotenv"):
            cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-from-env"
        assert cfg.model == "gpt-4o"
        assert cfg.base_url == "http://my-server/v1"

    def test_from_env_QWEN3优先于LLM(self, monkeypatch):
        """QWEN3_* 优先级高于 LLM_*，与 third_party/.env 惯例一致。"""
        monkeypatch.setenv("QWEN3_API_KEY", "sk-qwen3")
        monkeypatch.setenv("QWEN3_MODEL", "Qwen3-7B")
        monkeypatch.setenv("LLM_API_KEY", "sk-llm-fallback")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")

        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-qwen3"
        assert cfg.model == "Qwen3-7B"

    def test_from_env兼容QWEN3前缀(self, monkeypatch):
        """验证兼容 third_party/.env 中的 QWEN3_* 命名（无 LLM_*）。"""
        monkeypatch.setenv("QWEN3_API_KEY", "sk-qwen3")
        monkeypatch.setenv("QWEN3_MODEL", "Qwen3-7B")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)

        cfg = LLMConfig.from_env()
        assert cfg.api_key == "sk-qwen3"
        assert cfg.model == "Qwen3-7B"

    def test_from_env缺少API_KEY时抛出ValueError(self, monkeypatch):
        for key in ["QWEN3_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY",
                    "DEEPSEEK_API_KEY", "QWEN_API_KEY"]:
            monkeypatch.delenv(key, raising=False)

        with patch("journey.generator._load_dotenv"):
            with pytest.raises(ValueError, match="API Key"):
                LLMConfig.from_env()

    def test_from_env_temperature被正确解析(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
        for k in ["QWEN3_API_KEY", "QWEN3_TEMPERATURE"]:
            monkeypatch.delenv(k, raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.temperature == pytest.approx(0.3)

    def test_from_env_enable_thinking_false(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        monkeypatch.setenv("LLM_ENABLE_THINKING", "false")
        for k in ["QWEN3_API_KEY"]:
            monkeypatch.delenv(k, raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.enable_thinking is False

    def test_from_env_enable_thinking_true(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        monkeypatch.setenv("LLM_ENABLE_THINKING", "true")
        for k in ["QWEN3_API_KEY"]:
            monkeypatch.delenv(k, raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.enable_thinking is True

    def test_from_env_enable_thinking_未设置时为None(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "sk-x")
        monkeypatch.delenv("LLM_ENABLE_THINKING", raising=False)
        monkeypatch.delenv("ENABLE_THINKING", raising=False)
        monkeypatch.delenv("QWEN3_API_KEY", raising=False)
        cfg = LLMConfig.from_env()
        assert cfg.enable_thinking is None


# ---------------------------------------------------------------------------
# LLMClient — 思考模式解析
# ---------------------------------------------------------------------------

class TestLLMClient思考模式:
    def _make_client(self, model: str, enable_thinking=None) -> LLMClient:
        cfg = LLMConfig(api_key="sk-test", model=model, enable_thinking=enable_thinking)
        client = LLMClient.__new__(LLMClient)
        client._config = cfg
        client._litellm = MagicMock()
        # 手动复制 __init__ 中的思考模式解析逻辑
        from journey.generator import _is_thinking_model
        resolved = enable_thinking
        if resolved is None and _is_thinking_model(model):
            resolved = False
        client._resolved_thinking = resolved
        return client

    def test_qwen3模型自动禁用思考(self):
        client = self._make_client("Qwen3-7B")
        assert client._resolved_thinking is False

    def test_普通模型thinking为None(self):
        client = self._make_client("Qwen2.5-7B-Instruct")
        assert client._resolved_thinking is None

    def test_强制启用思考(self):
        client = self._make_client("Qwen3-7B", enable_thinking=True)
        assert client._resolved_thinking is True

    @pytest.mark.asyncio
    async def test_chat_async剥除think块(self):
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini")
        client = LLMClient.__new__(LLMClient)
        client._config = cfg
        client._resolved_thinking = None

        raw = "<think>这是内部推理，不应出现在输出中</think>这是最终回复"
        client._litellm = _make_litellm_mock(raw)

        result = await client.chat_async([{"role": "user", "content": "你好"}])
        assert result == "这是最终回复"
        assert "<think>" not in result

    @pytest.mark.asyncio
    async def test_chat_async连接失败抛出RuntimeError(self):
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini",
                        base_url="http://localhost:9999/v1")
        client = LLMClient.__new__(LLMClient)
        client._config = cfg
        client._resolved_thinking = None

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=Exception("connection refused to localhost:9999")
        )
        client._litellm = mock_litellm

        with pytest.raises(RuntimeError, match="LLM 服务"):
            await client.chat_async([{"role": "user", "content": "测试"}])


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
        """构造一个使用 mock LLMClient 的 JourneyGenerator。"""
        cfg = LLMConfig(api_key="sk-test", model="gpt-4o-mini")
        gen = JourneyGenerator.__new__(JourneyGenerator)
        gen._config = cfg
        # mock _llm：chat_async 直接返回 response_text（已是清洁文本）
        mock_llm = MagicMock()
        mock_llm.chat_async = AsyncMock(return_value=response_text)
        gen._llm = mock_llm
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
        """extra_instruction 应被追加到 system prompt 中（通过 chat_async 调用参数验证）。"""
        gen = self._make_generator()
        await gen.generate("流程", extra_instruction="状态不超过3个")

        call_args = gen._llm.chat_async.call_args
        # chat_async 的第一个位置参数是 messages 列表
        messages = call_args.args[0]
        system_msg = messages[0]["content"]
        assert "状态不超过3个" in system_msg

    @pytest.mark.asyncio
    async def test_generate_to_file写入json文件(self, tmp_path):
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
    async def test_llm连接失败时RuntimeError向上传递(self):
        """LLMClient.chat_async 抛出 RuntimeError 应传递到 generate()。"""
        gen = self._make_generator()
        gen._llm.chat_async = AsyncMock(
            side_effect=RuntimeError("无法连接到 LLM 服务（http://localhost:9999/v1）")
        )

        with pytest.raises(RuntimeError, match="LLM 服务"):
            await gen.generate("任意输入")

    @pytest.mark.asyncio
    async def test_llm其他异常直接向上传递(self):
        gen = self._make_generator()
        gen._llm.chat_async = AsyncMock(side_effect=ValueError("模型不存在"))

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
