# rag_query_bundle — Parlant RAG 单条查询

本目录是**自包含**的独立运行包，无需安装整个仓库。

---

## 示例运行命令

```bash
cd rag_query_bundle

python run_rag_query.py \
  --query "What is known for one of the stars of The Newcomers?" \
  --output-format json \
  --output-file results/rag_output.json \
  --top-k 3
```

---

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--query` | str | 内置示例问题 | 检索查询文本 |
| `--top-k` | int | None（使用 pipeline 默认值） | 最多返回的文档片段数 |
| `--include-metadata` | flag | False | 在 `chunks` 中附加每个片段的原始元数据 |
| `--answer` | flag | False | 后处理阶段调用 LLM 生成回答（默认关闭） |
| `--output-format` | text \| json | text | 终端输出格式 |
| `--output-file` | PATH | None | 保存 JSON 结果的文件路径；指定时自动使用 JSON 格式 |

---

## 输出结果

### JSON 格式（`--output-format json` 或 `--output-file`）

```json
{
  "query": "What is known for one of the stars of The Newcomers?",
  "retrieval": {
    "top_k": 3,
    "chunk_count": 3,
    "took_ms": 42,
    "chunks": [
      {
        "content": "...",
        "source": "doc_name.txt",
        "score": 0.9123
      }
    ]
  },
  "answer": "",
  "timing": {
    "init_ms": 28500,
    "rag_ms": 85,
    "postprocess_ms": 0,
    "total_ms": 28585
  }
}
```

### 输出字段说明

#### 顶层字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 用户输入的查询文本（即 `--query` 参数值） |
| `retrieval` | object | 向量检索结果 |
| `answer` | string | LLM 生成的回答（`--answer` 时有内容，否则为空字符串） |
| `timing` | object | 各阶段耗时统计（毫秒） |

#### `retrieval` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `top_k` | int | 本次检索使用的片段上限 |
| `chunk_count` | int | 实际命中并返回的片段数（`<= top_k`） |
| `took_ms` | int | 向量检索耗时（毫秒，不含 LLM 调用） |
| `chunks` | array | 检索到的文档片段列表，按相似度降序排列 |

#### `chunks[i]` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `content` | string | 文档片段正文 |
| `source` | string | 片段来源标识（文件名或文档 ID） |
| `score` | float | 向量相似度分数（0～1，越高越相关） |
| `metadata` | object | 原始文档元数据，**仅在 `--include-metadata` 时出现** |

#### `timing` 对象

| 字段 | 类型 | 说明 |
|---|---|---|
| `init_ms` | int | 初始化耗时：Parlant Server 启动 + LiteLLM 初始化 + Embedding 缓存（**一次性**，约 20-30s） |
| `rag_ms` | int | RAG 检索耗时：向量检索，不含 LLM 调用（通常 < 100ms） |
| `postprocess_ms` | int | 后处理耗时：无 `--answer` 时 < 1ms；有 `--answer` 时含 LLM 调用时间 |
| `total_ms` | int | 总耗时（毫秒） |

---

## 配置（`.env`）

```bash
cp .env.example .env
# 编辑 .env，填写实际值
```

必填项：

| 变量 | 说明 |
|---|---|
| `QWEN3_API_KEY` | LLM API 密钥 |
| `QWEN3_BASE_URL` | LLM 服务地址（OpenAI 兼容端点） |
| `QWEN3_MODEL` | LLM 模型名称 |
| `EMBEDDING_BASE_URL` | Embedding 服务地址 |
| `EMBEDDING_API_KEY` | Embedding API 密钥 |
| `EMBEDDING_MODEL` | Embedding 模型名称 |
| `KNOWLEDGE_PERSIST_DIR` | FAISS 索引目录路径（含 `faiss_index.bin` 和 `docstore.json`） |

---

完整架构说明见 `docs/planA.md`。

