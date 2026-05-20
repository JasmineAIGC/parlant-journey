"""
vector_store.py — 向量数据库层（VDB）。

对应架构图中的 ``VDB[(向量数据库)]`` 模块。

职责：
  1. 加载预先构建好的 FAISS 索引和文档库（不负责索引构建）
  2. 接受查询向量，执行近邻搜索（FAISS）
  3. 返回 :class:`~plan_a.core.types.DocChunk` 列表

支持多种常见持久化格式：

- 自定义格式：``faiss_index`` / ``faiss_index.bin`` + ``document_store.json``
- LlamaIndex / FlexRAG：``faiss_index.bin`` + ``docstore.json`` + ``index_store.json``
- 当文档库文件缺失或为空时，自动尝试从同数据集的原始 JSON 文件回填内容

若目录不存在或索引文件缺失，抛出 ``FileNotFoundError``。

本文件不依赖其他 RAG 实现模块。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from plan_a.core import DocChunk
from utils.logging import get_logger


logger = get_logger("plan_a.vdb.vector_store")


class VectorStore:
    """
    本地 FAISS 向量库。

    加载由外部工具（如 FlexRAG）预先构建的 FAISS 索引，
    并提供同步的向量近邻搜索接口。

    Examples
    --------
    ::

        store = VectorStore()
        await store.initialize("./knowledge_base/hotpotqa")
        chunks = store.search(query_vector, top_k=5)
    """

    _INDEX_FILENAMES = ("faiss_index.bin", "faiss_index")
    _STORE_FILENAMES = ("document_store.json", "docstore.json")
    _INDEX_STORE_FILENAME = "index_store.json"

    def __init__(self) -> None:
        self._faiss_index = None
        self._documents: List[Dict[str, Any]] = []
        self._persist_dir: Optional[Path] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self, persist_dir: str) -> None:
        """
        加载 FAISS 索引和文档库（幂等，可多次调用）。

        Args:
            persist_dir: 知识库目录路径（支持 ``faiss_index.bin`` / ``faiss_index`` 及多种文档存储格式）。

        Raises:
            FileNotFoundError: 若目录或 FAISS 索引文件不存在。
            ImportError: 若未安装 ``faiss-cpu`` / ``faiss-gpu``。
        """
        if self._initialized:
            return

        try:
            import faiss as _faiss
        except ImportError as exc:
            raise ImportError(
                "VectorStore 需要安装 faiss：\n"
                "  pip install faiss-cpu   # CPU 版\n"
                "  pip install faiss-gpu   # GPU 版"
            ) from exc

        persist_path = Path(persist_dir)
        index_path = self._find_existing_file(persist_path, self._INDEX_FILENAMES)

        if not persist_path.exists():
            raise FileNotFoundError(
                f"知识库目录不存在：{persist_dir}\n"
                "请先运行知识库构建脚本生成索引。"
            )
        if index_path is None:
            raise FileNotFoundError(
                f"FAISS 索引文件不存在：{persist_path}\n"
                "请先运行知识库构建脚本生成索引。"
            )

        logger.info(f"[VectorStore] 加载 FAISS 索引：{index_path}")
        self._faiss_index = _faiss.read_index(str(index_path))
        logger.info(
            f"[VectorStore] 索引加载完成  "
            f"维度={self._faiss_index.d}  向量数={self._faiss_index.ntotal}"
        )

        self._documents = self._load_documents(persist_path)
        if self._documents:
            logger.info(f"[VectorStore] 文档库加载完成  共 {len(self._documents)} 条")
            if len(self._documents) != self._faiss_index.ntotal:
                logger.warning(
                    "[VectorStore] 文档数与向量数不一致  documents=%d  vectors=%d",
                    len(self._documents),
                    self._faiss_index.ntotal,
                )
        else:
            logger.warning("[VectorStore] 未找到可解析的文档内容，向量搜索将仅返回空结果")

        self._persist_dir = persist_path
        self._initialized = True

    # ------------------------------------------------------------------
    # 搜索接口
    # ------------------------------------------------------------------

    def search(self, query_vector: List[float], top_k: int) -> List[DocChunk]:
        """
        向量近邻搜索，返回最相关的文档片段。

        Args:
            query_vector: 查询向量（维度须与 FAISS 索引一致）。
            top_k:        最多返回的片段数。

        Returns:
            按相似度降序排列的 :class:`~plan_a.core.types.DocChunk` 列表。

        Raises:
            RuntimeError: 若 ``initialize()`` 尚未调用。
        """
        if not self._initialized or self._faiss_index is None:
            raise RuntimeError(
                "VectorStore 尚未初始化，请先调用 initialize(persist_dir)"
            )

        import numpy as np

        if len(query_vector) != self._faiss_index.d:
            raise ValueError(
                f"查询向量维度与 FAISS 索引不匹配：query={len(query_vector)} index={self._faiss_index.d}"
            )

        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self._faiss_index.search(query_array, top_k)

        chunks: List[DocChunk] = []

        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self._documents):
                continue

            doc = self._documents[int(idx)]

            # L2 距离 → 相似度分数，使用 1/(1+d) 公式保证结果在 (0, 1] 范围内
            # （L2 距离无界，直接用 1-d 会在距离 >1 时产生负数）
            score = float(1.0 / (1.0 + distances[0][i]))

            chunks.append(
                DocChunk(
                    content=doc.get("content", ""),
                    source=doc.get("source", "unknown"),
                    score=score,
                    metadata=doc.get("metadata", {}),
                )
            )

        # 1/(1+d) is always in (0, 1], so no negative-score filtering needed
        return chunks

    def _find_existing_file(
        self,
        base_dir: Path,
        filenames: Iterable[str],
    ) -> Optional[Path]:
        for name in filenames:
            path = base_dir / name
            if path.exists():
                return path
        return None

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)

    def _load_documents(self, persist_path: Path) -> List[Dict[str, Any]]:
        ordered_node_ids = self._load_index_order(persist_path)

        store_path = self._find_existing_file(persist_path, self._STORE_FILENAMES)
        if store_path is not None:
            docs = self._parse_store_file(store_path, ordered_node_ids)
            if docs:
                return docs

        fallback_docs = self._load_fallback_dataset_docs(persist_path)
        if fallback_docs:
            if ordered_node_ids and len(fallback_docs) >= len(ordered_node_ids):
                return fallback_docs[: len(ordered_node_ids)]
            return fallback_docs

        return []

    def _load_index_order(self, persist_path: Path) -> List[str]:
        index_store_path = persist_path / self._INDEX_STORE_FILENAME
        data = self._read_json(index_store_path)
        if not isinstance(data, dict):
            return []

        store_data = data.get("index_store/data")
        if not isinstance(store_data, dict) or not store_data:
            return []

        first_entry = next(iter(store_data.values()))
        payload = self._decode_payload(first_entry)
        nodes_dict = payload.get("nodes_dict") if isinstance(payload, dict) else None
        if not isinstance(nodes_dict, dict):
            return []

        def _sort_key(item: tuple[str, str]) -> tuple[int, str]:
            key = item[0]
            return (0, f"{int(key):012d}") if key.isdigit() else (1, key)

        return [node_id for _, node_id in sorted(nodes_dict.items(), key=_sort_key)]

    def _parse_store_file(self, store_path: Path, ordered_node_ids: List[str]) -> List[Dict[str, Any]]:
        data = self._read_json(store_path)
        if data is None:
            return []

        if isinstance(data, dict) and "docstore/data" in data:
            docs_map = data.get("docstore/data")
            if not isinstance(docs_map, dict):
                return []

            node_ids = ordered_node_ids or list(docs_map.keys())
            docs: List[Dict[str, Any]] = []
            for node_id in node_ids:
                payload = self._decode_payload(docs_map.get(node_id))
                doc = self._normalize_doc(payload, default_source=node_id)
                if doc is not None:
                    docs.append(doc)
            return docs

        if isinstance(data, dict):
            docs = [
                self._normalize_doc(value, default_source=str(key))
                for key, value in data.items()
            ]
            return [doc for doc in docs if doc is not None]

        if isinstance(data, list):
            docs = [self._normalize_doc(item) for item in data]
            return [doc for doc in docs if doc is not None]

        return []

    def _decode_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict) and "__data__" in payload:
            raw = payload["__data__"]
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return raw
            return raw
        return payload

    def _normalize_doc(
        self,
        payload: Any,
        default_source: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        payload = self._decode_payload(payload)
        if not isinstance(payload, dict):
            return None

        content = (
            payload.get("content")
            or payload.get("text")
            or payload.get("text_resource", {}).get("text")
        )
        if not isinstance(content, str) or not content.strip():
            return None

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        source = (
            payload.get("source")
            or metadata.get("file_name")
            or metadata.get("title")
            or metadata.get("source")
            or payload.get("id_")
            or default_source
        )

        return {
            "content": content.strip(),
            "source": source,
            "metadata": metadata,
        }

    def _load_fallback_dataset_docs(self, persist_path: Path) -> List[Dict[str, Any]]:
        knowledge_base_dir = persist_path.parent.parent
        dataset_name = persist_path.name
        candidates = [
            knowledge_base_dir / f"{dataset_name}.json",
            knowledge_base_dir / "knowledge_files_dir" / dataset_name / f"{dataset_name}_corpus.json",
        ]

        for path in candidates:
            data = self._read_json(path)
            if data is None:
                continue

            docs = self._normalize_dataset_docs(data, dataset_name)
            if docs:
                logger.info(f"[VectorStore] 使用原始数据文件回填文档内容：{path}")
                return docs

        return []

    def _normalize_dataset_docs(
        self,
        data: Any,
        dataset_name: str,
    ) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "content" in data[0]:
                docs = [self._normalize_doc(item) for item in data]
                return [doc for doc in docs if doc is not None]

            docs: List[Dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                for ctx in item.get("context", []):
                    if not isinstance(ctx, (list, tuple)) or len(ctx) != 2:
                        continue
                    title = ctx[0]
                    passages = ctx[1]
                    if not isinstance(title, str) or not isinstance(passages, list):
                        continue
                    content = " ".join(
                        p.strip() for p in passages if isinstance(p, str) and p.strip()
                    ).strip()
                    if not content:
                        continue
                    docs.append(
                        {
                            "content": content,
                            "source": title,
                            "metadata": {"title": title, "dataset": dataset_name},
                        }
                    )
            return docs

        if isinstance(data, dict):
            for key in ("docs", "data", "items"):
                if key in data:
                    return self._normalize_dataset_docs(data[key], dataset_name)

        return []
