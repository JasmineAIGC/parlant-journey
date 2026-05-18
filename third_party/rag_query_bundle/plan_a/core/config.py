"""
config.py — Plan A RAG 模块独立配置。

从环境变量读取所有运行时参数。
本文件不依赖其他 RAG 实现模块，仅使用标准库。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PlanAConfig:
    """
    Plan A RAG 系统配置。

    所有参数均可通过环境变量覆盖：

    ========================  =================================================
    字段                       读取的环境变量
    ========================  =================================================
    knowledge_base_path       KNOWLEDGE_BASE_PATH（或 KNOWLEDGE_PERSIST_DIR）
    vector_db_collection      VECTOR_DB_COLLECTION（默认 "plan_a"）
    top_k                     TOP_K（默认 5）
    ========================  =================================================
    """

    knowledge_base_path: str
    """FAISS 索引和文档库的持久化目录路径。"""

    vector_db_collection: str
    """向量数据库集合名（FAISS 模式下作为命名标识）。"""

    top_k: int
    """检索返回的最大文档片段数。"""

    @classmethod
    def from_env(cls) -> "PlanAConfig":
        """从环境变量加载配置。兼容 KNOWLEDGE_BASE_PATH 和 KNOWLEDGE_PERSIST_DIR 两种命名。"""
        kb_path = (
            os.environ.get("KNOWLEDGE_BASE_PATH")
            or os.environ.get(
                "KNOWLEDGE_PERSIST_DIR",
                "./knowledge_base/knowledge_persist_dir/hotpotqa",
            )
        )
        return cls(
            knowledge_base_path=kb_path,
            vector_db_collection=os.environ.get("VECTOR_DB_COLLECTION", "plan_a"),
            top_k=int(os.environ.get("TOP_K", "5")),
        )
