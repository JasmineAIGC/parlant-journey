from __future__ import annotations

import os
from typing import Iterable, Optional


def first_env(keys: Iterable[str], default: Optional[str] = None) -> Optional[str]:
    """按优先级读取环境变量，返回首个非空值。"""
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


def load_dotenv_candidates(
    *,
    start_path: str,
    max_parents: int = 5,
) -> Optional[str]:
    """尝试加载 .env / third_party/.env，返回命中的路径。"""
    try:
        from dotenv import load_dotenv  # type: ignore[import]
    except ImportError:
        return None

    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, ".env"),
        os.path.join(cwd, "third_party", ".env"),
    ]

    current = os.path.abspath(start_path)
    for _ in range(max_parents):
        current = os.path.dirname(current)
        candidates.extend([
            os.path.join(current, ".env"),
            os.path.join(current, "third_party", ".env"),
        ])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            load_dotenv(candidate)
            return candidate
    return None
