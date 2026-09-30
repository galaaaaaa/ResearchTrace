"""通用工具：ID、哈希、规范化、JSON 提取。"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_id(prefix: str = "id") -> str:
    """生成短随机 ID，如 ev_1f3a9c2b。"""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm_title(title: str) -> str:
    """规范化题名用于去重：小写、去除非字母数字。"""
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    for pre in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if d.startswith(pre):
            d = d[len(pre):]
    return d or None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def truncate(text: str, n: int, suffix: str = "…") -> str:
    text = text or ""
    return text if len(text) <= n else text[: max(0, n - len(suffix))] + suffix


def estimate_tokens(text: str) -> int:
    """粗略 token 估计（中英混合）：英文约 4 字符/token，CJK 约 1.5 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


def extract_json_block(text: str) -> Any | None:
    """从模型输出中防御性提取 JSON。

    依次尝试：1) ```json 围栏；2) 首个平衡的花/方括号块。
    返回解析后的对象，失败返回 None。
    """
    if not text:
        return None
    # 1) 围栏代码块
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass
    # 2) 平衡括号扫描
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break  # 该 opener 失败，尝试下一个
    # 3) 整体直接解析
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]
