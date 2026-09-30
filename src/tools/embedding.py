"""bge-m3 向量化客户端（混合检索的稠密侧）。

设计要点：
- OpenAI 兼容协议 POST {base}/v1/embeddings（校内网关 aigw/硅基流动/自建 TEI 均此协议），
  Bearer 密钥；bge-m3：多语、1024 维、8192 token 上下文；
- **绝不抛异常**：未配置/任一批失败返回 None，调用方退纯 BM25——混合检索是增强不是依赖，
  离线测试与 key 失效场景零影响；
- SQLite 向量缓存（model+文本哈希 → float32 blob，data/indexes/embeddings.db）：
  同一论文跨 run/跨问答轮次不重复计费；
- 值里带行内注释的 .env（如 BASE_URL=... # 说明）统一剥 # 后内容，实测踩过。
"""
from __future__ import annotations

import os
import sqlite3
import struct
import threading
from pathlib import Path

import httpx

from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import sha256_text, truncate

_LOCK = threading.Lock()
_DB: sqlite3.Connection | None = None
_client: httpx.Client | None = None


def _cfg(key: str, env: str, default: str = "") -> str:
    """配置读取：环境变量优先（剥行内注释），退 sources.yaml embedding 节。"""
    raw = os.environ.get(env, "").strip()
    if not raw:
        raw = str(get_settings().source("embedding", key, default) or default)
    return raw.split("#")[0].strip()


def _base_url() -> str:
    return _cfg("base_url", "EMBEDDING_BASE_URL").rstrip("/")


def _api_key() -> str:
    return _cfg("api_key", "EMBEDDING_API_KEY")


def _model() -> str:
    return _cfg("model", "EMBEDDING_MODEL", "bge-m3")


def embedding_available() -> bool:
    """向量侧是否可用（三要素齐备才启用；False 时调用方直接走纯 BM25）。"""
    return bool(_base_url() and _api_key() and _model())


# --------------------------------------------------------------------------
# SQLite 向量缓存
# --------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    global _DB
    with _LOCK:
        if _DB is None:
            path = Path(get_settings().indexes_dir) / "embeddings.db"
            path.parent.mkdir(parents=True, exist_ok=True)
            _DB = sqlite3.connect(str(path), check_same_thread=False)
            # WAL + busy_timeout：回填/研究写入与问答检索并发时 rollback journal 会锁冲突，
            # 裸 except 把锁错误吞成缓存 miss（评审确认）——WAL 让读写不互斥，超时兜底
            _DB.execute("PRAGMA journal_mode=WAL")
            _DB.execute("PRAGMA busy_timeout=5000")
            _DB.execute(
                "CREATE TABLE IF NOT EXISTS emb_cache (k TEXT PRIMARY KEY, v BLOB)"
            )
            _DB.commit()
        return _DB


def _cache_get(key: str) -> list[float] | None:
    try:
        row = _db().execute("SELECT v FROM emb_cache WHERE k = ?", (key,)).fetchone()
        if not row or not row[0]:
            return None
        n = len(row[0]) // 4
        return list(struct.unpack(f"{n}f", row[0]))
    except Exception:  # noqa: BLE001 —— 缓存故障不应影响检索
        return None


def _cache_put(key: str, vec: list[float]) -> None:
    try:
        _db().execute(
            "INSERT OR REPLACE INTO emb_cache (k, v) VALUES (?, ?)",
            (key, struct.pack(f"{len(vec)}f", *vec)),
        )
        _db().commit()
    except Exception:  # noqa: BLE001
        pass


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=60.0)
    return _client


def _endpoint() -> str:
    """embeddings 端点：base 带/不带 /v1 后缀都兼容（两种写法都常见，拼出 /v1/v1 会 404）。"""
    base = _base_url()
    return f"{base}/embeddings" if base.endswith("/v1") else f"{base}/v1/embeddings"


def _post_batch(texts: list[str]) -> list[list[float]]:
    """一批文本 → 向量（内部使用；协议/网络错误向上抛，由 embed_texts 统一退 None）。"""
    timeout = float(get_settings().source("embedding", "request_timeout_s", 60) or 60)
    resp = _get_client().post(
        _endpoint(),
        json={"model": _model(), "input": texts},
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or []
    if len(data) != len(texts):
        raise ValueError(f"embedding 返回条数不符：{len(data)} != {len(texts)}")
    vecs = [[float(x) for x in (item.get("embedding") or [])] for item in data]
    if any(not v for v in vecs):
        raise ValueError("embedding 返回了空向量")
    return vecs


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """批量向量化；不可用或任一批失败 → None（调用方退纯词法检索）。

    缓存命中的条目不发请求；文本超长截到 8000 字符（bge-m3 8192 token 上限的保守值）。
    """
    if not texts or not embedding_available():
        return None
    tracer = get_active_tracer()
    todo = [i for i, t in enumerate(texts) if t and t.strip()]
    if not todo:
        return [[] for _ in texts]
    out: list[list[float] | None] = [None] * len(texts)
    keys = {i: sha256_text(f"{_model()}\x00{texts[i]}") for i in todo}
    for i in todo:
        cached = _cache_get(keys[i])
        if cached is not None:
            out[i] = cached
    missing = [i for i in todo if out[i] is None]
    batch = int(get_settings().source("embedding", "batch_size", 32) or 32)
    for start in range(0, len(missing), batch):
        ids = missing[start : start + batch]
        try:
            vecs = _post_batch([texts[i][:8000] for i in ids])
        except Exception as e:  # noqa: BLE001 —— 半向量半词法的混合排序不可信，整体退词法
            tracer.event("tool_error", tool="embed_texts", error=f"{type(e).__name__}: {truncate(str(e), 160)}")
            return None
        for i, vec in zip(ids, vecs):
            out[i] = vec
            _cache_put(keys[i], vec)
    return [v if v is not None else [] for v in out]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似（向量已归一化时即点积；bge-m3 默认归一化输出）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)
