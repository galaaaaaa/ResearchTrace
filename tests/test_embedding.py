"""embedding 客户端（bge-m3 混合检索稠密侧）离线单测。

零网络：monkeypatch _post_batch 返回假向量；sqlite 缓存指向 tmp_path。
"""
from __future__ import annotations

import sqlite3

import pytest

from src.tools import embedding as emb


def _fake_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> sqlite3.Connection:
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://fake-gateway")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-fake")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    db = sqlite3.connect(str(tmp_path / "emb.db"))
    db.execute("CREATE TABLE IF NOT EXISTS emb_cache (k TEXT PRIMARY KEY, v BLOB)")
    db.commit()
    monkeypatch.setattr(emb, "_DB", db)
    return db


def test_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """三要素不齐 → embedding_available False、embed_texts None（调用方退纯 BM25）。"""
    assert emb.embedding_available() is False  # autouse 夹具已剥 EMBEDDING_*
    assert emb.embed_texts(["x"]) is None


def test_batch_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """批量分批调用；缓存命中后重复请求不再发网络。"""
    _fake_config(monkeypatch, tmp_path)
    assert emb.embedding_available() is True
    from src.settings import get_settings

    monkeypatch.setitem(get_settings().sources.setdefault("embedding", {}), "batch_size", 2)
    calls: list[list[str]] = []

    def fake_post(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    monkeypatch.setattr(emb, "_post_batch", fake_post)

    v1 = emb.embed_texts(["aa", "bbb", "c"])
    assert len(calls) == 2  # batch_size=2 → 3 条文本分 2 批
    assert v1 == [[2.0, 1.0], [3.0, 1.0], [1.0, 1.0]]

    v2 = emb.embed_texts(["aa", "bbb", "c"])  # 全部缓存命中
    assert len(calls) == 2
    assert v2 == v1

    emb.embed_texts(["newcomer"])  # 未缓存 → 追加一批
    assert len(calls) == 3


def test_failure_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """任一批失败 → 整体 None（半向量半词法的混合排序不可信）。"""
    _fake_config(monkeypatch, tmp_path)

    def boom(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("401 Invalid token")

    monkeypatch.setattr(emb, "_post_batch", boom)
    assert emb.embed_texts(["a"]) is None


def test_env_inline_comment_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """.env 值尾的行内注释（# 说明）须剥掉，否则拼进 URL/key。"""
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://gw.example.com    # 校内网关")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-x")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    assert emb._base_url() == "http://gw.example.com"
    assert emb._endpoint() == "http://gw.example.com/v1/embeddings"


def test_endpoint_tolerates_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """base 带不带 /v1 后缀都要拼出正确端点（曾拼出 /v1/v1/embeddings 404）。"""
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://gw.example.com/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-x")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
    assert emb._endpoint() == "http://gw.example.com/v1/embeddings"


def test_cosine() -> None:
    assert emb.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert emb.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert emb.cosine([1.0], [1.0, 0.0]) == 0.0  # 维度不符 → 0


# ---------------------------------------------------------------------------
# pdf_tool 混合检索融合
# ---------------------------------------------------------------------------
class _Chunk:
    """pdf_tool.Chunk 的轻量桩（_fuse_with_dense 只读 page/section/text）。"""

    def __init__(self, text: str):
        self.page = 1
        self.section = None
        self.text = text


def test_fuse_with_dense_pulls_semantic_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    """语义近邻（零词法命中）经 RRF 进入 top-k——混合检索相对纯 BM25 的增量。"""
    from src.tools import pdf_tool

    a = _Chunk("reward hacking grpo")                       # 词法命中
    b = _Chunk("unrelated introduction text")               # 两者都不沾
    c = _Chunk("training instability and collapse in rl")   # 语义近邻、零词面重叠
    lex_hits = [(a, 3.0)]                                   # 纯 BM25 只命中 a（b/c 零词法分）

    monkeypatch.setattr(emb, "embedding_available", lambda: True)
    # 假向量：query 与 c 同向、与 a/b 正交
    monkeypatch.setattr(
        emb, "embed_texts",
        lambda texts: [[0.0, 1.0]] + [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )
    fused = pdf_tool._fuse_with_dense([a, b, c], lex_hits, "GRPO 训练稳定性", k=2)
    assert fused is not None
    picked = [ch for ch, _s in fused]
    # RRF：a=1/61+1/62（双路共识第一）> c=1/61（稠密第一）> b=1/63（两路皆末）
    assert picked[0] is a
    assert picked[1] is c            # 语义近邻进入 top-k——纯 BM25 永远不会返回它
    assert b not in picked


def test_fuse_with_dense_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """向量侧不可用 → None，retrieve_chunks 退纯 BM25（行为不变）。"""
    from src.tools import pdf_tool

    monkeypatch.setattr(emb, "embedding_available", lambda: False)
    a = _Chunk("x")
    assert pdf_tool._fuse_with_dense([a], [(a, 1.0)], "q", k=1) is None
