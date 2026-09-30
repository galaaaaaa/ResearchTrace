"""向量知识库（Milvus Lite + bge-m3）离线单测：真 milvus-lite 临时库 + 假向量。

不依赖网络：embedding.embed_texts 整体 monkeypatch（vector_store 懒导入模块属性，
monkeypatch 模块级函数即可生效）。
"""
from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("pymilvus", reason="kb extra 未安装（uv pip install '.[kb]'）——降级场景由其他测试覆盖")

from src.memory import vector_store as vs


def _kb_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """临时库路径 + 假 embedding（按内容关键词定向 1024 维向量）。"""
    from src.settings import get_settings
    from src.tools import embedding as emb

    monkeypatch.setattr(get_settings(), "indexes_dir", tmp_path)
    monkeypatch.setattr(vs, "_client", None)
    monkeypatch.setattr(vs, "_unavailable", False)
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://fake")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-fake")
    monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")

    def fake_embed(texts: list[str]):
        out = []
        for t in texts:
            v = [0.0] * 1024
            low = t.lower()
            if "grpo" in low:
                v[0] = 1.0
            if "weather" in low or "天气" in low:
                v[1] = 1.0
            if not any(v):
                seed = hashlib.sha256(t.encode()).digest()
                v[seed[0]] = 1.0
                v[seed[1]] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr(emb, "embed_texts", fake_embed)


def _ev(eid: str, paper: str, text: str, modality: str = "text", page: int | None = 3) -> dict:
    return {
        "evidence_id": eid, "paper_id": paper, "evidence_text": text,
        "claim_hint": f"hint-{eid}", "modality": modality, "page": page, "section": "S",
    }


def test_kb_roundtrip_and_semantic_order(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """upsert → 语义检索：相关证据排前；元数据过滤生效；upsert 幂等。"""
    _kb_env(monkeypatch, tmp_path)
    titles = {"p1": "GRPO 论文", "p2": "无关随笔"}
    evs = [
        _ev("e1", "p1", "GRPO training stability analysis"),
        _ev("e2", "p2", "the weather today is fine"),
        _ev("e3", "p1", "unrelated intro text here", modality="figure", page=7),
    ]
    assert vs.upsert_evidence(evs, run_id="run_x", paper_titles=titles) == 3
    assert vs.kb_count() == 3

    hits = vs.search_kb("GRPO 训练稳定性", k=3)
    assert hits and hits[0]["paper_id"] == "p1" and "GRPO" in hits[0]["evidence_text"]
    assert hits[0]["title"] == "GRPO 论文" and hits[0]["run_id"] == "run_x"
    # 元数据过滤：只留文本证据
    text_hits = vs.search_kb("GRPO", k=3, filter='modality == "text"')
    assert all(h["modality"] == "text" for h in text_hits)
    # 同键 upsert 幂等（不重复入库）
    assert vs.upsert_evidence(evs, run_id="run_x", paper_titles=titles) == 3
    assert vs.kb_count() == 3


def test_kb_page_none_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """page=None 存 -1，检索还原 None（web 在线阅读证据无页码场景）。"""
    _kb_env(monkeypatch, tmp_path)
    vs.upsert_evidence([_ev("e9", "p9", "GRPO doc without page", page=None)], run_id="r")
    hits = vs.search_kb("GRPO", k=1)
    assert hits and hits[0]["page_out"] is None


def test_kb_degrades_when_embedding_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """embedding 不可用 → available False、upsert 0、search []（绝不断链）。"""
    from src.settings import get_settings

    monkeypatch.setattr(get_settings(), "indexes_dir", tmp_path)
    monkeypatch.setattr(vs, "_client", None)
    monkeypatch.setattr(vs, "_unavailable", False)
    # autouse 夹具已剥 EMBEDDING_* → embedding_available False
    assert vs.kb_available() is False
    assert vs.upsert_evidence([_ev("e1", "p1", "x")], run_id="r") == 0
    assert vs.search_kb("q") == []


def test_kb_disabled_by_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """sources.yaml kb.enabled=false → 整体关闭。"""
    from src.settings import get_settings

    _kb_env(monkeypatch, tmp_path)
    monkeypatch.setitem(get_settings().sources.setdefault("kb", {}), "enabled", False)
    assert vs.kb_available() is False


def test_kb_ingestion_filter_blocks_failed_vlm(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """入库防线：自认失败的 VLM 解读不入库；桩 hint 不进嵌入文本（对抗评审确认项）。"""
    _kb_env(monkeypatch, tmp_path)
    seen_texts: list[str] = []
    from src.tools import embedding as emb

    real_embed = emb.embed_texts

    def spy_embed(texts):
        seen_texts.extend(texts)
        return real_embed(texts)

    monkeypatch.setattr(emb, "embed_texts", spy_embed)
    evs = [
        _ev("ok1", "p1", "GRPO training stability result: accuracy 82.4"),
        _ev("bad1", "p1", "[VLM 解读，须对照原文] 无法读取该图像的实际内容，图像仅以外链 URL 形式给出",
            modality="figure", page=5),
        dict(_ev("stub1", "p2", "GRPO reward hacking verbatim text"), claim_hint="（LLM 失败，自动保留的相关片段）"),
    ]
    n = vs.upsert_evidence(evs, run_id="r")
    assert n == 2  # bad1 被拦截
    assert any("无法读取" in t for t in seen_texts) is False  # 失败文案没进嵌入
    # 桩 hint 未编入向量文本（嵌入文本不含"（LLM 失败"）
    assert any("（LLM 失败" in t for t in seen_texts) is False
    hits = vs.search_kb("GRPO", k=5)
    assert all("无法读取" not in (h["evidence_text"] or "") for h in hits)


def _prime_env(monkeypatch: pytest.MonkeyPatch, tmp_path, hits):
    """prime_from_kb 测试环境：mock search_kb 返回构造命中 + 真 sqlite 记忆。"""
    from src.settings import get_settings

    monkeypatch.setattr(get_settings(), "db_path", tmp_path / "e.db")
    monkeypatch.setattr(vs, "search_kb", lambda q, **kw: hits)


def test_prime_from_kb_injects_evidence_and_paper(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """正常路径：命中证据重建（kb_prime 标记）+ 配套论文带 pdf_path 注入。"""
    from src.memory.evidence_store import EvidenceStore
    from src.schemas import PaperRecord
    from src.settings import get_settings

    db = tmp_path / "e.db"
    monkeypatch.setattr(get_settings(), "db_path", db)  # 必须先 mock 再开 store（曾误写生产库）
    store = EvidenceStore(db)
    pdf = tmp_path / "kb_paper.pdf"
    pdf.write_bytes(b"%PDF-cached")
    store.upsert_papers([PaperRecord(paper_id="arxiv:2305.18290", title="DPO",
                                     arxiv_id="2305.18290", pdf_path=str(pdf),
                                     source_url="https://arxiv.org/pdf/2305.18290")])
    store.close()

    hits = [
        {"score": 0.72, "paper_id": "arxiv:2305.18290", "title": "DPO", "run_id": "r1",
         "modality": "text", "page_out": 7, "claim_hint": "DPO 前沿更优", "evidence_text": "DPO yields the best reward-KL tradeoff"},
        # 跨 run 同段落（同 pid/page/前缀）→ 去重
        {"score": 0.70, "paper_id": "arxiv:2305.18290", "title": "DPO", "run_id": "r2",
         "modality": "text", "page_out": 7, "claim_hint": "重复", "evidence_text": "DPO yields the best reward-KL tradeoff"},
        # 低分 → 过滤
        {"score": 0.30, "paper_id": "arxiv:2305.18290", "title": "DPO", "run_id": "r1",
         "modality": "text", "page_out": 9, "claim_hint": "低分", "evidence_text": "low score text"},
        # sqlite 无此论文 → 证据跳过
        {"score": 0.80, "paper_id": "arxiv:9999.99999", "title": "Ghost", "run_id": "r1",
         "modality": "text", "page_out": 1, "claim_hint": "幽灵", "evidence_text": "ghost paper text"},
    ]
    monkeypatch.setattr(vs, "search_kb", lambda q, **kw: hits)

    from src.tools.pdf_tool import make_paper_id_from_pdf

    pid_eff = make_paper_id_from_pdf(str(pdf))  # 内容身份：与 planner 种子扫描同规则
    evs, papers = vs.prime_from_kb("DPO 效果对比")
    assert len(evs) == 1  # 去重 + 低分过滤 + 幽灵论文过滤后剩 1
    assert evs[0].paper_id == pid_eff and evs[0].page == 7
    assert evs[0].task_id == "kb_prime" and evs[0].extractor == "kb_prime"
    assert evs[0].evidence_text.startswith("DPO yields")
    assert len(papers) == 1 and papers[0].paper_id == pid_eff and papers[0].pdf_path == str(pdf)
    assert papers[0].arxiv_id == "2305.18290"  # 元数据保留（merge 时优于裸种子）


def test_prime_unifies_identity_with_local_seeds(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """pdf 在 papers_dir 内（planner 种子会扫同一文件）也照常注入——身份统一为
    sha256 内容哈希后，merge_papers 按 paper_id 自动去重合并，双身份不复存在。"""
    from src.memory.evidence_store import EvidenceStore
    from src.schemas import PaperRecord
    from src.settings import get_settings
    from src.tools.pdf_tool import make_paper_id_from_pdf

    seed_dir = tmp_path / "papers"
    seed_dir.mkdir()
    pdf = seed_dir / "arxiv_2305.18290.pdf"
    pdf.write_bytes(b"%PDF-seed")
    db = tmp_path / "e.db"
    monkeypatch.setattr(get_settings(), "db_path", db)
    store = EvidenceStore(db)
    store.upsert_papers([PaperRecord(paper_id="arxiv:2305.18290", title="DPO",
                                     arxiv_id="2305.18290", pdf_path=str(pdf))])
    store.close()
    monkeypatch.setattr(vs, "search_kb", lambda q, **kw: [
        {"score": 0.9, "paper_id": "arxiv:2305.18290", "title": "DPO", "run_id": "r1",
         "modality": "text", "page_out": 3, "claim_hint": "h", "evidence_text": "seed dup text"},
    ])
    evs, papers = vs.prime_from_kb("DPO")
    assert len(evs) == 1 and len(papers) == 1
    assert evs[0].paper_id == papers[0].paper_id == make_paper_id_from_pdf(str(pdf))


def test_prime_empty_when_kb_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vs, "search_kb", lambda q, **kw: [])
    assert vs.prime_from_kb("任何问题") == ([], [])
