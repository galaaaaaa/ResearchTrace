"""Milvus Lite 向量知识库：bge-m3 稠密向量 + 元数据过滤的全库证据检索。

设计要点：
- 嵌入式 Milvus Lite（pymilvus[milvus_lite]，单文件 data/indexes/kb.db）零服务零运维；
  将来升级 Milvus Standalone 只需把 MilvusClient(uri) 换成服务器地址，集合 schema 不变；
- 证据的唯一键是 evidence_id（VARCHAR 主键 upsert）——同一 run 重跑/回填不会重复入库；
- 嵌入文本 = claim_hint + evidence_text（与 webapp 问答重排一致）；
- **绝不抛异常**：pymilvus 未安装 / embedding 未配置 / 任何 Milvus 错误 → 0 / []，
  调用方（finalize 写入、webapp 问答）永远有退路（BM25 / 跳过），知识库是增强不是依赖；
- 向量维度固定 1024（bge-m3 稠密输出）；page 为 None 时存 -1，检索时换回 None。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import truncate

_DIM = 1024  # bge-m3 稠密维度
_COLLECTION = "evidence_kb"

# 入库防线（对抗评审确认：9/9 幻觉期 136 条 VLM"图表证据"曾无过滤入库、可被 kb_mode
# 引用为真证据）——VLM 自认读不到内容的解读行不入库
_VLM_FAIL_MARKERS = ("无法读取", "无法访问", "无法确认", "不能读取", "无法直接读出", "无法从图中", "cannot read", "unable to")
# 桩 hint 前缀（与 webapp._STUB_HINT_PREFIXES 同源）：锅炉文案不进嵌入文本
_STUB_HINTS = ("（LLM 失败", "LLM 摘要失败", "[VLM 解读", "VLM 解读", "图表证据提示（", "表格证据提示（", "（本片段为")


def _kb_worthy(text: str) -> bool:
    """自认失败的 VLM 解读不入库（其 evidence_text 开头即自述读不到内容）。"""
    return not any(m in (text or "")[:200] for m in _VLM_FAIL_MARKERS)


def _embed_text(hint: str, text: str) -> str:
    """嵌入文本：桩 hint 只引入语义噪声（"图表证据提示（figure，第 3 页）"），剔除。"""
    h = (hint or "").strip()
    if not h or h.startswith(_STUB_HINTS):
        return text or ""
    return f"{h} {text}"

_lock = threading.Lock()
_client: Any = None
_unavailable = False  # 导入/建库失败后短路，避免每次调用重复报错


def kb_enabled() -> bool:
    """配置开关（sources.yaml kb.enabled，默认开——真正的可用性还取决于依赖与 embedding）。"""
    return bool(get_settings().source("kb", "enabled", True))


def kb_available() -> bool:
    """知识库是否可用：pymilvus 可导入 + 集合可建 + embedding 侧可用。"""
    global _unavailable
    if _unavailable or not kb_enabled():
        return False
    try:
        from src.tools import embedding as emb

        if not emb.embedding_available():
            return False
        _get_client()  # 建库探测（失败会置 _unavailable）
        return not _unavailable
    except Exception:  # noqa: BLE001 —— 知识库不可用不是错误
        return False


def _get_client() -> Any:
    """懒建 MilvusClient 与集合（线程安全；建库失败置 _unavailable 短路后续调用）。"""
    global _client, _unavailable
    if _unavailable:
        raise RuntimeError("向量知识库不可用（已短路）")
    with _lock:
        if _client is not None:
            return _client
        try:
            from pymilvus import DataType, MilvusClient

            path = str(get_settings().indexes_dir / "kb.db")
            client = MilvusClient(path)
            if not client.has_collection(_COLLECTION):
                schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field("evidence_id", DataType.VARCHAR, is_primary=True, max_length=64)
                schema.add_field("vec", DataType.FLOAT_VECTOR, dim=_DIM)
                schema.add_field("paper_id", DataType.VARCHAR, max_length=160)
                schema.add_field("title", DataType.VARCHAR, max_length=512)
                schema.add_field("run_id", DataType.VARCHAR, max_length=64)
                schema.add_field("modality", DataType.VARCHAR, max_length=16)
                schema.add_field("page", DataType.INT64)
                schema.add_field("section", DataType.VARCHAR, max_length=512)
                schema.add_field("claim_hint", DataType.VARCHAR, max_length=2048)
                schema.add_field("text", DataType.VARCHAR, max_length=8192)
                idx = client.prepare_index_params()
                idx.add_index(field_name="vec", index_type="AUTOINDEX", metric_type="COSINE")
                client.create_collection(_COLLECTION, schema=schema, index_params=idx)
            # 跨进程重开后集合处于 released 态（实测坑：insert 的进程可用、新进程 search 报
            # code=101 "call load() before search"）——统一在拿客户端时加载
            try:
                client.load_collection(_COLLECTION)
            except Exception:  # noqa: BLE001 —— 部分版本自动加载，失败留给 search 报
                pass
        except ImportError:
            # pymilvus 未装：永久性缺失，短路后续调用（此前该标志从未被置 True，注释撒谎——评审确认）
            _unavailable = True
            raise
        except Exception:
            # 暂态失败（如 kb.db 被其他进程 flock 独占——milvus-lite 是进程级锁，backfill
            # 脚本跑完即释放）：不短路，下次调用重试，长驻进程不会因一次碰撞永久失去 KB
            raise
        _client = client
        return _client


def _to_item(ev: dict, *, run_id: str, title: str) -> dict:
    """审计包里的证据 dict → Milvus 行（不含 vec）。"""
    return {
        "evidence_id": str(ev.get("evidence_id") or "")[:64],
        "paper_id": truncate(str(ev.get("paper_id") or ""), 158),
        "title": truncate(str(title or ""), 510),
        "run_id": truncate(str(run_id or ""), 62),
        "modality": truncate(str(ev.get("modality") or "text"), 15),
        "page": int(ev.get("page") or -1),
        "section": truncate(str(ev.get("section") or ""), 510),
        "claim_hint": truncate(str(ev.get("claim_hint") or ""), 2046),
        "text": truncate(str(ev.get("evidence_text") or ""), 8190),
    }


def upsert_evidence(evidence: list[dict], *, run_id: str, paper_titles: dict[str, str] | None = None) -> int:
    """批量向量化并 upsert 证据（返回成功条数；任何失败 0——半写不可信时整批放弃）。

    evidence 为审计包风格的 dict（evidence_id/paper_id/page/modality/claim_hint/evidence_text）。
    """
    tracer = get_active_tracer()
    if not evidence:
        return 0
    try:
        if not kb_available():
            return 0
        from src.tools import embedding as emb

        titles = paper_titles or {}
        items = [_to_item(ev, run_id=run_id, title=titles.get(str(ev.get("paper_id")), "")) for ev in evidence]
        items = [it for it in items if it["evidence_id"] and _kb_worthy(it["text"])]
        if not items:
            return 0
        vecs = emb.embed_texts([_embed_text(it["claim_hint"], it["text"]) for it in items])
        if not vecs or len(vecs) != len(items):
            tracer.event("tool_error", tool="kb_upsert", error="向量化失败或条数不符，整批跳过")
            return 0
        rows = [{**it, "vec": v} for it, v in zip(items, vecs) if v]
        if not rows:
            return 0
        client = _get_client()
        client.upsert(collection_name=_COLLECTION, data=rows)
        tracer.event("tool_call", tool="kb_upsert", run_id=run_id, n=len(rows))
        return len(rows)
    except Exception as e:  # noqa: BLE001 —— 知识库写入失败不影响主流程
        tracer.event("tool_error", tool="kb_upsert", error=f"{type(e).__name__}: {truncate(str(e), 200)}")
        return 0


def search_kb(query: str, *, k: int = 6, filter: str | None = None) -> list[dict]:
    """全库语义检索（返回 [{score,paper_id,title,run_id,modality,page,claim_hint,text}]，降序）。

    filter 是 Milvus 布尔表达式，如 'modality == "text"'；None 为全模态。
    """
    tracer = get_active_tracer()
    q = " ".join((query or "").split())
    if not q:
        return []
    try:
        if not kb_available():
            return []
        from src.tools import embedding as emb

        vecs = emb.embed_texts([q])
        if not vecs or not vecs[0]:
            return []
        client = _get_client()
        res = client.search(
            collection_name=_COLLECTION,
            data=[vecs[0]],
            limit=max(1, min(int(k), 32)),
            filter=filter,
            output_fields=["paper_id", "title", "run_id", "modality", "page", "section", "claim_hint", "text"],
        )
        hits: list[dict] = []
        for r in (res[0] if res else []):
            ent = r.get("entity") or {}
            hits.append(
                {
                    "score": float(r.get("distance") or 0.0),
                    "paper_id": ent.get("paper_id"),
                    "title": ent.get("title") or "",
                    "run_id": ent.get("run_id") or "",
                    "modality": ent.get("modality") or "text",
                    "page": ent.get("page") if ent.get("page") is not None else None,
                    "page_out": None if (ent.get("page") or 0) < 0 else ent.get("page"),
                    "claim_hint": ent.get("claim_hint") or "",
                    "evidence_text": ent.get("text") or "",
                }
            )
        tracer.event("tool_call", tool="kb_search", query=truncate(q, 80), n_results=len(hits))
        return hits
    except Exception as e:  # noqa: BLE001 —— 检索失败返回空，调用方退其他通道
        tracer.event("tool_error", tool="kb_search", error=f"{type(e).__name__}: {truncate(str(e), 200)}")
        return []


def kb_count() -> int:
    """库内证据条数（不可用返回 -1）。"""
    try:
        if not kb_available():
            return -1
        return int(_get_client().get_collection_stats(_COLLECTION).get("row_count") or 0)
    except Exception:  # noqa: BLE001
        return -1


def prime_from_kb(query: str, *, k: int = 12, min_score: float = 0.5) -> tuple[list, list]:
    """新研究 run 的知识库预热：按用户问题检索历史证据，配套论文一起注入 state。

    - **身份统一**：论文按 PDF 内容哈希重建 paper_id（sha256:xxx，与 planner 的
      scan_local_pdfs 同规则）——KB 里的 arxiv:/DOI 身份与种子扫描的 sha256 身份
      在 merge_papers（按 paper_id 去重）处自动合并为一条，元数据取先注入的
      prime 版（更全）；历史 run 的身份碎片化由此不进入新 run；
    - 证据重建为全新 EvidenceRecord（新 evidence_id，task_id="kb_prime" 标记来源），
      evidence_text 即当时的逐字抽取——Writer 的 [paper_id:页码] 引用契约照常成立；
    - 论文自带 pdf_path（不重复下载）；searcher 对 existing 去重 → Reader 不重读；
    - (paper_id, page, 文本前缀) 去重跨 run 重复段落；分数低于 min_score 不注入；
      PDF 已丢失（文件不在）的论文其证据一并跳过（无从对照原文）；
    - KB 不可用/无命中 → ([], [])，研究照常从零开始。
    """
    from src.schemas import EvidenceRecord
    from src.tools.pdf_tool import make_paper_id_from_pdf
    from src.utils import now_iso

    hits = [h for h in search_kb(query, k=k) if h.get("score", 0.0) >= min_score]
    if not hits:
        return [], []
    paper_records: dict = {}
    evidence: list = []
    seen: set[tuple] = set()
    for h in hits:
        pid = h.get("paper_id") or ""
        if not pid or pid not in paper_records:
            rec = _paper_from_store(pid)
            if rec is None or not rec.pdf_path or not Path(rec.pdf_path).exists():
                paper_records[pid] = None  # 无可对照 PDF 的论文（库被清/文件丢失）——跳过
            else:
                # 统一为内容身份：与 planner 种子扫描同一规则，merge 处自动去重合并
                pid_eff = make_paper_id_from_pdf(rec.pdf_path)
                paper_records[pid] = rec.model_copy(update={"paper_id": pid_eff})
        rec_out = paper_records.get(pid)
        if rec_out is None:
            continue
        key = (rec_out.paper_id, h.get("page_out"), (h.get("evidence_text") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        hint = (h.get("claim_hint") or "").strip()
        evidence.append(
            EvidenceRecord(
                paper_id=rec_out.paper_id,
                task_id="kb_prime",
                claim_hint=hint or "（知识库历史证据，须对照原文）",
                evidence_text=h.get("evidence_text") or "",
                page=h.get("page_out"),
                section=None,
                modality=h.get("modality") or "text",
                relevance_score=min(max(float(h.get("score") or 0.5), 0.0), 1.0),
                source_url=rec_out.source_url or "",
                extractor="kb_prime",
                created_at=now_iso(),
            )
        )
    papers = [p for p in paper_records.values() if p is not None]
    if not papers:
        return [], []
    return evidence, papers


def _paper_from_store(paper_id: str):
    """sqlite 记忆取 PaperRecord；不可用/未找到返回 None。"""
    try:
        from src.memory.evidence_store import EvidenceStore

        store = EvidenceStore(get_settings().db_path)
        try:
            return store.get_paper(paper_id)
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        return None
