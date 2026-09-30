"""引用工具：撤稿核验、参考文献渲染、题名匹配。

设计要点：
- check_retraction 复用 crossref_tool 的请求与 relation 推断逻辑，绝不抛异常；
- render_references 纯字符串拼接：作者前 3 + et al.，doi/url/paper_id 逐级降级；
- match_paper：先 norm_title 精确相等，再 rapidfuzz token_set_ratio（忽略大小写）模糊匹配。
"""
from __future__ import annotations

from urllib.parse import quote

from rapidfuzz import fuzz

from src.schemas import PaperRecord, RetractionStatus
from src.settings import get_settings
from src.tools import crossref_tool
from src.tracing import get_active_tracer
from src.utils import norm_doi, norm_title


def check_retraction(paper: PaperRecord) -> RetractionStatus:
    """Crossref update-to 关系推断撤稿；is_retracted=None 表示无法判定；不抛异常。

    说明：schema 的 RetractionStatus 无 note 字段，状态说明统一写入 notice。
    """
    tracer = get_active_tracer()
    doi = crossref_tool._extract_doi(paper)
    if not doi:
        tracer.event("tool_call", tool="check_retraction", paper_id=paper.paper_id,
                     is_retracted=None, note="no doi")
        return RetractionStatus(paper_id=paper.paper_id, is_retracted=None, notice="no doi")
    if not get_settings().source("crossref", "enabled", True):
        return RetractionStatus(paper_id=paper.paper_id, is_retracted=None, notice="crossref disabled")
    try:
        resp = crossref_tool._http_get(f"/works/{quote(doi, safe='')}",
                                       params=crossref_tool._mailto_params())
        item = (resp.json() or {}).get("message") or {}
        is_retracted, _corrected_by, notice = crossref_tool._analyze_relations(item)
        if is_retracted is not None:
            paper.is_retracted = is_retracted  # 顺手回写，下游直接可用
        tracer.event("tool_call", tool="check_retraction", paper_id=paper.paper_id,
                     doi=doi, is_retracted=is_retracted)
        return RetractionStatus(paper_id=paper.paper_id, is_retracted=is_retracted,
                                notice=notice, source="crossref")
    except Exception as e:
        tracer.event("tool_error", tool="check_retraction", paper_id=paper.paper_id,
                     error=f"{type(e).__name__}: {e}"[:300])
        return RetractionStatus(paper_id=paper.paper_id, is_retracted=None, notice="lookup failed")


def _format_authors(authors: list[str]) -> str:
    """作者段：≤3 全列；>3 列前 3 + et al.；无作者返回空串。"""
    names = [a.strip() for a in (authors or []) if a and a.strip()]
    if not names:
        return ""
    if len(names) <= 3:
        return ", ".join(names)
    return ", ".join(names[:3]) + ", et al."


def render_references(papers: list[PaperRecord], *, style: str = "default") -> str:
    """编号参考文献列表：作者(前3+et al.). 题名. venue, year. doi/url；缺字段优雅降级。

    style 目前仅实现 "default"（GBM 风格编号列表），其余取值按 default 处理。
    """
    tracer = get_active_tracer()
    try:
        lines: list[str] = []
        for i, p in enumerate(papers or [], 1):
            seg: list[str] = []
            authors = _format_authors(p.authors)
            if authors:
                seg.append(authors)
            title = (p.title or "").strip()
            if title:
                seg.append(title)
            tail: list[str] = []
            if p.venue:
                tail.append(p.venue)
            if p.year is not None:
                tail.append(str(p.year))
            if tail:
                seg.append(", ".join(tail))
            # 定位信息逐级降级：doi > source_url > paper_id
            if p.doi:
                seg.append(f"doi:{p.doi}")
            elif (p.source_url or "").strip():
                seg.append(p.source_url.strip())
            elif p.paper_id:
                seg.append(p.paper_id)
            # 段内句点统一交给连接符（避免 "et al.." 双句点），行尾不再补句点
            lines.append(f"[{i}] " + ". ".join(s.rstrip(".") for s in seg if s))
        return "\n".join(lines)
    except Exception as e:
        tracer.event("tool_error", tool="render_references", error=f"{type(e).__name__}: {e}"[:300])
        return ""


def match_paper(title: str, papers: list[PaperRecord], *, threshold: float = 0.9) -> PaperRecord | None:
    """rapidfuzz 题名匹配去重（norm_title 基础上 token_set_ratio/100）。

    先 norm_title 精确相等（命中即返回），再忽略大小写做 token_set_ratio/100，
    返回得分最高且 ≥ threshold 的一条；无匹配返回 None。
    """
    tracer = get_active_tracer()
    try:
        q = (title or "").strip()
        if not q:
            return None
        nq = norm_title(q)
        best: PaperRecord | None = None
        best_score = 0.0
        for p in papers or []:
            p_title = (p.title or "").strip() if p is not None else ""
            if not p_title:
                continue
            if nq and norm_title(p_title) == nq:
                return p  # 精确命中优先于一切模糊结果
            score = fuzz.token_set_ratio(q.lower(), p_title.lower()) / 100.0
            if score > best_score:
                best, best_score = p, score
        return best if best_score >= threshold else None
    except Exception as e:
        tracer.event("tool_error", tool="match_paper", error=f"{type(e).__name__}: {e}"[:300])
        return None
