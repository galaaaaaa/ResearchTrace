"""Crossref 元数据查询与核验。

设计要点：
- lookup_crossref：DOI（含 / 且匹配 ^10\\.\\d{4,9}/）走 /works/{doi}，否则题名检索
  query.bibliographic + rapidfuzz token_set_ratio ≥ 0.75 才接受；
- verify_metadata：DOI 存在性 / 题名相似度 / 首作者姓 / 年份 ±1 一致性 / relation 撤稿推断；
  网络失败绝不抛异常，返回 note="lookup failed" 的空 MetadataCheck；
- 撤稿推断逻辑拆成 _analyze_relations，供 citation_tool.check_retraction 复用。
"""
from __future__ import annotations

import re
import threading
from urllib.parse import quote

import httpx
import tenacity
from rapidfuzz import fuzz

from src.schemas import MetadataCheck, PaperRecord
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import norm_doi

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+", re.IGNORECASE)
_TITLE_MATCH_THRESHOLD = 0.75
_RETRACT_KEYWORDS = ("retraction", "retract", "withdraw")

_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """模块级复用的 httpx.Client（线程安全，跟随重定向）。"""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                follow_redirects=True,
                headers={"User-Agent": "research-agent/0.1 (mailto: config in sources.yaml)"},
            )
        return _client


def _squash_ws(text: str | None) -> str:
    """压缩空白为单空格。"""
    return " ".join((text or "").split())


def _is_retryable(exc: BaseException) -> bool:
    """仅超时/连接错误与 429/5xx 重试。"""
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503}
    return False


_retry_policy = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


def _mailto_params() -> dict[str, str]:
    """Crossref polite pool 参数（配置了 mailto 才附带）。"""
    mailto = get_settings().source("crossref", "mailto", "")
    return {"mailto": str(mailto)} if mailto else {}


@_retry_policy
def _http_get(path: str, *, params: dict[str, str] | None = None) -> httpx.Response:
    """带重试的 GET（内部使用，公开函数统一兜底异常）。"""
    settings = get_settings()
    base = str(settings.source("crossref", "base_url", "https://api.crossref.org"))
    timeout = float(settings.source("crossref", "request_timeout_s", 25) or 25)
    resp = _get_client().get(f"{base}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def _title_similarity(a: str | None, b: str | None) -> float:
    """题名相似度：token_set_ratio/100，忽略大小写与多余空白。"""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0


def _extract_doi(paper: PaperRecord) -> str | None:
    """从 PaperRecord 提取可用 DOI（doi 字段优先，paper_id 形如 DOI 时兜底）。"""
    doi = norm_doi(paper.doi)
    if not doi and paper.paper_id and _DOI_RE.match(paper.paper_id.strip()):
        doi = norm_doi(paper.paper_id)
    return doi


def _work_year(item: dict) -> int | None:
    """issued.date-parts[0][0] → 年份。"""
    try:
        parts = (item.get("issued") or {}).get("date-parts") or []
        if parts and isinstance(parts[0], list) and parts[0]:
            return int(parts[0][0])
    except (TypeError, ValueError):
        pass
    return None


def _work_title(item: dict) -> str:
    """Crossref title（列表）→ 单字符串。"""
    return _squash_ws(" ".join(str(t) for t in (item.get("title") or [])))


def _work_to_paper(item: dict) -> PaperRecord | None:
    """Crossref work JSON → PaperRecord（纯函数，便于离线单测）。无 DOI 返回 None。"""
    if not isinstance(item, dict):
        return None
    doi = norm_doi(item.get("DOI"))
    if not doi:
        return None
    authors: list[str] = []
    for a in item.get("author") or []:
        if not isinstance(a, dict):
            continue
        name = _squash_ws(f"{a.get('given') or ''} {a.get('family') or ''}".strip())
        if name:
            authors.append(name)
    venue = None
    container = item.get("container-title") or []
    if container:
        venue = _squash_ws(str(container[0])) or None
    cc = item.get("is-referenced-by-count")
    citation_count = int(cc) if isinstance(cc, (int, float)) else None
    return PaperRecord(
        paper_id=doi,
        title=_work_title(item) or doi,
        authors=authors,
        year=_work_year(item),
        venue=venue,
        doi=doi,
        source_url=f"https://doi.org/{doi}",
        citation_count=citation_count,
        paper_type="full-paper",
        source_api="crossref",
    )


def _search_works_by_title(title: str) -> dict | None:
    """题名检索 crossref，返回相似度 ≥ 0.75 的最优 work JSON（无命中返回 None）。"""
    select = "DOI,title,author,issued,is-referenced-by-count,container-title"
    resp = _http_get("/works", params={"query.bibliographic": title, "rows": "3",
                                       "select": select, **_mailto_params()})
    items = (resp.json() or {}).get("message", {}).get("items") or []
    best_item: dict | None = None
    best_score = 0.0
    for it in items:
        if not isinstance(it, dict):
            continue
        score = _title_similarity(title, _work_title(it))
        if score > best_score:
            best_item, best_score = it, score
    if best_item is not None and best_score >= _TITLE_MATCH_THRESHOLD:
        return best_item
    return None


def _analyze_relations(item: dict) -> tuple[bool | None, list[str], str | None]:
    """从 work 的 relation 字段推断撤稿/更正。

    返回 (is_retracted, corrected_by, notice_title)：
    - 无 relation → (False, [], None)；
    - is-update-of 指向的更新记录 title/type/relation 键含 retraction|withdraw
      → (True, [], 撤稿声明题名)；
    - 更新记录只是勘误/更正 → (False, [更新 DOI], None)；
    - 更新记录全部拉取失败 → (None, [], None)（无法判定，不猜测）。
    """
    rel = item.get("relation") if isinstance(item.get("relation"), dict) else {}
    updates = [norm_doi(u.get("id")) for u in (rel.get("is-update-of") or [])
               if isinstance(u, dict) and u.get("id")]
    updates = [d for d in updates if d]
    if not updates:
        return False, [], None
    fetched_any = False
    corrected_by: list[str] = []
    for upd_doi in updates:
        try:
            resp = _http_get(f"/works/{quote(upd_doi, safe='')}", params=_mailto_params())
            upd = (resp.json() or {}).get("message") or {}
        except Exception:
            continue  # 单条更新拉取失败不阻断其余判断
        fetched_any = True
        blob = " ".join([
            _work_title(upd),
            str(upd.get("type") or ""),
            " ".join((upd.get("relation") or {}).keys()) if isinstance(upd.get("relation"), dict) else "",
        ]).lower()
        if any(k in blob for k in _RETRACT_KEYWORDS):
            notice = _work_title(upd) or upd_doi
            return True, [], notice
        corrected_by.append(upd_doi)
    if not fetched_any:
        return None, [], None
    return False, corrected_by, None


def lookup_crossref(title_or_doi: str) -> PaperRecord | None:
    """DOI 或题名查询；返回含 doi/venue/year/citation_count 的 PaperRecord；失败返回 None。"""
    tracer = get_active_tracer()
    raw = (title_or_doi or "").strip()
    if not raw:
        return None
    if not get_settings().source("crossref", "enabled", True):
        tracer.event("tool_error", tool="lookup_crossref", error="crossref disabled")
        return None
    try:
        doi = norm_doi(raw)
        if doi and "/" in doi and _DOI_RE.match(doi):
            resp = _http_get(f"/works/{quote(doi, safe='')}", params=_mailto_params())
            paper = _work_to_paper((resp.json() or {}).get("message") or {})
            tracer.event("tool_call", tool="lookup_crossref", mode="doi", doi=doi,
                         found=paper is not None)
            return paper
        # 题名检索：fuzz 过滤无关命中
        item = _search_works_by_title(raw)
        paper = _work_to_paper(item) if item else None
        tracer.event("tool_call", tool="lookup_crossref", mode="title", query=raw,
                     found=paper is not None, doi=paper.doi if paper else None)
        return paper
    except Exception as e:
        tracer.event("tool_error", tool="lookup_crossref",
                     error=f"{type(e).__name__}: {e}"[:300])
        return None


def verify_metadata(paper: PaperRecord) -> MetadataCheck:
    """DOI 存在性/题名相似度/作者年份一致性/撤稿推断；网络失败各字段为 None。"""
    tracer = get_active_tracer()
    if not get_settings().source("crossref", "enabled", True):
        return MetadataCheck(source="crossref", note="crossref disabled")
    try:
        doi = _extract_doi(paper)
        item: dict | None = None
        matched_by = "doi"
        if doi:
            try:
                resp = _http_get(f"/works/{quote(doi, safe='')}", params=_mailto_params())
                item = (resp.json() or {}).get("message") or {}
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    tracer.event("tool_call", tool="verify_metadata", paper_id=paper.paper_id,
                                 doi_valid=False, note="doi not found in crossref")
                    return MetadataCheck(source="crossref", doi_valid=False,
                                         note="doi not found in crossref")
                raise
        if item is None:
            # 无 DOI 的 paper：题名检索拿记录再比对
            matched_by = "title"
            if not (paper.title or "").strip():
                return MetadataCheck(source="crossref", note="not found in crossref")
            item = _search_works_by_title(paper.title)
            if item is None:
                tracer.event("tool_call", tool="verify_metadata", paper_id=paper.paper_id,
                             note="not found in crossref")
                return MetadataCheck(source="crossref", note="not found in crossref")

        cr_title = _work_title(item)
        cr_authors = [a for a in (item.get("author") or []) if isinstance(a, dict)]
        cr_year = _work_year(item)

        title_similarity = round(_title_similarity(paper.title, cr_title), 4) if cr_title else None

        authors_match: bool | None = None
        if paper.authors and cr_authors:
            family = _squash_ws(str(cr_authors[0].get("family") or "")).lower()
            first = _squash_ws(paper.authors[0])
            last = first.split()[-1].lower() if first else ""
            if family and last:
                authors_match = family == last

        year_match: bool | None = None
        if paper.year is not None and cr_year is not None:
            year_match = abs(int(paper.year) - cr_year) <= 1

        is_retracted, corrected_by, _notice = _analyze_relations(item)

        check = MetadataCheck(
            doi_valid=True if doi else None,  # 题名匹配时无 DOI 可验
            title_similarity=title_similarity,
            authors_match=authors_match,
            year_match=year_match,
            is_retracted=is_retracted,
            corrected_by=corrected_by,
            source="crossref",
            note=None if matched_by == "doi" else "matched by title",
        )
        tracer.event("tool_call", tool="verify_metadata", paper_id=paper.paper_id,
                     matched_by=matched_by, title_similarity=title_similarity,
                     authors_match=authors_match, year_match=year_match,
                     is_retracted=is_retracted)
        return check
    except Exception as e:
        tracer.event("tool_error", tool="verify_metadata", paper_id=paper.paper_id,
                     error=f"{type(e).__name__}: {e}"[:300])
        return MetadataCheck(source="crossref", note="lookup failed")
