"""Semantic Scholar 检索与引用遍历。

设计要点：
- 模块级限速：threading.Lock + 上次调用时间戳，保证相邻请求间隔 ≥ rate_limit_s
  （匿名限额 1req/s，429 极常见，限速能显著降低失败率）；
- paper_id 映射优先级：DOI（norm 后）> arxiv:xxx > s2:xxx；
- 所有公开函数绝不抛异常：失败/限流返回 [] / None；
- 映射逻辑拆成纯函数 _s2_to_paper 便于离线单测。
"""
from __future__ import annotations

import os
import re
import threading
import time
from urllib.parse import quote

import httpx
import tenacity

from src.schemas import PaperRecord
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import norm_doi, truncate

_FIELDS = "title,authors,year,abstract,externalIds,citationCount,venue,openAccessPdf"
_ABSTRACT_MAX_CHARS = 1500
_DOI_RE = re.compile(r"^10\.\d{4,9}/", re.IGNORECASE)

# ---- 模块级限速（并行 Researcher 共享同一个间隔窗口） ----
_rate_lock = threading.Lock()
_last_call = 0.0

_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _respect_rate_limit() -> None:
    """阻塞直到距上次 S2 调用 ≥ rate_limit_s。持锁睡眠以串行化并发调用。"""
    global _last_call
    with _rate_lock:
        wait_s = float(get_settings().source("semantic_scholar", "rate_limit_s", 1.2) or 1.2)
        delta = time.monotonic() - _last_call
        if delta < wait_s:
            time.sleep(wait_s - delta)
        _last_call = time.monotonic()


def _get_client() -> httpx.Client:
    """模块级复用的 httpx.Client；配置了 API key 时附加 x-api-key 头。"""
    global _client
    with _client_lock:
        if _client is None:
            headers = {"User-Agent": "research-agent/0.1 (literature survey)"}
            key_env = get_settings().source("semantic_scholar", "api_key_env", "SEMANTIC_SCHOLAR_API_KEY")
            api_key = os.environ.get(str(key_env or ""), "") if key_env else ""
            if api_key:
                headers["x-api-key"] = api_key
            _client = httpx.Client(follow_redirects=True, headers=headers)
        return _client


def _is_retryable(exc: BaseException) -> bool:
    """仅超时/连接错误与 429/5xx 重试。"""
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503}
    return False


# S2 匿名限额紧，重试过多会拖垮并行调用：只补 1 次
_retry_policy = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_fixed(2),
    reraise=True,
)


@_retry_policy
def _http_get(path: str, *, params: dict[str, str] | None = None) -> httpx.Response:
    """带限速与重试的 GET（内部使用，公开函数统一兜底异常）。"""
    _respect_rate_limit()
    settings = get_settings()
    base = str(settings.source("semantic_scholar", "base_url", "https://api.semanticscholar.org/graph/v1"))
    timeout = float(settings.source("semantic_scholar", "request_timeout_s", 25) or 25)
    url = path if path.startswith("http") else f"{base}{path}"
    resp = _get_client().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


def _squash_ws(text: str | None) -> str:
    """压缩空白为单空格。"""
    return " ".join((text or "").split())


def _to_int(value: object) -> int | None:
    """宽容转 int（S2 偶发返回字符串数字）。"""
    try:
        return int(value) if value is not None else None  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _s2_to_paper(item: dict, *, query: str | None = None) -> PaperRecord | None:
    """单条 S2 paper JSON → PaperRecord（纯函数，便于离线单测）。

    paper_id 优先级：externalIds.DOI（norm 后）> arxiv:{ArXiv} > s2:{paperId}；
    无任何标识或无题名时返回 None。
    """
    if not isinstance(item, dict):
        return None
    title = _squash_ws(item.get("title"))
    if not title:
        return None
    ext = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    doi = norm_doi(ext.get("DOI"))
    arxiv_id = _squash_ws(ext.get("ArXiv")) or None
    s2_id = _squash_ws(item.get("paperId")) or None
    if doi:
        paper_id = doi
    elif arxiv_id:
        paper_id = f"arxiv:{arxiv_id}"
    elif s2_id:
        paper_id = f"s2:{s2_id}"
    else:
        return None

    authors = [
        _squash_ws(a.get("name"))
        for a in (item.get("authors") or [])
        if isinstance(a, dict) and _squash_ws(a.get("name"))
    ]

    # 开放获取 PDF 链接优先；缺失时退化到 DOI/arXiv/学术主页
    oa = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
    oa_url = _squash_ws(oa.get("url")) or ""
    source_url = oa_url or (
        f"https://doi.org/{doi}" if doi
        else f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
        else f"https://www.semanticscholar.org/paper/{s2_id}" if s2_id
        else ""
    )

    abstract = _squash_ws(item.get("abstract")) or None
    if abstract:
        abstract = truncate(abstract, _ABSTRACT_MAX_CHARS)

    return PaperRecord(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=_to_int(item.get("year")),
        venue=_squash_ws(item.get("venue")) or None,
        abstract=abstract,
        doi=doi,
        arxiv_id=arxiv_id,
        source_url=source_url,
        citation_count=_to_int(item.get("citationCount")),
        retrieval_query=query,
        source_api="semantic_scholar",
    )


def _looks_like_title(pid: str) -> bool:
    """输入含空格且不像 DOI / 已知 id 前缀 → 视为题名（直接走题名检索）。"""
    low = pid.lower()
    if low.startswith(("doi:", "arxiv:", "corpusid:", "s2:")):
        return False
    if _DOI_RE.match(low):
        return False
    return " " in pid


def _to_s2_api_id(paper_id: str) -> str:
    """内部 paper_id（裸 DOI / arxiv:xxx / s2:xxx）→ S2 API 接受的 id 形式。"""
    pid = (paper_id or "").strip()
    low = pid.lower()
    if low.startswith("doi:"):
        return "DOI:" + pid[4:].strip()
    if low.startswith("arxiv:"):
        return "arXiv:" + pid[6:].strip()
    if low.startswith("corpusid:"):
        return "CorpusId:" + pid[9:].strip()
    if low.startswith("s2:"):
        return pid[3:].strip()
    if _DOI_RE.match(low):
        return "DOI:" + pid
    return pid  # 裸 S2 paperId 等直接交给 API


def search_semantic_scholar(query: str, *, max_results: int | None = None,
                            year: str | None = None) -> list[PaperRecord]:
    """year 形如 2020- / -2024 / 2020-2024；paper_id=DOI 优先否则 arXiv:xxx；失败/限流返回 []。"""
    tracer = get_active_tracer()
    settings = get_settings()
    if not settings.source("semantic_scholar", "enabled", True):
        tracer.event("tool_error", tool="search_semantic_scholar", query=query, error="semantic_scholar disabled")
        return []
    if not (query or "").strip():
        return []
    try:
        n = int(max_results or settings.source("semantic_scholar", "max_results_default", 10) or 10)
        params = {"query": query.strip(), "limit": str(max(1, min(n, 100))), "fields": _FIELDS}
        if year and str(year).strip():
            params["year"] = str(year).strip()
        resp = _http_get("/paper/search", params=params)
        data = (resp.json() or {}).get("data") or []
        papers = [p for p in (_s2_to_paper(it, query=query) for it in data) if p is not None]
        tracer.event("tool_call", tool="search_semantic_scholar", query=query, n_results=len(papers))
        return papers
    except Exception as e:
        tracer.event("tool_error", tool="search_semantic_scholar", query=query,
                     error=f"{type(e).__name__}: {e}"[:300])
        return []


def get_paper(paper_id: str) -> PaperRecord | None:
    """支持 DOI:10.x / arXiv:xxxx / CorpusId:nn；查不到退化为题名检索；失败返回 None。"""
    tracer = get_active_tracer()
    if not get_settings().source("semantic_scholar", "enabled", True):
        tracer.event("tool_error", tool="get_paper", paper_id=paper_id, error="semantic_scholar disabled")
        return None
    pid = (paper_id or "").strip()
    if not pid:
        return None

    item: dict | None = None
    if not _looks_like_title(pid):
        try:
            resp = _http_get(f"/paper/{quote(_to_s2_api_id(pid), safe='')}", params={"fields": _FIELDS})
            item = resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:  # 404 走题名退化，其余直接失败
                tracer.event("tool_error", tool="get_paper", paper_id=pid,
                             error=f"{type(e).__name__}: {e}"[:300])
                return None
        except Exception as e:
            tracer.event("tool_error", tool="get_paper", paper_id=pid,
                         error=f"{type(e).__name__}: {e}"[:300])
            return None

    if item is None:
        # 404 或输入像题名 → 退化为题名检索取第一条
        try:
            resp = _http_get("/paper/search", params={"query": pid, "limit": "1", "fields": _FIELDS})
            data = (resp.json() or {}).get("data") or []
            if data and isinstance(data[0], dict):
                item = data[0]
        except Exception as e:
            tracer.event("tool_error", tool="get_paper", paper_id=pid, stage="search_fallback",
                         error=f"{type(e).__name__}: {e}"[:300])
            return None

    paper = _s2_to_paper(item) if item else None
    tracer.event("tool_call", tool="get_paper", paper_id=pid, found=paper is not None)
    return paper


def traverse_citations(paper_id: str, *, direction: str = "citing", limit: int = 8) -> list[PaperRecord]:
    """direction: citing=谁引用了它 / cited=它引用了谁；失败返回 []。"""
    tracer = get_active_tracer()
    if not get_settings().source("semantic_scholar", "enabled", True):
        tracer.event("tool_error", tool="traverse_citations", paper_id=paper_id, error="semantic_scholar disabled")
        return []
    s2id = _to_s2_api_id(paper_id or "")
    if not s2id:
        return []
    endpoint = "citations" if direction == "citing" else "references"
    try:
        n = max(1, min(int(limit), 100))
        resp = _http_get(
            f"/paper/{quote(s2id, safe='')}/{endpoint}",
            params={"fields": _FIELDS, "limit": str(n)},
        )
        data = (resp.json() or {}).get("data") or []
        key = "citingPaper" if direction == "citing" else "citedPaper"
        papers: list[PaperRecord] = []
        for row in data:
            item = row.get(key) if isinstance(row, dict) else None
            paper = _s2_to_paper(item) if isinstance(item, dict) else None
            if paper is not None:
                papers.append(paper)
        tracer.event("tool_call", tool="traverse_citations", paper_id=paper_id,
                     direction=direction, n_results=len(papers))
        return papers
    except Exception as e:
        tracer.event("tool_error", tool="traverse_citations", paper_id=paper_id, direction=direction,
                     error=f"{type(e).__name__}: {e}"[:300])
        return []
