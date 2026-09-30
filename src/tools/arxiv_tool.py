"""arXiv 检索与 PDF 下载（Atom API）。

设计要点：
- 所有公开函数绝不向调用方抛异常：失败记 tracer 事件后返回 [] / None；
- tenacity 仅对超时/连接错误/429/5xx 重试（3 次、指数退避）；
- PDF 流式下载：Content-Type 含 pdf 或 %PDF 魔数校验，30MB 大小上限；
- Atom 解析拆成纯函数（_parse_atom/_entry_to_paper）便于离线单测。
"""
from __future__ import annotations

import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import tenacity

from src.schemas import PaperRecord
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import norm_doi, truncate

_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_PDF_MAGIC = b"%PDF"
_MAX_PDF_BYTES = 30 * 1024 * 1024
_ABSTRACT_MAX_CHARS = 1500

# 已带 arXiv 字段前缀（all:/ti:/abs:/au:/cat:...）的查询不再包裹 all:"..."
_FIELD_PREFIX_RE = re.compile(r"^(all|ti|abs|au|cat|rn|id|co|jr|pn):", re.IGNORECASE)

_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """模块级复用的 httpx.Client（线程安全，跟随重定向）。"""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                follow_redirects=True,
                headers={"User-Agent": "research-agent/0.1 (literature survey; mailto config in sources.yaml)"},
            )
        return _client


def _squash_ws(text: str | None) -> str:
    """压缩空白：换行/制表/连续空格 → 单空格（arXiv 标题与摘要常见断行）。"""
    return " ".join((text or "").split())


def _is_retryable(exc: BaseException) -> bool:
    """仅超时/连接错误与 429/5xx 值得重试，其余（4xx、解析错误）立即失败。"""
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


@_retry_policy
def _http_get(url: str, *, params: dict[str, str] | None = None, timeout: float = 25.0) -> httpx.Response:
    """带重试的 GET（内部使用，公开函数统一兜底异常）。"""
    resp = _get_client().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp


@_retry_policy
def _stream_download(url: str, *, timeout: float) -> bytes:
    """流式下载并校验 PDF：大小超 30MB 中止；Content-Type 或 %PDF 魔数不匹配则拒绝。"""
    with _get_client().stream("GET", url, timeout=timeout) as resp:
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()
        looks_pdf = "pdf" in ctype
        buf = bytearray()
        for chunk in resp.iter_bytes():
            buf.extend(chunk)
            if len(buf) > _MAX_PDF_BYTES:
                raise ValueError("pdf exceeds 30MB limit")
        if not looks_pdf and not bytes(buf[:4]).startswith(_PDF_MAGIC):
            raise ValueError(f"not a pdf: content-type={ctype or 'unknown'}")
        return bytes(buf)


# 英文停用词 + 中文虚词：拆词 AND 时不参与（否则过度约束到零命中）
_ARXIV_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "with", "by",
    "vs", "versus", "between", "how", "what", "why", "is", "are", "的", "与", "和", "在", "对", "是否",
}


def _build_search_query(query: str, date_from: str | None, date_to: str | None) -> str:
    """组装 arXiv search_query：裸多词查询拆词 AND 连接；日期转 submittedDate 区间。

    整句包 all:"..." 是精确短语匹配，多词查询几乎零命中（实测坑）；
    拆词 AND 只保留前 6 个实词（翻译/描述式长句 AND 全部词同样过度约束）。
    date_from/date_to 形如 YYYY-MM-DD → submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]。
    """
    q = (query or "").strip()
    if q and not _FIELD_PREFIX_RE.match(q):
        terms: list[str] = []
        for tok in re.split(r"\s+", q):
            clean = re.sub(r"[^\w\-.]", "", tok).strip(".-")  # 去括号/标点，保留 - . 数字
            if clean and clean.lower() not in _ARXIV_STOPWORDS:
                terms.append(clean)
        if terms:
            q = " AND ".join(f"all:{t}" for t in terms[:6])
    if date_from or date_to:
        start = (re.sub(r"\D", "", date_from or "")[:8] or "19900101") + "0000"
        end = (re.sub(r"\D", "", date_to or "")[:8] or "20991231") + "2359"
        q = f"({q}) AND submittedDate:[{start} TO {end}]"
    return q


def _entry_to_paper(entry: ET.Element, query: str | None = None) -> PaperRecord | None:
    """单条 Atom entry → PaperRecord（纯函数，便于离线单测）。无法提取 id 时返回 None。"""
    id_el = entry.find(f"{{{_ATOM_NS}}}id")
    raw_id = (id_el.text or "").strip() if id_el is not None else ""
    # 形如 http://arxiv.org/abs/1706.03762v7 或旧式 .../abs/cs/0301016v1
    part = raw_id.rstrip("/").rsplit("/abs/", 1)[-1] if raw_id else ""
    arxiv_id = re.sub(r"v\d+$", "", part).strip()
    if not arxiv_id:
        return None

    title_el = entry.find(f"{{{_ATOM_NS}}}title")
    title = _squash_ws(title_el.text if title_el is not None else None)

    authors: list[str] = []
    for name_el in entry.findall(f"{{{_ATOM_NS}}}author/{{{_ATOM_NS}}}name"):
        name = _squash_ws(name_el.text if name_el is not None else None)
        if name:
            authors.append(name)

    year: int | None = None
    pub_el = entry.find(f"{{{_ATOM_NS}}}published")
    if pub_el is not None and pub_el.text:
        m = re.match(r"(\d{4})", pub_el.text.strip())
        if m:
            year = int(m.group(1))

    summary_el = entry.find(f"{{{_ATOM_NS}}}summary")
    abstract = _squash_ws(summary_el.text if summary_el is not None else None).strip()
    abstract = truncate(abstract, _ABSTRACT_MAX_CHARS) if abstract else None

    doi: str | None = None
    doi_el = entry.find(f"{{{_ARXIV_NS}}}doi")
    if doi_el is not None and doi_el.text:
        doi = norm_doi(doi_el.text)

    return PaperRecord(
        paper_id=f"arxiv:{arxiv_id}",
        title=title,
        authors=authors,
        year=year,
        abstract=abstract or None,
        doi=doi,
        arxiv_id=arxiv_id,
        source_url=raw_id,
        paper_type="preprint",
        retrieval_query=query,
        source_api="arxiv",
    )


def _parse_atom(xml_text: str, *, query: str | None = None) -> list[PaperRecord]:
    """解析 arXiv Atom 响应为 PaperRecord 列表（单条坏 entry 跳过，不影响其余）。"""
    root = ET.fromstring(xml_text)
    papers: list[PaperRecord] = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        try:
            paper = _entry_to_paper(entry, query)
        except Exception:
            paper = None
        if paper is not None:
            papers.append(paper)
    return papers


def _extract_arxiv_id(paper: PaperRecord) -> str:
    """从 PaperRecord 提取裸 arXiv id（兼容 arxiv: 前缀与版本号后缀）。"""
    raw = (paper.arxiv_id or "").strip()
    if not raw and (paper.paper_id or "").lower().startswith("arxiv:"):
        raw = paper.paper_id.split(":", 1)[1].strip()
    if raw.lower().startswith("arxiv:"):
        raw = raw.split(":", 1)[1].strip()
    return re.sub(r"v\d+$", "", raw)


def search_arxiv(query: str, *, date_from: str | None = None, date_to: str | None = None,
                 max_results: int | None = None) -> list[PaperRecord]:
    """date_from/date_to: YYYY-MM-DD；返回 paper_id=arxiv:xxx 的 PaperRecord；失败返回 []。"""
    tracer = get_active_tracer()
    settings = get_settings()
    if not settings.source("arxiv", "enabled", True):
        tracer.event("tool_error", tool="search_arxiv", query=query, error="arxiv disabled")
        return []
    if not (query or "").strip():
        return []
    try:
        base = str(settings.source("arxiv", "base_url", "https://export.arxiv.org/api/query"))
        timeout = float(settings.source("arxiv", "request_timeout_s", 25) or 25)
        n = int(max_results or settings.source("arxiv", "max_results_default", 10) or 10)
        resp = _http_get(
            base,
            params={
                "search_query": _build_search_query(query, date_from, date_to),
                "start": "0",
                "max_results": str(max(1, n)),
                "sortBy": "relevance",
            },
            timeout=timeout,
        )
        papers = _parse_atom(resp.text, query=query)
        tracer.event("tool_call", tool="search_arxiv", query=query, n_results=len(papers))
        return papers
    except Exception as e:
        tracer.event("tool_error", tool="search_arxiv", query=query,
                     error=f"{type(e).__name__}: {e}"[:300])
        return []


def fetch_arxiv_pdf(paper: PaperRecord, *, dest_dir: str | Path | None = None) -> str | None:
    """下载 PDF 到 dest_dir（默认 settings.papers_dir），更新 paper.pdf_path，失败返回 None。"""
    tracer = get_active_tracer()
    settings = get_settings()
    arxiv_id = _extract_arxiv_id(paper)
    if not arxiv_id:
        tracer.event("tool_error", tool="fetch_arxiv_pdf", paper_id=paper.paper_id, error="no arxiv id")
        return None
    try:
        if not settings.source("arxiv", "enabled", True):
            tracer.event("tool_error", tool="fetch_arxiv_pdf", arxiv_id=arxiv_id, error="arxiv disabled")
            return None
        base = str(settings.source("arxiv", "pdf_base", "https://arxiv.org/pdf"))
        timeout = float(settings.source("arxiv", "request_timeout_s", 25) or 25)
        data = _stream_download(f"{base}/{arxiv_id}", timeout=timeout)
        target_dir = Path(dest_dir) if dest_dir else Path(settings.papers_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        # 文件名安全化：仅保留字母数字与 ._-，其余替换为 _
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id)
        path = target_dir / f"arxiv_{safe}.pdf"
        path.write_bytes(data)
        paper.pdf_path = str(path)
        tracer.event("tool_call", tool="fetch_arxiv_pdf", arxiv_id=arxiv_id, path=str(path), bytes=len(data))
        return paper.pdf_path
    except Exception as e:
        tracer.event("tool_error", tool="fetch_arxiv_pdf", arxiv_id=arxiv_id,
                     error=f"{type(e).__name__}: {e}"[:300])
        return None
