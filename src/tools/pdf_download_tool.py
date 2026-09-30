"""通用论文内容获取：arXiv PDF 优先 → PDF 直链抓取 → 在线阅读（extract）退路。

设计要点：
- 只吃"白给的直链"：URL 路径以 .pdf 结尾（openreview.net/pdf?id=x 这类已知 PDF
  服务除外）才抓；落地页不解析页面找隐藏链接——脆弱、易触发反爬，还可能下到
  付费墙拦截页（实测 IEEE 对 doi.org 落地返回 202 空 HTML）；
- 校园网订阅（IP 认证）让订阅型直链直接可读（2026-09-20 实测：Nature
  nature.com/articles/<id>.pdf、Springer link.springer.com/content/pdf/<doi>.pdf
  均返回真 PDF）——下载行为与人工点击等价：一次一篇、不批量、不重试轰炸；
- 在线阅读退路：非 PDF 直链且 AnySearch extract 可用时抓网页正文存 .md
  （web_<id>.md），Reader 走文本文档路径（伪分页，页码契约不变）；
  doi.org 落地页跳过（多为付费墙壳）；
- 校验复用 arxiv_tool._stream_download：Content-Type / %PDF 魔数 / 30MB 上限。
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from src.schemas import PaperRecord
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.tools import arxiv_tool
from src.utils import truncate

# 已知"路径不带 .pdf 但确实返回 PDF"的服务（host, path 精确匹配）
_KNOWN_PDF_SERVICES = {("openreview.net", "/pdf")}


def _is_pdf_url(url: str) -> bool:
    """直链判定：路径以 .pdf 结尾（忽略查询串），或命中已知 PDF 服务表。"""
    try:
        parts = urlparse(url.strip())
    except Exception:  # noqa: BLE001
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    path = (parts.path or "").lower()
    if path.endswith(".pdf"):
        return True
    return (parts.hostname.lower(), path.rstrip("/")) in _KNOWN_PDF_SERVICES


def fetch_paper_pdf(paper: PaperRecord, *, dest_dir: str | Path | None = None) -> str | None:
    """下载论文 PDF 到 dest_dir（默认 settings.papers_dir），回填 paper.pdf_path；失败返回 None。

    分发顺序：arxiv_id 可用 → arXiv 主站（fetch_arxiv_pdf）；
    否则 source_url 是 PDF 直链 → 通用抓取（订阅 IP 生效）；两者皆非 → None。
    """
    tracer = get_active_tracer()

    # 1) arXiv 优先（主站与 export API 是两台服务，API 被限流不影响 PDF 下载）
    if arxiv_tool._extract_arxiv_id(paper):
        path = arxiv_tool.fetch_arxiv_pdf(paper, dest_dir=dest_dir)
        if path:
            return path
        # arXiv 失败继续尝试 source_url（可能是别的镜像直链），不直接判死

    # 2) 通用直链抓取
    url = (paper.source_url or "").strip()
    if not url:
        tracer.event("tool_skip", tool="fetch_paper_pdf", paper_id=paper.paper_id, reason="无 source_url")
        return None
    if not _is_pdf_url(url):
        # 3) 在线阅读退路：非 PDF 直链但开启 extract 时抓网页正文存 .md
        #    （伪分页保 [paper_id:页码] 契约；doi.org 落地页多是付费墙壳，跳过）
        return _fetch_web_text(paper, url, dest_dir)

    try:
        settings = get_settings()
        timeout = float(settings.source("pdf_download", "request_timeout_s", 60) or 60)
        data = arxiv_tool._stream_download(url, timeout=timeout)
        target_dir = Path(dest_dir) if dest_dir else Path(settings.papers_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", paper.paper_id)[:120]
        path = target_dir / f"paper_{safe}.pdf"
        path.write_bytes(data)
        paper.pdf_path = str(path)
        tracer.event("tool_call", tool="fetch_paper_pdf", paper_id=paper.paper_id, path=str(path), bytes=len(data))
        return paper.pdf_path
    except Exception as e:  # noqa: BLE001 —— 下载失败只记日志，绝不上抛
        tracer.event("tool_error", tool="fetch_paper_pdf", paper_id=paper.paper_id,
                     error=f"{type(e).__name__}: {truncate(str(e), 200)}")
        return None


def _fetch_web_text(paper: PaperRecord, url: str, dest_dir: str | Path | None) -> str | None:
    """在线阅读退路：AnySearch extract 抓正文 → 存 .md（Reader 的文本文档路径可读）。"""
    tracer = get_active_tracer()
    if url.startswith("https://doi.org/") or url.startswith("http://doi.org/"):
        tracer.event("tool_skip", tool="fetch_paper_pdf", paper_id=paper.paper_id,
                     reason="doi.org 落地页不做 extract（多为付费墙壳）")
        return None
    try:
        from src.tools import web_reader

        text = web_reader.extract_url(url)
    except Exception as e:  # noqa: BLE001 —— extract 不可用（未启用/未配 key）
        tracer.event("tool_skip", tool="fetch_paper_pdf", paper_id=paper.paper_id,
                     reason=f"在线阅读不可用: {type(e).__name__}")
        return None
    if not text:
        return None
    try:
        target_dir = Path(dest_dir) if dest_dir else Path(get_settings().papers_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", paper.paper_id)[:120]
        path = target_dir / f"web_{safe}.md"
        path.write_text(text, encoding="utf-8")
        paper.pdf_path = str(path)
        tracer.event("tool_call", tool="fetch_paper_pdf", paper_id=paper.paper_id,
                     path=str(path), bytes=len(text), mode="web_extract")
        return paper.pdf_path
    except Exception as e:  # noqa: BLE001
        tracer.event("tool_error", tool="fetch_paper_pdf", paper_id=paper.paper_id,
                     error=f"web 正文落盘失败: {type(e).__name__}: {truncate(str(e), 160)}")
        return None
