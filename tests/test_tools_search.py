"""tools 检索层测试：离线单测（纯函数/撤稿推断/优雅降级）+ 在线冒烟测试（宽松断言）。

在线测试依赖外部 API（arXiv/Crossref/S2）；Semantic Scholar 匿名限额 1req/s，
429 返回 [] 视为通过并记 warning。测试不调用任何真实 LLM。
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.schemas import PaperRecord
from src.settings import get_settings
from src.tools.arxiv_tool import _build_search_query, _parse_atom, fetch_arxiv_pdf, search_arxiv
from src.tools.citation_tool import check_retraction, match_paper, render_references
from src.tools.crossref_tool import _work_to_paper, lookup_crossref, verify_metadata
from src.tools.semantic_scholar_tool import _s2_to_paper, search_semantic_scholar, traverse_citations

# ---------------------------------------------------------------------------
# 测试夹具数据
# ---------------------------------------------------------------------------
ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <updated>2023-10-03T00:41:18-04:00</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is   All You
Need</title>
    <summary>We propose a   new simple network architecture, the Transformer,
based solely on attention mechanisms.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:doi>10.5555/3295222.3295349</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0301016v2</id>
    <published>2003-01-16T18:22:23Z</published>
    <title>Old Style   Identifier</title>
    <summary></summary>
    <author><name>Some One</name></author>
  </entry>
</feed>
"""

S2_ITEM = {
    "paperId": "abc123def",
    "title": "Attention Is All You Need",
    "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
    "year": 2017,
    "abstract": "x" * 2000,
    "externalIds": {"DOI": "https://doi.org/10.5555/3295222.3295349", "ArXiv": "1706.03762"},
    "citationCount": 100000,
    "venue": "NeurIPS",
    "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
}

CROSSREF_WORK = {
    "DOI": "10.5555/3295222.3295349",
    "title": ["Attention is all you need"],
    "author": [{"given": "Ashish", "family": "Vaswani"}, {"given": "Noam", "family": "Shazeer"}],
    "issued": {"date-parts": [[2017, 6]]},
    "is-referenced-by-count": 112389,
    "container-title": ["Advances in Neural Information Processing Systems"],
}


class _FakeResp:
    """离线测试用的假 httpx.Response。"""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _router(routes: dict[str, dict]):
    """按请求 path 精确路由到假响应。"""

    def fake(path: str, *args, params=None, **kwargs):
        if path in routes:
            return _FakeResp(routes[path])
        raise AssertionError(f"unexpected request path: {path}")

    return fake


# ---------------------------------------------------------------------------
# 离线单测：arXiv
# ---------------------------------------------------------------------------
def test_build_search_query() -> None:
    """拆词 AND（整句短语匹配几乎零命中）；字段前缀透传；日期区间不变。"""
    assert _build_search_query("deep learning", None, None) == "all:deep AND all:learning"
    assert _build_search_query("ti:transformer AND au:vaswani", None, None) == "ti:transformer AND au:vaswani"
    # 停用词剔除 + 超过 6 个实词截断（长翻译句不过度约束）
    assert _build_search_query("the study of how and why models fail", None, None) == \
        "all:study AND all:models AND all:fail"
    # 标点清洗：括号丢弃，连字符/点保留
    assert _build_search_query("(vision-language) 2.0 survey", None, None) == \
        "all:vision-language AND all:2.0 AND all:survey"
    q = _build_search_query("deep learning", "2024-01-01", "2024-12-31")
    assert q == "(all:deep AND all:learning) AND submittedDate:[202401010000 TO 202412312359]"
    q_open = _build_search_query("q", "2024-01-01", None)
    assert q_open.endswith("TO 209912312359]")


def test_parse_atom() -> None:
    papers = _parse_atom(ATOM_FEED, query="q")
    assert len(papers) == 2
    p0 = papers[0]
    assert p0.paper_id == "arxiv:1706.03762"  # 版本号已剥离
    assert p0.title == "Attention Is All You Need"  # 空白已压缩
    assert p0.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert p0.year == 2017
    assert p0.doi == "10.5555/3295222.3295349"
    assert p0.source_url == "http://arxiv.org/abs/1706.03762v7"
    assert p0.source_api == "arxiv"
    assert p0.paper_type == "preprint"
    assert p0.arxiv_id == "1706.03762"
    assert p0.retrieval_query == "q"
    assert p0.abstract and "Transformer" in p0.abstract
    # 旧式 id + 空 summary
    p1 = papers[1]
    assert p1.paper_id == "arxiv:cs/0301016"
    assert p1.year == 2003
    assert p1.abstract is None
    assert p1.doi is None


def test_search_arxiv_swallows_errors(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise httpx.ConnectError("net down")

    monkeypatch.setattr("src.tools.arxiv_tool._http_get", boom)
    assert search_arxiv("deep learning") == []


def test_search_arxiv_disabled(monkeypatch) -> None:
    settings = get_settings()
    monkeypatch.setitem(settings.sources, "arxiv", {"enabled": False})
    assert search_arxiv("deep learning") == []


# ---------------------------------------------------------------------------
# 离线单测：Semantic Scholar 映射
# ---------------------------------------------------------------------------
def test_s2_to_paper_doi_priority() -> None:
    p = _s2_to_paper(S2_ITEM, query="q")
    assert p is not None
    assert p.paper_id == "10.5555/3295222.3295349"  # DOI 优先且已 norm
    assert p.doi == "10.5555/3295222.3295349"
    assert p.arxiv_id == "1706.03762"
    assert p.source_url == "https://arxiv.org/pdf/1706.03762"  # openAccessPdf 优先
    assert p.venue == "NeurIPS"
    assert p.citation_count == 100000
    assert p.year == 2017
    assert p.abstract is not None and len(p.abstract) <= 1500  # 摘要截断
    assert p.source_api == "semantic_scholar"


def test_s2_to_paper_id_fallbacks() -> None:
    arxiv_only = dict(S2_ITEM, externalIds={"ArXiv": "1706.03762"}, openAccessPdf=None)
    p = _s2_to_paper(arxiv_only)
    assert p is not None
    assert p.paper_id == "arxiv:1706.03762"
    assert p.source_url == "https://arxiv.org/abs/1706.03762"

    bare = dict(S2_ITEM, externalIds={}, openAccessPdf=None)
    p2 = _s2_to_paper(bare)
    assert p2 is not None
    assert p2.paper_id == "s2:abc123def"
    assert p2.source_url == "https://www.semanticscholar.org/paper/abc123def"

    assert _s2_to_paper(dict(S2_ITEM, title="")) is None  # 无题名跳过


# ---------------------------------------------------------------------------
# 离线单测：Crossref 映射与核验
# ---------------------------------------------------------------------------
def test_work_to_paper() -> None:
    p = _work_to_paper(CROSSREF_WORK)
    assert p is not None
    assert p.paper_id == "10.5555/3295222.3295349"
    assert p.title == "Attention is all you need"
    assert p.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert p.year == 2017
    assert p.venue == "Advances in Neural Information Processing Systems"
    assert p.citation_count == 112389
    assert p.source_api == "crossref"
    assert p.source_url == "https://doi.org/10.5555/3295222.3295349"
    assert _work_to_paper({"title": ["no doi"]}) is None


def test_verify_metadata_doi_branch(monkeypatch) -> None:
    work = dict(CROSSREF_WORK, DOI="10.1/orig")
    monkeypatch.setattr(
        "src.tools.crossref_tool._http_get",
        _router({"/works/10.1%2Forig": {"message": work}}),
    )
    paper = PaperRecord(paper_id="10.1/orig", title="Attention Is All You Need",
                        authors=["Ashish Vaswani"], year=2017, doi="10.1/orig")
    check = verify_metadata(paper)
    assert check.doi_valid is True
    assert check.title_similarity is not None and check.title_similarity >= 0.99
    assert check.authors_match is True  # 首作者姓 Vaswani == Vaswani
    assert check.year_match is True
    assert check.is_retracted is False
    assert check.note is None


def test_verify_metadata_title_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.tools.crossref_tool._http_get",
        _router({"/works": {"message": {"items": [CROSSREF_WORK]}}}),
    )
    paper = PaperRecord(paper_id="arxiv:1706.03762", title="Attention Is All You Need",
                        authors=["Ashish Vaswani"], year=2018)  # 无 DOI，年差 1
    check = verify_metadata(paper)
    assert check.doi_valid is None
    assert check.note == "matched by title"
    assert check.title_similarity == 1.0
    assert check.authors_match is True
    assert check.year_match is True  # |2018-2017| <= 1


def test_verify_metadata_not_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.tools.crossref_tool._http_get",
        _router({"/works": {"message": {"items": []}}}),
    )
    paper = PaperRecord(paper_id="arxiv:1", title="Nonexistent Paper XYZ")
    check = verify_metadata(paper)
    assert check.note == "not found in crossref"
    assert check.doi_valid is None


def test_verify_metadata_network_failure(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("src.tools.crossref_tool._http_get", boom)
    paper = PaperRecord(paper_id="10.1/x", title="t", doi="10.1/x")
    check = verify_metadata(paper)
    assert check.note == "lookup failed"  # 网络失败绝不让 verifier 崩掉
    assert check.title_similarity is None
    assert check.authors_match is None
    assert check.year_match is None
    assert check.is_retracted is None


# ---------------------------------------------------------------------------
# 离线单测：撤稿推断（relation）
# ---------------------------------------------------------------------------
def test_check_retraction_no_doi() -> None:
    paper = PaperRecord(paper_id="arxiv:1234.5678", title="Some Preprint")
    status = check_retraction(paper)
    assert status.paper_id == "arxiv:1234.5678"
    assert status.is_retracted is None
    assert status.notice == "no doi"  # schema 无 note 字段，说明写入 notice


def test_check_retraction_detects_retraction(monkeypatch) -> None:
    orig = {
        "DOI": "10.1/orig",
        "title": ["Original Paper"],
        "author": [{"given": "Ann", "family": "Author"}],
        "issued": {"date-parts": [[2015]]},
        "relation": {"is-update-of": [{"id": "10.1/retraction-notice"}]},
    }
    notice = {"DOI": "10.1/retraction-notice", "title": ["RETRACTION: Original Paper"],
              "type": "journal-article"}
    monkeypatch.setattr(
        "src.tools.crossref_tool._http_get",
        _router({
            "/works/10.1%2Forig": {"message": orig},
            "/works/10.1%2Fretraction-notice": {"message": notice},
        }),
    )
    paper = PaperRecord(paper_id="10.1/orig", title="Original Paper", doi="10.1/orig")
    status = check_retraction(paper)
    assert status.is_retracted is True
    assert status.notice and "retraction" in status.notice.lower()
    assert paper.is_retracted is True  # 回写


def test_check_retraction_correction_vs_network_fail(monkeypatch) -> None:
    orig = {
        "DOI": "10.1/orig", "title": ["Original Paper"],
        "relation": {"is-update-of": [{"id": "10.1/correction"}]},
    }
    correction = {"DOI": "10.1/correction", "title": ["Correction to: Original Paper"],
                  "type": "journal-article"}
    monkeypatch.setattr(
        "src.tools.crossref_tool._http_get",
        _router({
            "/works/10.1%2Forig": {"message": orig},
            "/works/10.1%2Fcorrection": {"message": correction},
        }),
    )
    paper = PaperRecord(paper_id="10.1/orig", title="Original Paper", doi="10.1/orig")
    check = verify_metadata(paper)
    assert check.is_retracted is False
    assert check.corrected_by == ["10.1/correction"]  # 非撤稿 → 更正记录

    def boom(*args, **kwargs):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr("src.tools.crossref_tool._http_get", boom)
    status = check_retraction(paper)
    assert status.is_retracted is None
    assert status.notice == "lookup failed"


# ---------------------------------------------------------------------------
# 离线单测：参考文献渲染与题名匹配
# ---------------------------------------------------------------------------
def test_render_references() -> None:
    papers = [
        PaperRecord(paper_id="10.1/a", title="Deep Learning", authors=["A B", "C D", "E F", "G H"],
                    year=2015, venue="Nature", doi="10.1/a"),
        PaperRecord(paper_id="10.1/b", title="No Meta Paper"),
        PaperRecord(paper_id="10.1/c", title="Url Paper", authors=["X Y"],
                    source_url="https://arxiv.org/abs/1"),
        PaperRecord(paper_id="10.1/d", title="Venue Only", authors=["P Q"], year=2020),
    ]
    lines = render_references(papers).split("\n")
    assert lines[0] == "[1] A B, C D, E F, et al. Deep Learning. Nature, 2015. doi:10.1/a"
    assert lines[1] == "[2] No Meta Paper. 10.1/b"  # 无 doi/url → paper_id 兜底
    assert lines[2] == "[3] X Y. Url Paper. https://arxiv.org/abs/1"  # doi 缺失 → source_url
    assert lines[3] == "[4] P Q. Venue Only. 2020. 10.1/d"  # 无 doi/url → paper_id，无 venue 省略
    assert render_references(papers, style="apa")  # 未知 style 按 default 处理不崩溃
    assert render_references([]) == ""


def test_match_paper() -> None:
    p_exact = PaperRecord(paper_id="a", title="Attention Is All You Need")
    p_fuzzy = PaperRecord(paper_id="b", title="Attention Is All You Need Extended Abstract")
    p_other = PaperRecord(paper_id="c", title="GAN Training Tricks")
    # norm_title 精确相等（大小写/空白不敏感）
    assert match_paper("ATTENTION is all you need", [p_other, p_exact]) is p_exact
    # 精确命中优先于排序在前的模糊高分
    assert match_paper("attention is all you need", [p_fuzzy, p_exact]) is p_exact
    # token 子集 → token_set_ratio 100
    assert match_paper("attention is all you need", [p_fuzzy]) is p_fuzzy
    # 无匹配
    assert match_paper("quantum error correction", [p_exact, p_other]) is None
    assert match_paper("", [p_exact]) is None


# ---------------------------------------------------------------------------
# 在线冒烟测试（宽松断言；S2 容忍 429）
# ---------------------------------------------------------------------------
@pytest.mark.timeout(90)
def test_arxiv_search_online() -> None:
    papers = search_arxiv("attention is all you need", max_results=3)
    assert len(papers) >= 1
    assert any("attention" in (p.title or "").lower() for p in papers)
    assert all(p.paper_id.startswith("arxiv:") for p in papers)


@pytest.mark.timeout(120)
def test_fetch_arxiv_pdf_online(tmp_path) -> None:
    papers = search_arxiv("attention is all you need", max_results=3)
    target = next((p for p in papers if p.paper_id.startswith("arxiv:")), None)
    assert target is not None, "arxiv search returned nothing to download"
    path = fetch_arxiv_pdf(target, dest_dir=tmp_path)
    assert path is not None
    file = Path(path)
    assert file.exists() and file.stat().st_size > 1000
    assert file.read_bytes()[:4] == b"%PDF"
    assert target.pdf_path == path


@pytest.mark.timeout(90)
def test_crossref_lookup_and_verify_online() -> None:
    rec = lookup_crossref("Attention Is All You Need")
    assert rec is not None
    assert rec.doi is not None and rec.doi.startswith("10.")
    # DOI 形式再查一次（覆盖 %2F 编码路径）
    rec2 = lookup_crossref(rec.doi)
    assert rec2 is not None and rec2.paper_id == rec.doi
    # 核验：与自身来源比对，题名相似度应很高
    check = verify_metadata(rec)
    assert check.title_similarity is not None and check.title_similarity >= 0.8
    assert check.is_retracted is False  # 经典论文无 relation → 未撤稿


@pytest.mark.timeout(90)
def test_s2_search_online_tolerant() -> None:
    import warnings

    res = search_semantic_scholar("attention is all you need", max_results=3)
    if not res:
        warnings.warn("Semantic Scholar rate-limited (429): empty result treated as pass")
        return
    assert res[0].paper_id and res[0].title


@pytest.mark.timeout(90)
def test_s2_traverse_citations_online_tolerant() -> None:
    import warnings

    res = traverse_citations("arxiv:1706.03762", direction="citing", limit=5)
    if not res:
        warnings.warn("Semantic Scholar rate-limited (429): empty citations treated as pass")
        return
    assert all(p.paper_id for p in res)


# ---------------------------------------------------------------------------
# 通用 PDF 下载器（fetch_paper_pdf）：arXiv 优先 + 直链抓取 + 落地页放弃
# ---------------------------------------------------------------------------
def test_is_pdf_url() -> None:
    from src.tools.pdf_download_tool import _is_pdf_url

    assert _is_pdf_url("https://www.nature.com/articles/s41586-025-09422-z.pdf")
    assert _is_pdf_url("https://link.springer.com/content/pdf/10.1007/s11704-026-60308-3.pdf?x=1")  # 查询串忽略
    assert _is_pdf_url("https://openreview.net/pdf?id=xyz123")  # 已知 PDF 服务
    assert not _is_pdf_url("https://doi.org/10.1109/tmi.2026.3661001")  # 落地页
    assert not _is_pdf_url("https://example.com/blog/post")  # 普通网页
    assert not _is_pdf_url("ftp://example.com/x.pdf")  # 非 http(s)
    assert not _is_pdf_url("")


def test_fetch_paper_pdf_direct_link(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """非 arXiv 的 .pdf 直链 → 抓取落盘并回填 pdf_path（Nature/Springer 订阅直链场景）。"""
    from src.tools import arxiv_tool, pdf_download_tool

    got: list[str] = []

    def fake_stream(url: str, *, timeout: float) -> bytes:
        got.append(url)
        return b"%PDF-1.7 fake-content"

    monkeypatch.setattr(arxiv_tool, "_stream_download", fake_stream)
    paper = PaperRecord(
        paper_id="10.1038/s41586-025-09422-z", doi="10.1038/s41586-025-09422-z",
        title="DeepSeek-R1", source_url="https://www.nature.com/articles/s41586-025-09422-z.pdf",
    )
    path = pdf_download_tool.fetch_paper_pdf(paper, dest_dir=tmp_path)
    assert path and Path(path).exists()
    assert Path(path).read_bytes() == b"%PDF-1.7 fake-content"
    assert paper.pdf_path == path
    assert got == [paper.source_url]
    assert "paper_10.1038_s41586-025-09422-z.pdf" in str(path)  # paper_id 安全化（/ → _）


def test_fetch_paper_pdf_landing_page_refused(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """非直链（doi.org 落地页）明确放弃：不发请求、返回 None。"""
    from src.tools import arxiv_tool, pdf_download_tool

    def no_call(url: str, *, timeout: float) -> bytes:  # pragma: no cover —— 不应被调到
        raise AssertionError(f"落地页不应发请求: {url}")

    monkeypatch.setattr(arxiv_tool, "_stream_download", no_call)
    paper = PaperRecord(paper_id="10.1109/x", title="IEEE Paper",
                        source_url="https://doi.org/10.1109/tmi.2026.3661001")
    assert pdf_download_tool.fetch_paper_pdf(paper, dest_dir=tmp_path) is None
    assert paper.pdf_path is None


def test_fetch_paper_pdf_arxiv_priority(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """有 arxiv_id 优先走 arXiv 主站；source_url 直链只在无 arxiv_id 时启用。"""
    from src.tools import arxiv_tool, pdf_download_tool

    calls: list[str] = []

    def fake_arxiv(paper: PaperRecord, *, dest_dir=None) -> str:
        calls.append("arxiv")
        return str(tmp_path / "arxiv_ok.pdf")

    monkeypatch.setattr(arxiv_tool, "fetch_arxiv_pdf", fake_arxiv)
    paper = PaperRecord(
        paper_id="arxiv:2305.18290", arxiv_id="2305.18290", title="DPO",
        source_url="https://arxiv.org/pdf/2305.18290",
    )
    assert pdf_download_tool.fetch_paper_pdf(paper, dest_dir=tmp_path) == str(tmp_path / "arxiv_ok.pdf")
    assert calls == ["arxiv"]


# ---------------------------------------------------------------------------
# 在线阅读：web_reader.extract_url + 文本文档解析 + fetch 退路
# ---------------------------------------------------------------------------
def test_extract_url_gates_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """未启用/非 http(s)/业务失败/正文过短 → None，绝不抛异常。"""
    from src.tools import web_reader

    monkeypatch.setattr(web_reader, "extract_available", lambda: False)
    assert web_reader.extract_url("https://x.example.com/a") is None  # 未启用

    monkeypatch.setattr(web_reader, "extract_available", lambda: True)
    assert web_reader.extract_url("ftp://x/y") is None  # 非 http(s)

    monkeypatch.setattr(web_reader, "_post_extract", lambda u: {"code": -1, "message": "extract_failed"})
    assert web_reader.extract_url("https://x.example.com/a") is None  # 业务失败

    monkeypatch.setattr(
        web_reader, "_post_extract",
        lambda u: {"code": 0, "data": {"content": "[nav](http://x)\n\nshort"}},  # 清洗后过短
    )
    assert web_reader.extract_url("https://x.example.com/a") is None


def test_extract_url_cleans_and_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """链接留锚文本、导航行剔除、正文够长 → 返回清洗后 Markdown。"""
    from src.tools import web_reader

    monkeypatch.setattr(web_reader, "extract_available", lambda: True)
    body = "[Jump to content](http://x#nav)\n\n" + ("DPO [avoids](http://x) reward modeling. " * 40)
    monkeypatch.setattr(web_reader, "_post_extract", lambda u: {"code": 0, "data": {"content": body}})
    out = web_reader.extract_url("https://x.example.com/a")
    assert out and "avoids" in out and "](http" not in out and "Jump to content" not in out


def test_parse_text_doc_pseudo_pages(tmp_path) -> None:
    """.md → 伪分页 PaperDoc：页码顺序、块非空、[paper_id:页码] 契约可用。"""
    from src.tools.pdf_tool import parse_pdf, retrieve_chunks

    md = tmp_path / "web_test.md"
    md.write_text("\n\n".join(f"Paragraph {i} about GRPO reward hacking stability." for i in range(120)),
                  encoding="utf-8")
    doc = parse_pdf(md, paper_id="web:abc123")
    assert doc.n_pages >= 2 and doc.chunks
    assert all(1 <= c.page <= doc.n_pages for c in doc.chunks)
    hits = retrieve_chunks(doc, "GRPO reward hacking", k=2)
    assert hits and "GRPO" in hits[0]["text"]


def test_fetch_paper_pdf_web_text_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """非 PDF 直链 + extract 可用 → 抓正文存 .md 并回填 pdf_path（在线阅读退路）。"""
    from src.tools import pdf_download_tool, web_reader

    monkeypatch.setattr(web_reader, "extract_url", lambda u: "# 报告\n\n" + "DPO 分析正文。 " * 200)
    paper = PaperRecord(paper_id="web:abc", title="Blog Report", source_url="https://example.com/blog/dpo")
    path = pdf_download_tool.fetch_paper_pdf(paper, dest_dir=tmp_path)
    assert path and Path(path).suffix == ".md" and Path(path).exists()
    assert paper.pdf_path == path

    # extract 不可用 → None（不硬来）
    monkeypatch.setattr(web_reader, "extract_url", lambda u: None)
    paper2 = PaperRecord(paper_id="web:def", title="T", source_url="https://example.com/x")
    assert pdf_download_tool.fetch_paper_pdf(paper2, dest_dir=tmp_path) is None

    # doi.org 落地页即使 extract 可用也跳过
    paper3 = PaperRecord(paper_id="10.1/x", title="T", source_url="https://doi.org/10.1/x")
    assert pdf_download_tool.fetch_paper_pdf(paper3, dest_dir=tmp_path) is None
