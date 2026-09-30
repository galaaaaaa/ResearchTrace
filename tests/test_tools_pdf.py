"""PDF 解析 / BM25 检索 / PaperQA 适配 / 图表多模态工具的离线单元测试。

全部使用 pymupdf 合成 PDF，LLM 一律走 fake 后端，不烧真实 token、不发网络请求。
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

from src.llm import get_fake_llm
from src.settings import get_settings
from src.tools.figure_tool import analyze_figure, detect_figure_regions, render_region_png
from src.tools.paperqa_tool import ask_papers, paperqa_available
from src.tools.pdf_tool import BM25Index, Chunk, extract_title, make_paper_id_from_pdf, parse_pdf, retrieve_chunks

# 重复关键词的正文段（保证 BM25 能稳定命中）
_BODY = (
    "GRPO reinforcement learning optimizes the language model policy with group "
    "relative policy optimization and dense reward signals. "
) * 6


def _tbox(page: pymupdf.Page, rect: pymupdf.Rect, text: str, *, fontsize: float) -> None:
    """insert_textbox 并断言全部写入（返回值 < 0 表示文本未完全放入）。"""
    left = page.insert_textbox(rect, text, fontsize=fontsize, fontname="helv")
    assert left >= 0, f"文本未放入文本框（剩余 {left:.1f}pt）: {text[:40]!r}"


def _make_paper_pdf(path: Path) -> Path:
    """3 页合成论文：大字号章节标题 + 关键词正文 + 第 2 页一组相邻矩形（模拟 figure）。"""
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()

    p1 = doc[0]
    _tbox(p1, pymupdf.Rect(72, 60, 523, 100), "Deep Research with GRPO Agents", fontsize=20)
    _tbox(p1, pymupdf.Rect(72, 110, 523, 145), "1. Introduction", fontsize=16)
    _tbox(p1, pymupdf.Rect(72, 150, 523, 420), _BODY, fontsize=10)

    p2 = doc[1]
    _tbox(p2, pymupdf.Rect(72, 60, 523, 95), "2. Method", fontsize=16)
    _tbox(p2, pymupdf.Rect(72, 100, 523, 300), _BODY, fontsize=10)
    # 2x2 相邻矩形簇：水平/垂直间隙均为 10pt（< 20pt 聚类阈值）
    for i in range(4):
        x0 = 90 + (i % 2) * 110
        y0 = 330 + (i // 2) * 90
        p2.draw_rect(pymupdf.Rect(x0, y0, x0 + 100, y0 + 80), color=(0, 0, 0), width=1)

    p3 = doc[2]
    _tbox(p3, pymupdf.Rect(72, 60, 523, 95), "3. Results", fontsize=16)
    _tbox(p3, pymupdf.Rect(72, 100, 523, 300), _BODY, fontsize=10)
    _tbox(p3, pymupdf.Rect(72, 320, 523, 355), "References", fontsize=16)
    _tbox(p3, pymupdf.Rect(72, 360, 523, 480), "[1] Author et al. Group Relative Policy Optimization. 2024.", fontsize=9)

    doc.save(str(path))
    doc.close()
    return path


def _make_empty_pdf(path: Path) -> Path:
    """无可提取文本的 PDF（模拟扫描件）。"""
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def test_parse_pdf_sections_and_chunks(tmp_path: Path) -> None:
    pdf = _make_paper_pdf(tmp_path / "paper.pdf")
    doc = parse_pdf(pdf)

    assert doc.n_pages == 3
    assert doc.is_scanned is False
    titles = " | ".join(s.title for s in doc.sections).lower()
    assert "introduction" in titles
    assert "method" in titles

    assert doc.chunks, "必须产出非空 chunks"
    assert all(1 <= c.page <= 3 for c in doc.chunks), "chunk 页码必须在 1..3 内"
    assert any(c.page == 1 and "grpo" in c.text.lower() for c in doc.chunks), "第 1 页正文应落入 page=1 的块"
    assert all(c.chunk_id for c in doc.chunks)
    assert all(c.section for c in doc.chunks)

    # paper_id 稳定且与解析结果一致
    pid = make_paper_id_from_pdf(pdf)
    assert pid.startswith("sha256:") and len(pid) == len("sha256:") + 12
    assert pid == make_paper_id_from_pdf(pdf)
    assert doc.paper_id == pid
    assert doc.sha256

    # page_text 重新打开取页文本
    page1 = doc.page_text(1)
    assert "GRPO" in page1
    assert doc.page_text(99) == ""

    # info 结构化摘要可用
    assert doc.info.n_chunks == len(doc.chunks)
    assert doc.info.sections == [s.title for s in doc.sections]


def test_retrieve_chunks_and_extract_title(tmp_path: Path) -> None:
    pdf = _make_paper_pdf(tmp_path / "paper.pdf")
    hits = retrieve_chunks(str(pdf), "grpo reward optimization", k=3)
    assert hits, "BM25 必须命中关键词文本"
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True), "得分必须降序"
    assert any("grpo" in h["text"].lower() for h in hits)
    for h in hits:
        assert {"page", "section", "text", "score"} <= set(h)
        assert 1 <= h["page"] <= 3

    title = extract_title(str(pdf))
    assert title and "GRPO" in title

    # PaperDoc 也可直接作为 source
    doc = parse_pdf(pdf)
    assert retrieve_chunks(doc, "reward", k=2)
    assert extract_title(doc)

    # 缓存路径：同一 (abspath, mtime) 重复检索不重复解析
    again = retrieve_chunks(str(pdf), "grpo", k=2)
    assert again


def test_scanned_pdf_detection(tmp_path: Path) -> None:
    pdf = _make_empty_pdf(tmp_path / "scan.pdf")
    doc = parse_pdf(pdf)
    assert doc.is_scanned is True
    assert doc.n_pages == 2
    assert doc.chunks == []
    assert retrieve_chunks(str(pdf), "anything") == []


def test_bm25_index_direct() -> None:
    chunks = [
        Chunk("c0", 1, "intro", "GRPO uses group relative policy optimization with reward."),
        Chunk("c1", 2, "method", "The transformer encoder attends over token representations."),
        Chunk("c2", 3, "refs", "强化学习在近年的大模型训练中被广泛采用。"),
    ]
    index = BM25Index(chunks)
    hits = index.query("grpo reward", k=2)
    assert hits and hits[0][0].chunk_id == "c0"
    assert hits[0][1] >= hits[-1][1]

    # CJK 逐字分词可用
    hits_cjk = index.query("强化学习")
    assert hits_cjk and hits_cjk[0][0].chunk_id == "c2"

    assert index.query("") == []
    assert BM25Index([]).query("grpo") == []


def test_paperqa_unavailable() -> None:
    assert paperqa_available() is False
    assert ask_papers("what is grpo?", []) is None
    assert ask_papers("what is grpo?", ["/nonexistent/a.pdf"]) is None


def test_detect_and_render_figure(tmp_path: Path) -> None:
    pdf = _make_paper_pdf(tmp_path / "paper.pdf")
    regions = detect_figure_regions(str(pdf))
    assert regions, "第 2 页矩形簇应被识别为图表区域"
    region = next((r for r in regions if r.page == 2), None)
    assert region is not None
    assert region.kind in {"figure", "table"}
    assert region.area_ratio > 0.05

    png = render_region_png(str(pdf), region.page, region.bbox, dest_dir=tmp_path / "figs")
    assert png is not None and png.exists() and png.stat().st_size > 0
    assert png.suffix == ".png"

    # 越界页 / 不存在文件安全降级
    assert render_region_png(str(pdf), 99, (0, 0, 100, 100), dest_dir=tmp_path / "figs") is None
    assert detect_figure_regions(str(tmp_path / "missing.pdf")) == []


def test_analyze_figure_guards(tmp_path: Path, monkeypatch) -> None:
    pdf = _make_paper_pdf(tmp_path / "paper.pdf")
    regions = detect_figure_regions(str(pdf))
    region = next((r for r in regions if r.page == 2), regions[0])
    s = get_settings()

    # vision 未启用 → 直接跳过（不调用 VLM）
    monkeypatch.setitem(s.sources["vision"], "enabled", False)
    assert analyze_figure(str(pdf), region.page, region.bbox, "图的内容？", llm=get_fake_llm("vision")) is None

    # vision 启用但未提供 llm → 跳过
    monkeypatch.setitem(s.sources["vision"], "enabled", True)
    assert analyze_figure(str(pdf), region.page, region.bbox, "图的内容？") is None
    assert analyze_figure(str(pdf), region.page, region.bbox, "图的内容？", llm=None) is None


def test_analyze_figure_with_fake_llm(tmp_path: Path, monkeypatch) -> None:
    pdf = _make_paper_pdf(tmp_path / "paper.pdf")
    s = get_settings()
    monkeypatch.setitem(s.sources["vision"], "enabled", True)
    regions = detect_figure_regions(str(pdf))
    region = next((r for r in regions if r.page == 2), regions[0])

    reading = analyze_figure(
        str(pdf), region.page, region.bbox, "这组矩形表示什么？", llm=get_fake_llm("vision"), figure_no="Figure 1"
    )
    assert reading is not None
    # 元数据强制绑定真实值，不信任模型输出
    assert reading.page == region.page
    assert reading.figure_no == "Figure 1"
    assert Path(reading.figure_path).exists()
