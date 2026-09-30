"""测试共享 fixture：合成 PDF、离线 settings 开关。"""

from __future__ import annotations

import pytest


def make_synthetic_pdf(path, title: str, pages: list[list[tuple[str, float]]]) -> None:
    """生成合成论文 PDF。pages: 每页 [(文本, 字号), ...]，字号大者模拟标题。"""
    import pymupdf

    doc = pymupdf.open()
    for page_lines in pages:
        page = doc.new_page()
        y = 72
        for text, size in page_lines:
            page.insert_text((72, y), text, fontsize=size, fontname="helv")
            y += size + 10
    doc.save(str(path))
    doc.close()


@pytest.fixture
def two_synthetic_pdfs(tmp_path):
    """两篇含章节结构与关键词的合成论文。"""
    make_synthetic_pdf(
        tmp_path / "grpo_for_vlm_2024.pdf",
        "GRPO for Vision-Language Models",
        [
            [
                ("GRPO for Vision-Language Models", 16),
                ("1. Introduction", 12),
                ("Reinforcement learning with GRPO improves reasoning of vision-language models.", 10),
                ("2. Method", 12),
                ("We apply group relative policy optimization to VLM post-training.", 10),
            ],
            [
                ("3. Results", 12),
                ("GRPO achieves 62.5 accuracy on MathVista, higher than SFT baseline 55.1.", 10),
                ("4. Limitations", 12),
                ("Reward hacking appears in 12 percent of runs without KL constraint.", 10),
            ],
        ],
    )
    make_synthetic_pdf(
        tmp_path / "dpo_vs_sft_2023.pdf",
        "A Comparative Study of DPO and SFT",
        [
            [
                ("A Comparative Study of DPO and SFT", 16),
                ("Abstract", 12),
                ("Direct preference optimization avoids reward modeling compared to RLHF.", 10),
                ("1. Introduction", 12),
                ("Post-training aligns language models with human preference.", 10),
            ],
            [
                ("References", 12),
                ("Rafailov et al. Direct Preference Optimization. NeurIPS 2023.", 9),
            ],
        ],
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _no_embedding_env(monkeypatch: pytest.MonkeyPatch):
    """全局剥掉 EMBEDDING_* 环境变量：settings 会加载 .env，真实网关配置漏进
    离线测试会让 retrieve_chunks 发网络请求（401 重试拖慢且不稳定）。
    需要向量侧的测试自行 monkeypatch embedding 模块函数。"""
    for var in ("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def offline_sources():
    """临时关闭所有在线数据源（测试后恢复）。"""
    from src.settings import get_settings

    s = get_settings()
    original = {}
    for section in ("arxiv", "semantic_scholar", "crossref"):
        original[section] = dict(s.sources.get(section) or {})
        s.sources.setdefault(section, {})["enabled"] = False
    original["search"] = dict(s.sources.get("search") or {})
    s.sources.setdefault("search", {})["download_pdfs"] = False
    original["web_search"] = dict(s.sources.get("web_search") or {})
    s.sources.setdefault("web_search", {})["enable_for_research"] = False
    original["vision_enabled"] = (s.sources.get("vision") or {}).get("enabled")
    yield s
    for section, cfg in original.items():
        if section == "vision_enabled":
            if original["vision_enabled"] is not None:
                s.sources.setdefault("vision", {})["enabled"] = original["vision_enabled"]
            continue
        if cfg:
            s.sources[section] = cfg
