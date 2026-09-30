"""Writer / Verifier / 三个评测模块的离线单元测试（全部 FakeBackend，无网络、不烧 token）。"""

from __future__ import annotations

import json
import re

import pytest

from src.agents.verifier import build_repair_hints, build_repair_tasks, route_verification, verify
from src.agents.writer import (
    VALID_PERSPECTIVES,
    ReportDraft,
    _strip_duplicated_header,
    generate_outline,
    render_final,
    write_report,
)
from src.eval.agent_eval import evaluate as agent_evaluate
from src.eval.citation_eval import evaluate as citation_evaluate
from src.eval.coverage_eval import evaluate as coverage_evaluate
from src.llm import FakeBackend, get_fake_llm
from src.schemas import (
    ClaimRecord,
    EvidenceRecord,
    MetadataCheck,
    OutlineSection,
    PaperCard,
    PaperRecord,
    ResearchBrief,
    ResearchTask,
    VerificationCheck,
    VerificationReport,
)


# --------------------------------------------------------------------------
# 公共设施
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_fake_registry():
    FakeBackend.reset()
    yield
    FakeBackend.reset()


@pytest.fixture()
def no_crossref(monkeypatch):
    """屏蔽 crossref_tool 真实网络查询：统一返回 unavailable。"""
    import src.tools.crossref_tool as ct

    monkeypatch.setattr(ct, "verify_metadata", lambda paper: MetadataCheck(note="unavailable"))


def _boom(prompt, schema=None):
    raise RuntimeError("fake backend boom")


def make_brief() -> ResearchBrief:
    return ResearchBrief(
        objective="调研 GRPO 算法在数学推理中的遗忘现象",
        core_questions=["GRPO 训练是否出现灾难性遗忘"],
    )


def make_tasks() -> list[ResearchTask]:
    return [
        ResearchTask(question="GRPO 遗忘的实验证据", perspective="experiment"),
        ResearchTask(question="GRPO 方法原理", perspective="method"),
        ResearchTask(question="相关批评与局限", perspective="critique"),
    ]


def make_papers() -> list[PaperRecord]:
    return [
        PaperRecord(paper_id="paperA", title="Forgetting in GRPO Training", authors=["Alice", "Bob"], year=2024, doi="10.1000/a"),
        PaperRecord(paper_id="paperB", title="Stability of RL Fine-tuning", authors=["Carol"], year=2025, doi="10.1000/b"),
        PaperRecord(paper_id="paperC", title="Unrelated Survey", authors=["Dave"], year=2023, doi="10.1000/c"),
    ]


def make_evidence(tasks: list[ResearchTask]) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            paper_id="paperA", task_id=tasks[0].task_id, claim_hint="GRPO 长训练中准确率下降 22%",
            evidence_text="实验显示 GSM8K 准确率从 92% 降至 70%", page=3, relevance_score=0.9,
        ),
        EvidenceRecord(
            paper_id="paperB", task_id=tasks[0].task_id, claim_hint="另一研究认为性能保持稳定",
            evidence_text="在相同设置下未观察到显著退化", page=5, relevance_score=0.7,
        ),
        EvidenceRecord(
            paper_id="paperA", task_id=tasks[1].task_id, claim_hint="GRPO 采用组内相对优势",
            evidence_text="方法章节描述组内归一化", page=7, relevance_score=0.8,
        ),
    ]


# --------------------------------------------------------------------------
# generate_outline
# --------------------------------------------------------------------------
def test_generate_outline_from_llm():
    FakeBackend.register("outline", {"sections": [
        {"title": "研究背景", "key_points": ["问题来源"], "task_ids": [], "perspective": "background"},
        {"title": "方法综述", "key_points": ["方法分类"], "task_ids": [], "perspective": "综述视角"},
        {"title": "实验对比", "key_points": ["结果对比"], "task_ids": [], "perspective": "experiment"},
        {"title": "典型应用", "key_points": [], "task_ids": [], "perspective": "application"},
        {"title": "争议与局限", "key_points": [], "task_ids": [], "perspective": "critique"},
        {"title": "结论", "key_points": [], "task_ids": [], "perspective": "critique"},
    ]})
    tasks = make_tasks()
    sections = generate_outline(make_brief(), tasks, make_evidence(tasks), make_papers(), llm=get_fake_llm("orchestrator"))
    assert 4 <= len(sections) <= 8
    assert all(s.perspective in VALID_PERSPECTIVES for s in sections)
    assert {s.perspective for s in sections} >= {"background", "method", "experiment", "critique"}
    assert any(s.evidence_ids for s in sections)  # 小节挂上了相关证据
    assert any("GRPO" in s.title for s in sections) is False or True  # 标题不做硬约束


def test_generate_outline_clamps_to_eight():
    FakeBackend.register("outline", {"sections": [
        {"title": f"章节 {i}", "perspective": "background"} for i in range(10)
    ]})
    tasks = make_tasks()
    sections = generate_outline(make_brief(), tasks, make_evidence(tasks), make_papers(), llm=get_fake_llm("orchestrator"))
    assert len(sections) == 8


def test_generate_outline_fallback_template():
    FakeBackend.register("outline", _boom)
    tasks = make_tasks()
    sections = generate_outline(make_brief(), tasks, make_evidence(tasks), make_papers(), llm=get_fake_llm("orchestrator"))
    assert len(sections) == 5
    joined = " ".join(s.title for s in sections)
    for kw in ("背景", "方法", "实验", "争议", "结论"):
        assert kw in joined
    assert "GRPO" in joined  # 标题插入 objective 关键词
    assert all(s.perspective in VALID_PERSPECTIVES for s in sections)


# --------------------------------------------------------------------------
# write_report
# --------------------------------------------------------------------------
def test_write_report_with_fake_llm():
    FakeBackend.register(
        "writer",
        "GRPO 长训练中准确率下降 22% [paperA:3]，但另一研究认为保持稳定 [paperB:5]。",
    )
    FakeBackend.register("claim_extract", {"claims": [
        {"claim_text": "GRPO 长训练中准确率下降 22%", "citation_keys": ["[paperA:3]"], "evidence_ids": [], "section": "实验对比", "confidence": 0.9},
        {"claim_text": "另一研究认为性能保持稳定", "citation_keys": ["[paperB:5]"], "evidence_ids": ["不存在的id"], "section": "实验对比", "confidence": 0.7},
    ]})
    brief = make_brief()
    evs = make_evidence(make_tasks())
    outline = [OutlineSection(title="实验对比", key_points=["结果对比"], perspective="experiment",
                              evidence_ids=[evs[0].evidence_id, evs[1].evidence_id])]
    cards = [PaperCard(paper_id="paperA", method="GRPO", results="准确率下降", evidence_ids=[evs[0].evidence_id])]
    draft = write_report(brief, outline, evs, make_papers(), cards, llm=get_fake_llm("writer"))

    assert isinstance(draft, ReportDraft)
    assert "[paperA:3]" in draft.markdown and "## 实验对比" in draft.markdown
    assert len(draft.claims) == 2
    c0, c1 = draft.claims
    assert c0.citation_keys == ["[paperA:3]"]
    assert c0.evidence_ids == [evs[0].evidence_id]  # 引用键回填证据
    assert c0.section == "实验对比"
    assert c0.status == "supported"
    assert c1.evidence_ids == [evs[1].evidence_id]  # 无效证据 id 被替换为键对应证据
    assert c1.citation_keys == ["[paperB:5]"]


def test_write_report_fallback_never_empty():
    FakeBackend.register("writer", _boom)
    FakeBackend.register("claim_extract", _boom)
    brief = make_brief()
    evs = make_evidence(make_tasks())
    outline = [OutlineSection(title="实验对比", perspective="experiment",
                              evidence_ids=[evs[0].evidence_id, evs[1].evidence_id])]
    cards = [PaperCard(paper_id="paperA", method="GRPO", results="准确率下降")]
    draft = write_report(brief, outline, evs, make_papers(), cards, llm=get_fake_llm("writer"))

    assert draft.markdown.strip()
    assert "[paperA:3]" in draft.markdown  # 兜底事实清单仍带内部引用键
    assert draft.claims and draft.claims[0].evidence_ids
    assert draft.claims[0].citation_keys


def test_stub_hints_never_become_claims():
    """Reader 失败桩 / VLM 图表解读提示不得被抽成结论（污染核验与修复轮）。"""
    tasks = make_tasks()
    evs = make_evidence(tasks)
    stub_figure = EvidenceRecord(
        paper_id="paperA", task_id=tasks[0].task_id, claim_hint="图表证据提示（figure，第 27 页）",
        evidence_text="[VLM 解读，须对照原文] 柱状图…", page=27, modality="figure",
    )
    evs = evs + [stub_figure]
    outline = [OutlineSection(title="实验对比", perspective="experiment",
                              evidence_ids=[evs[0].evidence_id, stub_figure.evidence_id])]

    # 路径一：LLM 抽取器把桩文本当结论返回 → _post_claims 过滤
    FakeBackend.register("writer", _boom)
    FakeBackend.register(
        "claim_extract",
        {"claims": [
            {"claim_text": "（LLM 失败，自动保留的相关片段）", "citation_keys": ["[paperA:3]"]},
            {"claim_text": "GRPO 长训练中准确率下降 22%", "citation_keys": ["[paperA:3]"]},
            {"claim_text": "图表证据提示（figure，第 27 页）", "citation_keys": ["[paperA:27]"]},
        ]},
    )
    draft = write_report(make_brief(), outline, evs, make_papers(), [], llm=get_fake_llm("writer"))
    assert [c.claim_text for c in draft.claims] == ["GRPO 长训练中准确率下降 22%"]

    # 路径二：抽取器也失败 → 规则兜底按引用键取 claim_hint → 桩 hint 跳过
    FakeBackend.register("claim_extract", _boom)
    draft2 = write_report(make_brief(), outline, evs, make_papers(), [], llm=get_fake_llm("writer"))
    texts = [c.claim_text for c in draft2.claims]
    assert texts and not any(t.startswith("（LLM 失败") or t.startswith("图表证据提示") for t in texts)


# --------------------------------------------------------------------------
# render_final
# --------------------------------------------------------------------------
def test_render_final_numbering_and_refs():
    md = "结论一 [paperA:3]；结论二复引 [paperA]；结论三 [paperB:5]。"
    out = render_final(md, [], make_papers())
    assert "[1]" in out and "[2]" in out
    body, _, refs = out.partition("## 参考文献")
    assert "paperA" not in body and "paperB" not in body  # 正文内部键全部转编号
    assert "Forgetting in GRPO Training" in refs  # 被引论文进参考文献
    assert "Stability of RL Fine-tuning" in refs
    assert "Unrelated Survey" not in refs  # 未引用论文不进参考文献


def test_render_final_fallback_when_tool_unavailable(monkeypatch):
    import src.tools.citation_tool as ct

    monkeypatch.setattr(ct, "render_references", lambda papers, **kw: (_ for _ in ()).throw(NotImplementedError("pending")))
    out = render_final("引用 [paperA:3]。", [], make_papers())
    assert "## 参考文献" in out
    assert "10.1000/a" in out  # 内联兜底格式：作者. 题名. 年份. doi


def test_render_final_uses_citation_tool_when_available(monkeypatch):
    import src.tools.citation_tool as ct

    monkeypatch.setattr(ct, "render_references", lambda papers, **kw: "TOOL-REFS:" + ",".join(p.paper_id for p in papers))
    out = render_final("引用 [paperA:3] 与 [paperB]。", [], make_papers())
    assert "TOOL-REFS:paperA,paperB" in out


def test_render_final_colon_paper_id():
    """本地 PDF 的 paper_id 形如 sha256:<hash>，本身含冒号；页码是结尾 ':数字]'。"""
    papers = [PaperRecord(paper_id="sha256:92cb3a2b7136", title="Direct Preference Optimization", year=2023)]
    md = "结论一 [sha256:92cb3a2b7136:9]；结论二复引 [sha256:92cb3a2b7136]。"
    out = render_final(md, [], papers)
    body, _, refs = out.partition("## 参考文献")
    assert "[1]" in body and "[2]" not in body  # 同一论文两处引用共用编号 [1]
    assert "sha256" not in body  # 裸键不应残留
    assert "Direct Preference Optimization" in refs
    # 正文无键时用 claims 的键兜底编号
    claim = ClaimRecord(claim_text="DPO 获 58% 人工偏好率", citation_keys=["[sha256:92cb3a2b7136:9]"])
    out2 = render_final("没有任何引用键的正文。", [claim], papers)
    assert "Direct Preference Optimization" in out2.partition("## 参考文献")[2]


def test_strip_duplicated_header():
    body = "## 训练机制对比\n\n正文第一段。"
    assert _strip_duplicated_header(body, "训练机制对比") == "正文第一段。"
    assert _strip_duplicated_header(body, "训练机制对比") != body
    # 前导空行 + 完全相同的标题（含井号层级差异）
    assert _strip_duplicated_header("\n\n### 训练机制对比\n\n正文。", "训练机制对比") == "正文。"
    # 标题不同 / 正文以其他标题开头 → 原样保留
    assert _strip_duplicated_header("## 另一个标题\n\n正文。", "训练机制对比") == "## 另一个标题\n\n正文。"
    assert _strip_duplicated_header("正文直接开始。", "训练机制对比") == "正文直接开始。"


def test_coverage_recognizes_colon_citation_keys(no_crossref):
    """覆盖率核验须把 [sha256:xxx:9] 视为已带引用（否则误报 citation_missing）。"""
    from src.agents.verifier import _has_citation_key

    assert _has_citation_key("DPO 在人工评测中获 58% 偏好率 [sha256:92cb3a2b7136:9]。") is True
    assert _has_citation_key("DPO 在人工评测中获 58% 偏好率 [9]。") is False
    assert _has_citation_key("参见 [论文](https://example.com) 的讨论。") is False
    # 渲染后的 [编号:页码] 形态是正文引用（曾误报"缺引用"的回归防线）
    assert _has_citation_key("PPO 对 λ 系数高度敏感 [1:14]。") is True
    assert _has_citation_key("GRPO 去除了价值模型 [2:27]。") is True


# --------------------------------------------------------------------------
# verify：四层核验
# --------------------------------------------------------------------------
def _judge_respond(prompt, schema=None):
    if "GOOD" in prompt:
        return {"label": "entailed", "reason": "证据充分"}
    if "CONTRA" in prompt:
        return {"label": "contradicted", "reason": "证据相反"}
    if "PARTIAL" in prompt:
        return {"label": "partial", "reason": "部分支撑", "missing_information": "缺少跨数据集验证"}
    return {"label": "entailed", "reason": "默认"}


def _consistency_respond(prompt, schema=None):
    pairs = re.findall(r"\[(claim_[0-9a-f]+)\]\s*(.+)", prompt)
    contra = [cid for cid, text in pairs if "CONTRA" in text]
    if len(contra) >= 2:
        return {"conflicts": [{"claim_ids": contra, "description": "两结论方向相反"}]}
    return {"conflicts": []}


def test_verify_four_layers(no_crossref):
    tasks = make_tasks()
    evs = make_evidence(tasks)
    papers = make_papers() + [PaperRecord(paper_id="paperBad", title="Fake DOI Paper", doi="10.9999/fake")]
    ev_bad = EvidenceRecord(paper_id="paperBad", task_id=tasks[2].task_id, claim_hint="伪 DOI 论文结论",
                            evidence_text="fake content", page=1, relevance_score=0.5)
    ev_a = EvidenceRecord(paper_id="paperA", task_id=tasks[0].task_id, claim_hint="CONTRA_A：方法 X 优于 Y",
                          evidence_text="X 优于 Y", page=9)
    ev_b = EvidenceRecord(paper_id="paperB", task_id=tasks[0].task_id, claim_hint="CONTRA_B：方法 X 不优于 Y",
                          evidence_text="X 不优于 Y", page=2)
    evs = evs + [ev_bad, ev_a, ev_b]

    claims = [
        ClaimRecord(claim_text="GOOD：GRPO 长训练中准确率下降 22%", evidence_ids=[evs[0].evidence_id],
                    citation_keys=["[paperA:3]"], section="实验对比", confidence=0.9),
        ClaimRecord(claim_text="无证据结论，准确率 99%", evidence_ids=[], citation_keys=["[paperA:3]"]),
        ClaimRecord(claim_text="META：引用伪 DOI 论文的结论", evidence_ids=[ev_bad.evidence_id],
                    citation_keys=["[paperBad:1]"]),
        ClaimRecord(claim_text="CONTRA_A：方法 X 优于 Y", evidence_ids=[ev_a.evidence_id], confidence=0.8),
        ClaimRecord(claim_text="CONTRA_B：方法 X 不优于 Y", evidence_ids=[ev_b.evidence_id], confidence=0.8),
        ClaimRecord(claim_text="PARTIAL：GRPO 在部分数据集上不稳定", evidence_ids=[evs[1].evidence_id],
                    confidence=0.6),
    ]
    FakeBackend.register("judge", _judge_respond)
    FakeBackend.register("consistency", _consistency_respond)
    report_md = "好的句子带引用 [paperA:3]。坏句子没有引用但含数字 88%。"

    report = verify(claims, evs, papers, report_md, llm=get_fake_llm("judge"),
                    metadata_checks={"paperBad": MetadataCheck(doi_valid=False)})

    by_text = {c.claim_text: c for c in claims}
    # L2：好证据 → supported
    assert report.claim_status[by_text["GOOD：GRPO 长训练中准确率下降 22%"].claim_id] == "supported"
    # 无证据 → unsupported
    assert report.claim_status[by_text["无证据结论，准确率 99%"].claim_id] == "unsupported"
    assert by_text["无证据结论，准确率 99%"].claim_id in report.unsupported_claim_ids
    # L1：伪 DOI → metadata_error
    assert report.claim_status[by_text["META：引用伪 DOI 论文的结论"].claim_id] == "metadata_error"
    assert by_text["META：引用伪 DOI 论文的结论"].claim_id in report.metadata_error_claim_ids
    # L2+L4：矛盾 → conflicted（并列呈现，不触发修复）
    assert report.claim_status[by_text["CONTRA_A：方法 X 优于 Y"].claim_id] == "conflicted"
    assert report.claim_status[by_text["CONTRA_B：方法 X 不优于 Y"].claim_id] == "conflicted"
    assert report.conflicts
    # partial → supported + note
    partial = by_text["PARTIAL：GRPO 在部分数据集上不稳定"]
    assert report.claim_status[partial.claim_id] == "supported"
    assert partial.verifier_note and "partial" in partial.verifier_note
    # L3：数字句无引用 → coverage_missing
    assert report.coverage_missing == ["坏句子没有引用但含数字 88%"]
    layers = {c.layer for c in report.checks}
    assert layers >= {"metadata", "entailment", "coverage", "consistency"}
    assert report.passed is False
    assert report.summary


def test_verify_all_supported_passes(no_crossref):
    tasks = make_tasks()
    evs = make_evidence(tasks)
    claim = ClaimRecord(claim_text="GOOD：GRPO 长训练中准确率下降 22%", evidence_ids=[evs[0].evidence_id],
                        citation_keys=["[paperA:3]"])
    FakeBackend.register("judge", _judge_respond)
    report = verify([claim], evs, make_papers(), "结论句带引用 [paperA:3]。", llm=get_fake_llm("judge"))
    assert report.claim_status[claim.claim_id] == "supported"
    assert report.coverage_missing == []
    assert report.passed is True


def test_norm_label_aliases():
    """judge 标签归一：中文/大小写/空白/标点变体不误杀；未知值返回 unresolved。"""
    from src.agents.verifier import _norm_label

    assert _norm_label("支持") == "entailed"
    assert _norm_label("被支撑。") == "entailed"
    assert _norm_label("  Entailed ") == "entailed"
    assert _norm_label("supported") == "entailed"
    assert _norm_label("部分支持") == "partial"
    assert _norm_label("矛盾") == "contradicted"
    assert _norm_label("无关") == "unrelated"
    assert _norm_label("完全正确") == "unresolved"  # 未知 → 待定而非误杀


def test_verify_chinese_judge_label_not_killed(no_crossref):
    """deepseek-flash 实测回 label="支持"（理由正确）：旧版严格英文匹配把它误杀成 unsupported。"""
    tasks = make_tasks()
    evs = make_evidence(tasks)
    claim = ClaimRecord(claim_text="GOOD：GRPO 长训练中准确率下降 22%", evidence_ids=[evs[0].evidence_id],
                        citation_keys=["[paperA:3]"])
    FakeBackend.register("judge", lambda prompt, schema=None: {"label": "支持", "reason": "与结论完全一致"})
    report = verify([claim], evs, make_papers(), "结论句带引用 [paperA:3]。", llm=get_fake_llm("judge"))
    assert report.claim_status[claim.claim_id] == "supported"
    assert report.passed is True


def test_verify_unknown_judge_label_stays_pending(no_crossref):
    """无法识别的标签按待定（partial 语义）处理：不改状态、记 note，不触发 unsupported。"""
    tasks = make_tasks()
    evs = make_evidence(tasks)
    claim = ClaimRecord(claim_text="GOOD：GRPO 长训练中准确率下降 22%", evidence_ids=[evs[0].evidence_id],
                        citation_keys=["[paperA:3]"])
    FakeBackend.register("judge", lambda prompt, schema=None: {"label": "完全正确", "reason": "证据吻合"})
    report = verify([claim], evs, make_papers(), "结论句带引用 [paperA:3]。", llm=get_fake_llm("judge"))
    assert report.claim_status[claim.claim_id] == "supported"  # 待定：维持 supported 并记待定说明
    assert report.unsupported_claim_ids == []
    assert claim.verifier_note and "无法解析" in claim.verifier_note


def test_verify_retracted_paper_is_metadata_error(no_crossref):
    tasks = make_tasks()
    evs = make_evidence(tasks)
    claim = ClaimRecord(claim_text="GOOD：GRPO 长训练中准确率下降 22%", evidence_ids=[evs[0].evidence_id])
    FakeBackend.register("judge", _judge_respond)
    report = verify([claim], evs, make_papers(), "句 [paperA:3]。", llm=get_fake_llm("judge"),
                    metadata_checks={"paperA": MetadataCheck(is_retracted=True)})
    assert report.claim_status[claim.claim_id] == "metadata_error"
    assert report.passed is False


def test_verify_judge_failure_degrades(no_crossref):
    tasks = make_tasks()
    evs = make_evidence(tasks)
    claim = ClaimRecord(claim_text="GOOD：GRPO 长训练中准确率下降 22%", evidence_ids=[evs[0].evidence_id])
    FakeBackend.register("judge", _boom)
    report = verify([claim], evs, make_papers(), "句 [paperA:3]。", llm=get_fake_llm("judge"))
    assert report is not None  # 不崩
    assert report.claim_status[claim.claim_id] == "supported"  # 维持默认，不误杀
    assert claim.verifier_note and "待定" in claim.verifier_note
    assert any(c.layer == "entailment" and c.label == "partial" for c in report.checks)


def test_verify_coverage_llm_assist_failure_falls_back_to_rules(no_crossref):
    sentences = ["这是一句不含任何量化信息的纯过渡表述" for _ in range(20)] + [
        f"数字句{i}包含 12% 但没有引用" for i in range(15)
    ] + ["带 DOI 引用键的数字句 30% [10.1000/a:5] 不应缺失"]
    report_md = "\n".join(sentences)
    FakeBackend.register("judge", _judge_respond)  # coverage 标记不注册 → fake 返回空 results → 退规则
    report = verify([], [], [], report_md, llm=get_fake_llm("judge"))
    assert len(report.coverage_missing) == 15
    assert all("[10.1000/a:5]" not in s for s in report.coverage_missing)  # DOI 形态引用键有效
    assert report.passed is False


# --------------------------------------------------------------------------
# 修复路由
# --------------------------------------------------------------------------
def test_route_verification():
    assert route_verification(VerificationReport(), repair_round=0) == "finalize"
    with_unsupported = VerificationReport(unsupported_claim_ids=["c1"])
    assert route_verification(with_unsupported, repair_round=0) == "repair"
    assert route_verification(with_unsupported, repair_round=2) == "finalize"  # 轮次已满
    with_conflict_only = VerificationReport(conflicts=["矛盾"], claim_status={"a": "conflicted"})
    assert route_verification(with_conflict_only, repair_round=0) == "finalize"  # conflicted 不触发修复
    with_coverage = VerificationReport(coverage_missing=["句子"])
    assert route_verification(with_coverage, repair_round=1, max_repair_rounds=2) == "repair"


def test_build_repair_tasks_and_hints():
    claims = [
        ClaimRecord(claim_id="c1", claim_text="无证据的结论数字 50%", section="实验对比"),
        ClaimRecord(claim_id="c2", claim_text="伪 DOI 结论", section="研究背景"),
    ]
    report = VerificationReport(
        unsupported_claim_ids=["c1"],
        metadata_error_claim_ids=["c2"],
        checks=[
            VerificationCheck(claim_id="c2", layer="metadata", label="metadata_error", reason="DOI 不存在"),
            VerificationCheck(claim_id="c1", layer="entailment", label="unrelated", reason="无证据"),
        ],
        coverage_missing=["需要引用的事实句 99%"],
    )
    tasks = build_repair_tasks(report, claims, max_tasks=3)
    assert len(tasks) == 1  # 仅 unsupported 生成检索任务；metadata_error / citation_missing 不生成
    assert tasks[0].origin == "repair"
    assert tasks[0].perspective == "critique"
    assert "定向补充证据" in tasks[0].question
    assert "无证据的结论" in tasks[0].question
    assert tasks[0].required_evidence == ["直接实验证据"]

    hints = build_repair_hints(report, claims)
    assert any("删除或降级" in h and "无证据" in h for h in hints)
    assert any("元数据" in h for h in hints)
    assert any("重绑引用" in h for h in hints)
    assert all(h.startswith("【") for h in hints)
    assert any(h.startswith("【实验对比】") for h in hints)  # 指令路由到对应章节


# --------------------------------------------------------------------------
# 评测模块
# --------------------------------------------------------------------------
def test_citation_eval_metrics():
    audit = {"claims": [
        {"claim_id": "a", "claim_text": "多智能体在合作任务中具有优势", "status": "supported", "confidence": 0.9},
        {"claim_id": "b", "claim_text": "单智能体在小任务上更稳", "status": "supported", "confidence": 0.7},
        {"claim_id": "c", "claim_text": "无证据结论", "status": "unsupported", "confidence": 0.5},
        {"claim_id": "d", "claim_text": "伪引用结论", "status": "metadata_error", "confidence": 0.5},
        {"claim_id": "e", "claim_text": "相互矛盾结论一", "status": "conflicted", "confidence": 0.6},
    ]}
    m = citation_evaluate(audit)
    assert m["n_claims"] == 5
    assert m["citation_precision"] == pytest.approx(0.5)  # 2 supported / (2+1+1)，conflicted 不进分母
    assert m["conflicted_rate"] == pytest.approx(0.2)
    assert m["avg_confidence"] == pytest.approx(0.64)

    gold = {"key_points": ["多智能体在合作任务中具有优势", "完全无关的另一个主题"]}
    m2 = citation_evaluate(audit, gold=gold)
    assert m2["key_points_covered"] == 1
    assert m2["key_point_coverage"] == pytest.approx(0.5)


def test_coverage_eval_metrics():
    report_md = (
        "## 对比了 GRPO 与 PPO 的遗忘现象\n\n"
        "本节对比了 GRPO 与 PPO 的遗忘现象，结果显示两者存在差异 [paperA:3]。\n"
    )
    gold = {"key_points": ["对比了 GRPO 与 PPO 的遗忘现象", "报告了撤稿论文的处理"], "must_cite_papers": ["paperA", "paperZ"]}
    m = coverage_evaluate(report_md, gold)
    assert m["key_point_coverage"] == pytest.approx(0.5)
    assert m["missing_key_points"] == ["报告了撤稿论文的处理"]
    assert m["must_cite_coverage"] == pytest.approx(0.5)
    assert m["missing_must_cite"] == ["paperZ"]


def _write_trace(tmp_path, events):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")
    return str(path)


def test_agent_eval_metrics(tmp_path):
    events = [
        {"kind": "run_start", "run_id": "r1"},
        {"kind": "llm_call", "role": "writer", "input_tokens": 100, "output_tokens": 50, "duration_ms": 10},
        {"kind": "llm_call", "role": "judge", "input_tokens": 200, "output_tokens": 80, "duration_ms": 20},
        {"kind": "tool_call", "node": "researcher", "tool": "search_arxiv", "args": {"query": "grpo forgetting analysis"}, "ok": True, "duration_ms": 100},
        {"kind": "tool_call", "node": "researcher", "tool": "search_arxiv", "args": {"query": "grpo forgetting analysis 2024"}, "ok": True, "duration_ms": 90},
        {"kind": "tool_call", "node": "researcher", "tool": "search_semantic_scholar", "args": {"query": "totally different topic"}, "ok": True, "duration_ms": 80},
        {"kind": "tool_call", "node": "researcher", "tool": "read_pdf", "args": {}, "ok": False, "error": "parse failed", "duration_ms": 50},
        {"kind": "node_end", "node": "writer", "duration_ms": 500},
        {"kind": "run_end", "status": "finalized"},
    ]
    m = agent_evaluate(_write_trace(tmp_path, events))
    assert m["n_events"] == len(events)
    assert m["llm_calls"] == 2
    assert m["llm_input_tokens"] == 300
    assert m["llm_output_tokens"] == 130
    assert m["tool_calls"] == 4
    assert m["tool_error_rate"] == pytest.approx(0.25)
    assert m["repeated_search_rate"] == pytest.approx(0.5)  # 前两个查询相似度 > 0.85
    assert m["stop_correctness"] is True
    assert m["node_avg_duration_ms"]["writer"] == 500.0
    assert m["node_avg_duration_ms"]["researcher"] == 80.0


def test_agent_eval_budget_exhausted(tmp_path):
    events = [
        {"kind": "llm_call", "role": "writer", "input_tokens": 10, "output_tokens": 5},
        {"kind": "budget_exceeded"},
    ]
    m = agent_evaluate(_write_trace(tmp_path, events))
    assert m["budget_exhausted"] is True
    assert m["stop_correctness"] is False
