"""tests/test_memory.py：记忆与存储层（EvidenceStore / build_context / SkillStore）。"""
from __future__ import annotations

import re
import threading

import yaml

from src.memory.context_builder import build_context, compress_findings
from src.memory.evidence_store import EvidenceStore
from src.memory.skill_store import Skill, SkillStore
from src.schemas import (
    ClaimRecord,
    ConflictReport,
    EvidenceRecord,
    Finding,
    PaperCard,
    PaperRecord,
    ResearchBrief,
    ResearchTask,
)
from src.utils import estimate_tokens, norm_title


# --------------------------------------------------------------------------
# EvidenceStore
# --------------------------------------------------------------------------
def test_evidence_store_paper_roundtrip_and_find(tmp_path):
    store = EvidenceStore(tmp_path / "ev.sqlite3")
    p = PaperRecord(
        paper_id="p1",
        title="Attention Is All You Need",
        doi="https://doi.org/10.5555/3292049",
        arxiv_id="arXiv:1706.03762",
    )
    store.upsert_paper(p)
    assert store.get_paper("p1") == p
    assert store.get_paper("missing") is None

    # 三种键均可命中（doi/arxiv_id 自动规范化）
    assert store.find_paper(doi="10.5555/3292049").paper_id == "p1"
    assert store.find_paper(doi="https://doi.org/10.5555/3292049").paper_id == "p1"
    assert store.find_paper(arxiv_id="arXiv:1706.03762").paper_id == "p1"
    assert store.find_paper(title_norm=norm_title("Attention Is All You Need")).paper_id == "p1"
    # 组合条件 AND：任一不匹配则不命中
    assert store.find_paper(doi="10.5555/3292049", arxiv_id="9999.9999") is None
    assert store.find_paper() is None
    store.close()


def test_evidence_store_created_at_preserved(tmp_path):
    store = EvidenceStore(tmp_path / "ev.sqlite3")
    first = "2026-01-01T00:00:00+00:00"
    ev = EvidenceRecord(
        evidence_id="ev_fixed0001",
        paper_id="p1",
        task_id="t1",
        claim_hint="hint",
        evidence_text="v1",
        created_at=first,
    )
    store.upsert_evidence(ev)
    store.upsert_evidence(ev.model_copy(update={"created_at": "2026-09-09T00:00:00+00:00", "evidence_text": "v2"}))
    got = store.get_evidence("ev_fixed0001")
    assert got is not None
    assert got.created_at == first  # 保留首次 created_at
    assert got.evidence_text == "v2"  # 其余字段以最新为准
    store.close()


def test_evidence_store_parallel_writes(tmp_path):
    store = EvidenceStore(tmp_path / "par.sqlite3")
    errors: list[Exception] = []

    def worker(w: int) -> None:
        try:
            for i in range(20):
                store.upsert_evidence(
                    EvidenceRecord(
                        evidence_id=f"ev_w{w}_{i:02d}",
                        paper_id=f"p{w}",
                        task_id=f"task_{w}",
                        claim_hint="hint",
                        evidence_text="x" * 50,
                    )
                )
        except Exception as ex:  # noqa: BLE001
            errors.append(ex)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert store.stats()["evidence"] == 8 * 20
    assert len(store.evidence_for_task("task_3")) == 20
    assert len(store.evidence_for_paper("p5")) == 20
    store.close()


def test_evidence_store_task_filter_cards_claims_findings(tmp_path):
    store = EvidenceStore(tmp_path / "mix.sqlite3")
    for t in ("t1", "t2"):
        for i in range(3):
            store.upsert_evidence(
                EvidenceRecord(
                    evidence_id=f"ev_{t}_{i}",
                    paper_id=f"p_{t}",
                    task_id=t,
                    claim_hint="h",
                    evidence_text="text",
                )
            )
    assert {e.task_id for e in store.evidence_for_task("t1")} == {"t1"}
    assert len(store.all_evidence(limit=4)) == 4
    assert len(store.all_evidence()) == 6
    assert store.evidence_by_ids(["ev_t1_0", "nope", "ev_t2_2"])[0].evidence_id == "ev_t1_0"
    assert len(store.evidence_by_ids(["nope"])) == 0

    store.upsert_card(PaperCard(card_id="c1", paper_id="p_t1", method="m"))
    # 同 paper_id 换新 card_id 覆盖旧卡片
    store.upsert_card(PaperCard(card_id="c2", paper_id="p_t1", method="m2"))
    assert store.card_for_paper("p_t1").card_id == "c2"
    assert len(store.all_cards()) == 1

    store.upsert_claim(ClaimRecord(claim_id="cl1", claim_text="claim"))
    assert len(store.all_claims()) == 1

    store.add_finding(Finding(task_id="t1", summary="s1", tool_calls=3))
    store.add_finding(Finding(task_id="t1", summary="s2", tool_calls=1, status="partial"))
    store.add_finding(Finding(task_id="t2", summary="s3", tool_calls=2))
    assert [f.summary for f in store.findings_for("t1")] == ["s1", "s2"]
    assert len(store.findings_for("t2")) == 1
    store.close()


def test_evidence_store_close_and_reopen(tmp_path):
    db = tmp_path / "reopen.sqlite3"
    store = EvidenceStore(db)
    store.upsert_paper(PaperRecord(paper_id="p1", title="T"))
    store.upsert_evidence(EvidenceRecord(evidence_id="ev_1", paper_id="p1", task_id="t", claim_hint="h", evidence_text="x"))
    store.upsert_card(PaperCard(card_id="c1", paper_id="p1"))
    store.upsert_claim(ClaimRecord(claim_id="cl1", claim_text="c"))
    store.add_finding(Finding(task_id="t", summary="s"))
    stats_before = store.stats()
    store.close()
    # close 后操作降级为空结果，不抛异常
    assert store.get_paper("p1") is None
    store.upsert_paper(PaperRecord(paper_id="p2", title="T2"))  # 静默失败

    store2 = EvidenceStore(db)
    assert store2.stats() == stats_before
    assert store2.get_paper("p1").title == "T"
    assert store2.get_evidence("ev_1") is not None
    assert store2.card_for_paper("p1") is not None
    assert store2.all_claims()[0].claim_id == "cl1"
    assert store2.findings_for("t")[0].summary == "s"
    store2.close()


# --------------------------------------------------------------------------
# build_context
# --------------------------------------------------------------------------
def _make_corpus():
    """50 条超长证据 + 10 篇论文（2 篇无 PDF 的 blog）+ 卡片 + 1 个冲突。"""
    papers = [
        PaperRecord(
            paper_id=f"paper_{i:03d}",
            title=f"Alignment Training Study {i}",
            authors=["A", "B"],
            year=2023,
            doi=f"10.1000/x.{i}",
            arxiv_id=f"2401.{10000 + i}",
            source_url=f"https://arxiv.org/abs/2401.{10000 + i}",
            pdf_path=None if i >= 8 else f"/tmp/p{i}.pdf",
            paper_type="blog" if i >= 8 else "full-paper",
        )
        for i in range(10)
    ]
    cards = [
        PaperCard(
            card_id=f"card_{i:03d}",
            paper_id=f"paper_{i:03d}",
            motivation="动机" * 60,
            method="方法描述" * 60,
            datasets=["C4", "OpenWebText"],
            metrics=["loss", "accuracy"],
            results="结论" * 60,
            limitations=["局限一" * 30, "局限二" * 30],
        )
        for i in range(8)  # 仅核心论文有卡片
    ]
    evs = []
    for i in range(50):
        if i == 49:  # 冲突证据：与任务关键词零重叠，靠提权进入上下文
            text = "这是一段与任务无关的烹饪记录文本，用于验证冲突提权。" * 15
            hint = "无关提示"
        else:
            text = f"大模型对齐方法在训练效率上的提升幅度为{i}%，实验设置一致。" * 20
            hint = f"对齐方法效率结论 {i}"
        evs.append(
            EvidenceRecord(
                evidence_id=f"ev_{i:010x}",
                paper_id=f"paper_{i % 8:03d}",
                task_id="task_main",
                claim_hint=hint,
                evidence_text=text,
                page=i % 10 + 1,
                section="4",
                relevance_score=0.8,
            )
        )
    conflict = ConflictReport(
        conflict_id="cf_1",
        topic="对齐训练是否提升效率",
        paper_ids=["paper_000", "paper_001"],
        description="两项研究结论方向相反",
        possible_causes=["数据规模不同"],
        severity="high",
        evidence_ids=["ev_0000000031"],
    )
    task = ResearchTask(task_id="task_main", question="大模型对齐方法对训练效率的影响", perspective="method")
    brief = ResearchBrief(
        objective="梳理对齐方法与训练效率的关系",
        out_of_scope=["强化学习之外的内容"],
        core_questions=["对齐是否拖慢训练", "效率与效果如何权衡"],
        criteria={"效果": "下游任务精度", "成本": "GPU 时"},
    )
    return task, brief, evs, papers, cards, [conflict]


def test_build_context_budget_and_identity_integrity():
    task, brief, evs, papers, cards, conflicts = _make_corpus()
    all_ids = {e.evidence_id for e in evs}

    out = build_context(
        task=task, brief=brief, evidence=evs, papers=papers, cards=cards,
        conflicts=conflicts, token_budget=6000,
    )
    # 六个规定区块 + 线索区
    for section in ("## 研究任务", "## 研究范围", "## 论文卡片（压缩）", "## 关键证据", "## 冲突点", "## 线索", "## 附注"):
        assert section in out, section
    # 预算约束（允许 100 token 余量）
    assert estimate_tokens(out) <= 6000 + 100
    # 引用身份完整：出现的 evidence_id 都是完整 id（未被腰斩）
    matched = re.findall(r"ev_[0-9a-f]+", out)
    assert set(matched) <= all_ids
    assert all(len(m) == len("ev_") + 10 for m in matched)
    # 截断确实发生（50 条不可能全进 6000 预算）
    assert len(set(matched)) < 50
    assert "整条丢弃" in out
    # 冲突证据被提权纳入并排在普通高相关证据之前
    ev_section = out[out.index("## 关键证据"):out.index("## 冲突点")]
    assert "ev_0000000031" in ev_section
    assert ev_section.index("ev_0000000031") < ev_section.index("ev_0000000000")
    # 页码与 paper_id 引用身份完整
    assert re.search(r"\[ev_0000000000\] 论文 paper_000 页码 p\.1", out)
    # 无 PDF 的 blog 论文只在“线索”区出现，不与核心证据混排
    lead_section = out[out.index("## 线索"):]
    card_section = out[out.index("## 论文卡片"):out.index("## 关键证据")]
    assert "paper_008" in lead_section
    assert "paper_008" not in card_section
    assert "paper_009" not in card_section
    # 卡片每篇 ≤240 字符且保留 paper_id
    for line in card_section.splitlines():
        if line.startswith("- [paper_"):
            assert len(line) <= 240
            assert re.match(r"^- \[paper_\d{3}\]", line)


def test_build_context_generous_budget_keeps_all():
    task, brief, evs, papers, cards, conflicts = _make_corpus()
    out = build_context(
        task=task, brief=brief, evidence=evs, papers=papers, cards=cards,
        conflicts=conflicts, token_budget=60000,
    )
    assert estimate_tokens(out) <= 60100
    assert set(re.findall(r"ev_[0-9a-f]+", out)) == {e.evidence_id for e in evs}


def test_build_context_minimal_inputs():
    out = build_context()
    assert "## 附注" in out
    out2 = build_context(task=ResearchTask(question="测试问题"))
    assert "测试问题" in out2
    assert estimate_tokens(out2) <= 24000 + 100


def test_compress_findings():
    findings = [
        Finding(task_id=f"task_{i:02d}", summary="很长的摘要内容" * 30,
                status=["done", "partial", "failed"][i % 3], tool_calls=i)
        for i in range(30)
    ]
    s = compress_findings(findings, max_chars=600)
    assert len(s) <= 600
    assert "task_00" in s
    assert "省略" in s
    lines = [l for l in s.splitlines() if "|" in l]
    parts = lines[0].split(" | ")
    assert parts[0] == "task_00"
    assert parts[1] == "done"
    assert parts[3].startswith("0 次工具调用")
    # 空输入
    assert compress_findings([]) == ""


# --------------------------------------------------------------------------
# SkillStore
# --------------------------------------------------------------------------
def test_skill_store_load_project_skills_and_match():
    store = SkillStore()  # 项目 skills/ 目录
    skills = store.load()
    assert len(skills) >= 3
    names = {s.name for s in skills}
    assert {"survey", "paper_compare", "experiment_design"} <= names
    assert all(s.version == "0.1" for s in skills)

    matched = store.match("帮我比较 SFT 和 DPO 两种方法")
    matched_names = {s.name for s in matched}
    assert "paper_compare" in matched_names
    assert all(s.status == "enabled" for s in matched)
    assert "survey" not in matched_names  # 粗匹配不应误伤无关技能


def test_skill_store_bad_files_skipped(tmp_path):
    sd = tmp_path / "skills"
    sd.mkdir()
    (sd / "good.yaml").write_text(
        yaml.safe_dump({"name": "good", "version": "0.1", "status": "enabled",
                        "trigger": "比较两个及以上方法", "actions": ["a"], "notes": "n"},
                       allow_unicode=True),
        encoding="utf-8",
    )
    (sd / "broken.yaml").write_text("name: [unclosed\n  bad: :", encoding="utf-8")
    (sd / "invalid.yaml").write_text("trigger: 缺少 name 字段", encoding="utf-8")
    store = SkillStore(sd)
    assert [s.name for s in store.load()] == ["good"]
    assert [s.name for s in store.enabled()] == ["good"]
    assert store.candidates() == []
    assert [s.name for s in store.match("帮我比较两个方法")] == ["good"]


def test_skill_store_add_candidate(tmp_path):
    sd = tmp_path / "skills"
    sd.mkdir()
    (sd / "base.yaml").write_text(
        yaml.safe_dump({"name": "base", "status": "enabled", "trigger": "t"}, allow_unicode=True),
        encoding="utf-8",
    )
    store = SkillStore(sd)
    path = store.add_candidate(
        Skill(name="new_skill", trigger="新触发词", actions=["act1"], status="enabled")
    )
    assert path.exists()
    assert path.parent.name == "candidates"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["status"] == "candidate"  # 强制 candidate
    assert data["actions"] == ["act1"]
    # 重开 store 能从 candidates/ 读回候选
    store2 = SkillStore(sd)
    assert "new_skill" in {s.name for s in store2.candidates()}
    assert "new_skill" not in {s.name for s in store2.enabled()}
    # 非法名称拒绝
    try:
        store.add_candidate(Skill(name="../evil", trigger="x"))
        raised = False
    except ValueError:
        raised = True
    assert raised
