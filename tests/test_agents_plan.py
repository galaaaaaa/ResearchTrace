"""规划与治理层离线测试：scope_agent / perspective_planner / supervisor / gap_analyzer。

全部离线：真实 LLM 路径用 get_fake_llm + FakeBackend.register（#FAKE:<name> 标记），
失败路径用 BoomLLM（duck-typing，chat_json 抛 JSONValidationError）。
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from rapidfuzz import fuzz

from src.agents.gap_analyzer import analyze_gaps
from src.agents.perspective_planner import run_perspective_planner
from src.agents.scope_agent import run_scope
from src.agents.supervisor import Supervisor, detect_stagnation
from src.llm import FakeBackend, JSONValidationError, get_fake_llm
from src.schemas import EvidenceRecord, Finding, PaperRecord, ResearchBrief, ResearchTask

ALL_PERSPECTIVES = {"background", "method", "experiment", "application", "critique"}


@pytest.fixture(autouse=True)
def _reset_fake_backend():
    """每个用例前后清空 FakeBackend 注册，避免跨用例串扰。"""
    FakeBackend.reset()
    yield
    FakeBackend.reset()


class BoomLLM:
    """chat_json 一律抛 JSONValidationError 的假客户端（duck-typing）。"""

    def chat_json(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise JSONValidationError("mock 解析失败", raw_text="not-json", errors="boom")


def _brief() -> ResearchBrief:
    return ResearchBrief(
        objective="VLM 后训练方法综述",
        core_questions=["SFT、DPO、GRPO 各解决什么问题"],
        year_from=2021,
        year_to=2026,
        time_range="2021-2026",
    )


# --------------------------------------------------------------------------
# Scope Agent
# --------------------------------------------------------------------------
def test_scope_fake_llm_full_fields() -> None:
    FakeBackend.register(
        "scope",
        {
            "objective": "VLM 后训练方法综述",
            "out_of_scope": ["纯预训练架构搜索"],
            "time_range": "2021-2026",
            "year_from": 2021,
            "year_to": 2026,
            "disciplines": ["多模态机器学习"],
            "paper_types": ["survey", "full-paper"],
            "core_questions": ["SFT/DPO/GRPO 各解决什么问题", "评测口径是否一致"],
            "criteria": {"效果": "与基线相比的提升幅度"},
            "clarification_needed": False,
            "assumptions": ["用户指定了时间范围"],
        },
    )
    brief = run_scope("VLM 后训练方法综述", llm=get_fake_llm())
    assert brief.objective == "VLM 后训练方法综述"
    assert brief.out_of_scope == ["纯预训练架构搜索"]
    assert (brief.year_from, brief.year_to) == (2021, 2026)
    assert brief.time_range == "2021-2026"
    assert brief.disciplines == ["多模态机器学习"]
    assert len(brief.core_questions) == 2
    assert brief.criteria == {"效果": "与基线相比的提升幅度"}
    assert brief.clarification_needed is False
    # LLM 已给出年份与判断标准 → 代码不应再追加默认假设
    assert not any("默认近 5 年" in a for a in brief.assumptions)


def test_scope_year_defaults_and_papers_dir(tmp_path) -> None:
    FakeBackend.register("scope", {"objective": "GRPO 遗忘问题"})
    brief = run_scope("GRPO 遗忘问题", papers_dir=str(tmp_path), llm=get_fake_llm())
    now_year = datetime.now().year
    assert brief.year_to == now_year
    assert brief.year_from == now_year - 5
    assert brief.time_range == f"{now_year - 5}-{now_year}"
    assert brief.papers_dir == str(tmp_path)
    assert "local" in brief.paper_types
    assert any("默认近 5 年" in a for a in brief.assumptions)
    # criteria 缺省 → 效果/成本/数据规模/可复现性
    assert set(brief.criteria) >= {"效果", "成本", "数据规模", "可复现性"}


def test_scope_llm_failure_fallback() -> None:
    brief = run_scope("量子纠码综述", llm=BoomLLM())
    assert brief.objective == "量子纠码综述"
    assert brief.core_questions == ["量子纠码综述"]
    assert any("scope 降级" in a for a in brief.assumptions)
    assert brief.clarification_needed is False
    assert brief.year_to == datetime.now().year
    assert brief.year_from == brief.year_to - 5


# --------------------------------------------------------------------------
# Perspective Planner
# --------------------------------------------------------------------------
_PLANNER_RAW_TASKS = [
    {
        "question": "SFT、DPO、GRPO 分别解决什么问题，机制差异是什么",
        "perspective": "method",
        "required_evidence": ["方法机制原文"],
        "success_criteria": "能比较三种方法",
    },
    {"question": "主流方法在哪些基准上评测，口径是否一致", "perspective": "experiment"},
    {"question": "VLM 后训练的发展脉络与代表工作", "perspective": "背景"},
    {"question": "真实场景如何落地 VLM 后训练", "perspective": "application", "success_criteria": ""},
    {"question": "主流结论存在哪些反例或无消融归因", "perspective": "critique"},
    # 与第 1 条语义重复 → 应被去重（丢后出现的）
    {
        "question": "SFT、DPO、GRPO 分别解决什么问题，机制上的差异是什么",
        "perspective": "method",
    },
    # perspective 非法（空串）→ 按“基准/评测”关键词猜为 experiment
    {"question": "reward hacking 相关的评测基准有哪些", "perspective": ""},
]


def test_planner_dedup_normalize_and_seed_prompt() -> None:
    captured: dict[str, str] = {}

    def _respond(prompt: str, schema) -> str:  # noqa: ANN001
        captured["prompt"] = prompt
        return json.dumps({"dimensions": ["训练范式", "评测"], "tasks": _PLANNER_RAW_TASKS}, ensure_ascii=False)

    FakeBackend.register("planner", _respond)
    seeds = [
        PaperRecord(paper_id="s1", title="A Survey of VLM Post-Training", year=2024, paper_type="survey"),
        PaperRecord(paper_id="s2", title="Vision Language Model Fine-tuning Review", year=2023),
    ]
    tasks = run_perspective_planner(_brief(), llm=get_fake_llm(), seed_papers=seeds, max_tasks=8)

    # 种子论文题名进入了 LLM prompt
    assert "A Survey of VLM Post-Training" in captured["prompt"]
    # 7 个原始任务去重掉 1 个 → 6 个
    assert len(tasks) == 6
    questions = [t.question for t in tasks]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            assert fuzz.token_set_ratio(questions[i], questions[j]) / 100.0 <= 0.8
    # 视角归一化：中文别名“背景”→ background；空串按关键词猜 → experiment
    by_question = {t.question: t for t in tasks}
    assert by_question["VLM 后训练的发展脉络与代表工作"].perspective == "background"
    assert by_question["reward hacking 相关的评测基准有哪些"].perspective == "experiment"
    # 五视角全覆盖（无需补模板）
    assert {t.perspective for t in tasks} == ALL_PERSPECTIVES
    # required_evidence / success_criteria 缺省按 perspective 模板补全
    exp = by_question["主流方法在哪些基准上评测，口径是否一致"]
    assert exp.required_evidence and exp.success_criteria
    # 任务元数据
    assert all(t.origin == "planner" and t.round_created == 0 for t in tasks)
    assert all(t.max_tool_calls == 5 for t in tasks)  # budgets.max_tool_calls_per_researcher
    assert all("search_arxiv" in t.allowed_tools for t in tasks)

    # 截断到 max_tasks
    truncated = run_perspective_planner(_brief(), llm=get_fake_llm(), seed_papers=seeds, max_tasks=4)
    assert len(truncated) == 4
    assert [t.question for t in truncated] == questions[:4]


def test_planner_perspective_completion() -> None:
    FakeBackend.register(
        "planner",
        {
            "dimensions": [],
            "tasks": [
                {"question": "方法甲与方法乙怎么对比", "perspective": "method"},
                {"question": "基准评测结果如何", "perspective": "experiment"},
            ],
        },
    )
    tasks = run_perspective_planner(_brief(), llm=get_fake_llm(), max_tasks=8, online=False)
    assert {t.perspective for t in tasks} == ALL_PERSPECTIVES
    # 补全的模板问题以 objective 填充
    critique = next(t for t in tasks if t.perspective == "critique")
    assert "VLM 后训练方法综述" in critique.question
    assert any("反例" in t.question for t in tasks if t.perspective == "critique")
    assert all(t.required_evidence and t.success_criteria for t in tasks)


def test_planner_llm_failure_fallback(monkeypatch) -> None:
    # online=True 且 seed_papers=None → 种子检索工具不可用（NotImplementedError）应被吞掉，流程继续
    import src.tools.arxiv_tool as arxiv_mod
    import src.tools.semantic_scholar_tool as s2_mod

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError("pending implementation")

    monkeypatch.setattr(arxiv_mod, "search_arxiv", _boom)
    monkeypatch.setattr(s2_mod, "search_semantic_scholar", _boom)
    tasks = run_perspective_planner(_brief(), llm=BoomLLM(), seed_papers=None, online=True)
    assert len(tasks) == 5
    assert {t.perspective for t in tasks} == ALL_PERSPECTIVES
    assert all("VLM 后训练方法综述" in t.question for t in tasks)
    assert all(t.origin == "planner" and t.round_created == 0 for t in tasks)
    assert all(t.required_evidence and t.success_criteria for t in tasks)


# --------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------
def _make_tasks(n: int = 10) -> list[ResearchTask]:
    cycle = ["background", "method", "experiment", "application", "critique"]
    return [ResearchTask(question=f"Q{i}", perspective=cycle[i % 5]) for i in range(n)]


def test_supervisor_next_batch() -> None:
    sup = Supervisor()
    tasks = _make_tasks(10)
    findings = [Finding(task_id=tasks[i].task_id, status="done") for i in range(3)]
    batch = sup.next_batch(tasks, findings, research_round=0)
    assert 0 < len(batch) <= 4  # budgets.max_concurrent_researchers
    done_ids = {tasks[i].task_id for i in range(3)}
    assert not (done_ids & {t.task_id for t in batch})
    # 按 perspective 轮转交错 → 批次内视角互不相同（并发上限 ≤ 5 时成立）
    assert len({t.perspective for t in batch}) == len(batch)
    # 显式并发上限
    assert len(sup.next_batch(tasks, findings, research_round=0, max_concurrent=2)) == 2


def test_supervisor_all_done() -> None:
    sup = Supervisor()
    tasks = _make_tasks(3)
    assert sup.all_done(tasks, []) is False
    assert sup.all_done([], []) is True  # 空任务列表视为全部完成
    fs = [Finding(task_id=t.task_id, status="done") for t in tasks]
    assert sup.all_done(tasks, fs) is True
    # failed 也视为已处理（交给 Gap Analyzer 决策）
    fs2 = [Finding(task_id=tasks[0].task_id, status="failed")] + [
        Finding(task_id=t.task_id, status="partial") for t in tasks[1:]
    ]
    assert sup.all_done(tasks, fs2) is True


def test_detect_stagnation() -> None:
    assert detect_stagnation(["grpo forget", "grpo forgetting"]) is True
    assert detect_stagnation(["grpo forgetting", "vision transformer applications"]) is False
    # 中文：无空格与分词写法应视为同一查询
    assert detect_stagnation(["GRPO 训练中的遗忘问题", "grpo 训练 遗忘 问题"]) is True
    assert detect_stagnation(["only one"]) is False
    assert detect_stagnation([]) is False
    assert detect_stagnation(["a b", "a c"], threshold=0.0) is True


# --------------------------------------------------------------------------
# Gap Analyzer
# --------------------------------------------------------------------------
def test_gap_no_evidence_high_gap_and_new_task() -> None:
    task = ResearchTask(question="GRPO 遗忘的负结果有哪些", perspective="critique")
    report = analyze_gaps([task], [], [], [], research_round=0)
    assert report.sufficient is False
    # 全局 + 该任务的 no_evidence high 缺口
    assert any(g.task_id is None and g.severity == "high" for g in report.gaps)
    task_gaps = [g for g in report.gaps if g.task_id == task.task_id]
    assert any(g.reason == "no_evidence" and g.severity == "high" for g in task_gaps)
    assert all(g.fix_question for g in report.gaps)  # LLM=None → 模板也非空
    # high 缺口 → 定向新任务
    assert report.new_tasks and all(t.origin == "gap" for t in report.new_tasks)
    nt = report.new_tasks[0]
    assert nt.parent_task_id == task.task_id
    assert nt.round_created == 1
    assert nt.perspective == "critique"  # 沿用原任务视角
    assert nt.question.startswith("定向补充")
    assert nt.max_tool_calls == 5
    assert nt.allowed_tools


def test_gap_max_new_tasks_cap() -> None:
    tasks = [ResearchTask(question=f"问题{i}", perspective="background") for i in range(3)]
    report = analyze_gaps(tasks, [], [], [], research_round=0, max_new_tasks=1)
    assert len(report.new_tasks) == 1


def test_gap_single_source() -> None:
    task = ResearchTask(question="方法对比矩阵", perspective="method", required_evidence=["方法原文"])
    papers = [PaperRecord(paper_id="p1", title="Paper One", year=2024)]
    ev = [
        EvidenceRecord(paper_id="p1", task_id=task.task_id, claim_hint="A 方法描述", evidence_text="片段A"),
        EvidenceRecord(paper_id="p1", task_id=task.task_id, claim_hint="B 方法描述", evidence_text="片段B"),
    ]
    report = analyze_gaps([task], papers, ev, [], research_round=0)
    assert any(g.reason == "single_source" and g.severity == "medium" for g in report.gaps)
    assert not any(g.severity == "high" for g in report.gaps)
    assert report.sufficient is True
    assert report.new_tasks == []


def test_gap_no_experiment_support() -> None:
    task = ResearchTask(question="X 方法有效的证据", perspective="experiment", required_evidence=["消融结果"])
    papers = [
        PaperRecord(paper_id="p1", title="P", year=2024),
        PaperRecord(paper_id="p2", title="Q", year=2025),
    ]
    ev = [
        EvidenceRecord(paper_id="p1", task_id=task.task_id, claim_hint="作者认为有效", evidence_text="片段"),
        EvidenceRecord(paper_id="p2", task_id=task.task_id, claim_hint="与基线相当", evidence_text="片段"),
    ]
    report = analyze_gaps([task], papers, ev, [], research_round=0)
    assert any(g.reason == "no_experiment_support" and g.severity == "high" for g in report.gaps)
    assert report.sufficient is False
    assert report.new_tasks and report.new_tasks[0].origin == "gap"


def test_gap_unexplained_conflict() -> None:
    task = ResearchTask(question="X 方法是否有效", perspective="experiment", required_evidence=["指标原文"])
    papers = [
        PaperRecord(paper_id="p1", title="P", year=2024),
        PaperRecord(paper_id="p2", title="Q", year=2025),
    ]
    ev = [
        EvidenceRecord(paper_id="p1", task_id=task.task_id, claim_hint="A 上有效", evidence_text="片段"),
        EvidenceRecord(paper_id="p2", task_id=task.task_id, claim_hint="B 上无效", evidence_text="片段"),
    ]
    findings = [Finding(task_id=task.task_id, status="done", notes=["两篇结论 conflict：数据集版本不同"])]
    report = analyze_gaps([task], papers, ev, findings, research_round=0)
    assert any(g.reason == "unexplained_conflict" and g.severity == "medium" for g in report.gaps)
    assert report.sufficient is True


def test_gap_stale_sources() -> None:
    task = ResearchTask(question="早期工作脉络", perspective="background")
    papers = [
        PaperRecord(paper_id="p_old", title="Old", year=2015),
        PaperRecord(paper_id="p2", title="Mid", year=2024),
        PaperRecord(paper_id="p3", title="New", year=2025),
    ]
    ev = [EvidenceRecord(paper_id="p_old", task_id=task.task_id, claim_hint="早期脉络", evidence_text="片段")]
    report = analyze_gaps([task], papers, ev, [], research_round=0)
    # 中位数 2024 → 截止 2018，2015 全部早于截止 → stale_sources/low
    assert any(g.reason == "stale_sources" and g.severity == "low" for g in report.gaps)
    assert report.sufficient is True


def test_gap_round_limit_sufficient() -> None:
    task = ResearchTask(question="无证据问题", perspective="background")
    report = analyze_gaps([task], [], [], [], research_round=2, max_rounds=2)
    assert report.sufficient is True  # research_round+1 > max_rounds → 停止循环
    assert any(g.severity == "high" for g in report.gaps)  # 缺口仍如实记录


def test_gap_llm_fix_questions() -> None:
    task = ResearchTask(question="另一个无证据问题", perspective="method")
    FakeBackend.register(
        "gap_fix",
        {"items": [{"index": 1, "fix_question": "窄化：补 2025 年之后的 GRPO 负结果论文"}]},
    )
    report = analyze_gaps([task], [], [], [], llm=get_fake_llm(), research_round=0)
    # index 0 是全局缺口（无 LLM 改写 → 模板），index 1 是该任务缺口（LLM 改写）
    task_gap = next(g for g in report.gaps if g.task_id == task.task_id)
    assert task_gap.fix_question == "窄化：补 2025 年之后的 GRPO 负结果论文"
    global_gap = next(g for g in report.gaps if g.task_id is None)
    assert global_gap.fix_question.startswith("定向补充")
    assert report.new_tasks[0].question == "窄化：补 2025 年之后的 GRPO 负结果论文"


def test_gap_llm_failure_uses_template() -> None:
    task = ResearchTask(question="再一个无证据问题", perspective="background")
    report = analyze_gaps([task], [], [], [], llm=BoomLLM(), research_round=0)
    assert all(g.fix_question.startswith("定向补充") for g in report.gaps)


def test_gap_description_no_recursive_nesting() -> None:
    """缺口描述内嵌的任务问题须剥离逐轮叠加的「定向补充」前缀，不得嵌套膨胀。"""
    task = ResearchTask(
        question='定向补充（缺实验/消融支撑）：定向补充（尚无直接证据）：GRPO 遗忘的负结果有哪些',
        perspective="critique",
    )
    report = analyze_gaps([task], [], [], [], research_round=1)
    task_gaps = [g for g in report.gaps if g.task_id == task.task_id]
    assert task_gaps
    for g in task_gaps:
        assert g.description.count("定向补充") <= 1
        assert "定向补充（缺实验/消融支撑）：定向补充" not in g.description
    # 新任务的 fix_question 同样只保留一层前缀
    if report.new_tasks:
        assert all(t.question.count("定向补充") <= 1 for t in report.new_tasks)
