"""Gap Analyzer：证据缺口检测与定向任务生成（文档 §7.7）。

对每个研究子问题检查：
- 无任何直接相关证据 → no_evidence / high；
- 证据全部来自单一论文 → single_source / medium；
- 要求实验/消融证据但现有证据均不含 → no_experiment_support / high；
- Finding 记录了未消解冲突（notes 含 conflict/冲突）→ unexplained_conflict / medium；
- 证据论文年份相对整体过旧（全部早于 papers 年份中位数 - 6）→ stale_sources / low。

缺口经 LLM（fast 角色，一次调用）改写成"窄化"的 fix_question（补年份/补反例/补具体
方法，不重新宽泛搜索）；LLM 不可用或失败 → 模板 fix_question。
severity=high 的缺口转换成新的定向任务（origin="gap"），受 max_new_tasks 与
max_rounds 双重上限约束：达到轮次上限时 sufficient=True，缺口写入报告而非继续循环。
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.schemas import (
    EvidenceRecord,
    Finding,
    Gap,
    GapReport,
    PaperRecord,
    ResearchTask,
)
from src.settings import get_settings
from src.tracing import get_active_tracer

_NODE = "gap_analyzer"

#: 与 schemas.ResearchTask 默认白名单一致
_DEFAULT_ALLOWED_TOOLS: list[str] = [
    "search_arxiv",
    "search_semantic_scholar",
    "search_web_literature",
    "traverse_citations",
    "download_pdf",
    "read_pdf",
]

#: 判定"需要实验支撑"的关键词（出现在 required_evidence 中）
_EXPERIMENT_KEYWORDS: tuple[str, ...] = ("消融", "ablation", "实验", "对照", "benchmark")

#: 各缺口原因对应的补证任务所需证据
_GAP_EVIDENCE: dict[str, list[str]] = {
    "no_evidence": ["与结论直接相关的原文证据片段"],
    "single_source": ["第二来源的独立佐证"],
    "no_experiment_support": ["消融或对照实验的表格原文"],
    "unexplained_conflict": ["冲突双方的原文表述与实验设置"],
    "stale_sources": ["近年同主题论文的原文证据"],
    "low_quality": ["更可靠来源（同行评审/官方）的原文证据"],
}

_REASON_LABEL: dict[str, str] = {
    "no_evidence": "无任何直接相关证据",
    "single_source": "仅单一来源",
    "no_experiment_support": "关键结论缺实验支撑",
    "unexplained_conflict": "存在未解释的冲突",
    "stale_sources": "来源过旧",
    "low_quality": "来源质量低",
}


class _FixItem(BaseModel):
    index: int
    fix_question: str


class _FixDraft(BaseModel):
    items: list[_FixItem] = Field(default_factory=list)


_GAP_SYSTEM = """#FAKE:gap_fix
你是 Gap Analyzer 的缺口改写器（fast 角色）。输入是证据缺口列表，把每个缺口改写成一个定向补充任务的问题（fix_question）。

改写要求（窄化，不放宽）：
1. 只补具体缺口：补年份（如"2025 年之后"）、补反例/负结果、补具体方法或数据集；
2. 不得重新发起宽泛搜索（"再全面调研一遍 X"是错误示范）；
3. 每个问题一句中文，可检索、可判定完成；
4. items 必须覆盖输入列表中的每一个 index。"""


# --------------------------------------------------------------------------
# 规则检查
# --------------------------------------------------------------------------
def _needs_experiment(task: ResearchTask) -> bool:
    text = " ".join(task.required_evidence).lower()
    return any(k.lower() in text for k in _EXPERIMENT_KEYWORDS)


def _hints_experiment(evidence: EvidenceRecord) -> bool:
    hint = (evidence.claim_hint or "").lower()
    return any(k.lower() in hint for k in _EXPERIMENT_KEYWORDS)


_REASON_ZH = {
    "no_evidence": "尚无直接证据",
    "single_source": "来源单一",
    "no_experiment_support": "缺实验/消融支撑",
    "unexplained_conflict": "冲突未解释",
    "stale_sources": "来源过旧",
    "low_quality": "来源质量低",
}

_NESTED_PREFIXES = ("定向补充：", "定向补充（", "补充直接证据：")


def _strip_nesting(text: str, *, limit: int = 90) -> str:
    """剥离 fix_question 逐轮叠加的定向前缀并截断，防止"定向补充（…）：任务'定向补充…'"式膨胀。"""
    out = (text or "").strip()
    while len(out) > 8:
        stripped = False
        for pre in ("定向补充", "补充直接证据"):
            if not out.startswith(pre):
                continue
            rest = out[len(pre):]
            if rest.startswith("（"):  # 跳过括号内的缺口类型说明
                depth = 0
                for i, ch in enumerate(rest):
                    if ch == "（":
                        depth += 1
                    elif ch == "）":
                        depth -= 1
                        if depth == 0:
                            rest = rest[i + 1:]
                            break
            rest = rest.lstrip("：: ").strip()
            if rest.startswith("“") and rest.endswith("”") and len(rest) > 2:
                rest = rest[1:-1].strip()
            if len(rest) >= 8:  # 剥离后仍有实质内容才接受
                out = rest
                stripped = True
            break
        if not stripped:
            break
    if len(out) > limit:
        out = out[: limit - 1] + "…"
    return out


def _check_task(
    task: ResearchTask,
    papers_by_id: dict[str, PaperRecord],
    task_evidence: list[EvidenceRecord],
    task_findings: list[Finding],
    stale_cutoff: int,
) -> list[Gap]:
    """对单个任务执行 §7.7 的五项规则检查，返回缺口列表（按严重度从高到低）。"""
    # 描述内嵌的问题先剥离逐轮叠加的定向前缀并截断，防止"任务'定向补充…：…'"式嵌套膨胀
    question = _strip_nesting(task.question, limit=60)
    gaps: list[Gap] = []

    if not task_evidence:
        gaps.append(
            Gap(
                task_id=task.task_id,
                description=f"任务“{question}”没有任何直接相关证据",
                severity="high",
                reason="no_evidence",
            )
        )
        return gaps

    paper_ids = {ev.paper_id for ev in task_evidence}
    if len(paper_ids) == 1:
        gaps.append(
            Gap(
                task_id=task.task_id,
                description=f"任务“{question}”的证据全部来自单一论文（{next(iter(paper_ids))}），缺少独立来源交叉验证",
                severity="medium",
                reason="single_source",
            )
        )

    if _needs_experiment(task) and not any(_hints_experiment(ev) for ev in task_evidence):
        gaps.append(
            Gap(
                task_id=task.task_id,
                description=f"任务“{question}”要求实验/消融证据，但现有证据的结论提示均不含消融或实验结果",
                severity="high",
                reason="no_experiment_support",
            )
        )

    notes_text = " ".join(n for f in task_findings for n in (f.notes or [])).lower()
    if "conflict" in notes_text or "冲突" in notes_text:
        gaps.append(
            Gap(
                task_id=task.task_id,
                description=f"任务“{question}”的 Finding 记录了尚未解释的结论冲突，需要分析实验设置/数据/定义差异",
                severity="medium",
                reason="unexplained_conflict",
            )
        )

    years = [
        papers_by_id[ev.paper_id].year
        for ev in task_evidence
        if ev.paper_id in papers_by_id and papers_by_id[ev.paper_id].year
    ]
    if years and all(y < stale_cutoff for y in years):
        gaps.append(
            Gap(
                task_id=task.task_id,
                description=f"任务“{question}”的证据论文年份全部早于 {stale_cutoff}，可能错过近年进展",
                severity="low",
                reason="stale_sources",
            )
        )
    return gaps


# --------------------------------------------------------------------------
# LLM fix_question 改写
# --------------------------------------------------------------------------
def _llm_fix_questions(llm: Any, gaps: list[Gap], research_round: int) -> dict[int, str]:
    """把缺口列表交给 LLM 改写成定向问题；失败返回空 dict（走模板）。"""
    tracer = get_active_tracer()
    lines = [f"研究轮次：{research_round}", "缺口列表（index 从 0 开始）："]
    for i, gap in enumerate(gaps):
        label = _REASON_LABEL.get(gap.reason, gap.reason)
        lines.append(f"[{i}] ({gap.reason}/{gap.severity}，{label}) {gap.description}")
    prompt = "\n".join(lines) + "\n请输出 items，覆盖每一个 index。"
    try:
        draft: _FixDraft = llm.chat_json(prompt, _FixDraft, system=_GAP_SYSTEM, label="gap_fix")
        return {
            item.index: item.fix_question.strip()
            for item in draft.items
            if item.fix_question and item.fix_question.strip()
        }
    except Exception as exc:  # noqa: BLE001 —— 失败走模板
        tracer.event("error", node=_NODE, error=f"fix_question 改写失败（{type(exc).__name__}: {str(exc)[:200]}），使用模板")
        return {}


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def analyze_gaps(
    tasks: list[ResearchTask],
    papers: list[PaperRecord],
    evidence: list[EvidenceRecord],
    findings: list[Finding],
    *,
    llm: Any = None,
    research_round: int = 0,
    max_new_tasks: int | None = None,
    max_rounds: int = 2,
) -> GapReport:
    """检查证据缺口并生成定向补充任务。

    Args:
        tasks: 全部研究任务。
        papers: 当前全部论文记录（用于年份中位数判定来源过旧）。
        evidence: 当前全部证据记录（按 task_id 归属）。
        findings: 当前全部 Finding（notes 含 conflict/冲突 → 未消解冲突）。
        llm: 可选 LLM 客户端（建议 fast 角色）；None 时 fix_question 全走模板。
        research_round: 当前研究轮次。
        max_new_tasks: 新任务数上限；缺省取 budgets.max_tasks_per_round。
        max_rounds: 研究轮次上限；research_round+1 超过它则 sufficient=True 停止循环。

    Returns:
        GapReport：gaps（含 fix_question）、new_tasks（origin="gap"）、sufficient。
    """
    tracer = get_active_tracer()
    tracer.event("node_start", node=_NODE, round=research_round, n_tasks=len(tasks), n_evidence=len(evidence))
    settings = get_settings()
    max_new = int(max_new_tasks if max_new_tasks is not None else settings.budget("max_tasks_per_round", 4))
    max_new = max(0, max_new)
    tool_budget = int(settings.budget("max_tool_calls_per_researcher", 5))

    papers_by_id = {p.paper_id: p for p in papers}
    evidence_by_task: dict[str, list[EvidenceRecord]] = {}
    for ev in evidence:
        evidence_by_task.setdefault(ev.task_id, []).append(ev)
    findings_by_task: dict[str, list[Finding]] = {}
    for f in findings:
        findings_by_task.setdefault(f.task_id, []).append(f)

    # 来源过旧判定：全部 year < papers 年份中位数 - 6（无年份时以当前年兜底）
    all_years = [p.year for p in papers if p.year]
    median_year = int(statistics.median(all_years)) if all_years else datetime.now().year
    stale_cutoff = median_year - 6

    gaps: list[Gap] = []
    # 全局：Evidence Store 完全为空（仅在确有任务时记录，空任务列表无事可查）
    if tasks and not evidence:
        gaps.append(
            Gap(
                task_id=None,
                description="全局：Evidence Store 为空，没有任何可引用证据",
                severity="high",
                reason="no_evidence",
            )
        )
    for task in tasks:
        gaps.extend(
            _check_task(
                task,
                papers_by_id,
                evidence_by_task.get(task.task_id, []),
                findings_by_task.get(task.task_id, []),
                stale_cutoff,
            )
        )

    # fix_question：优先 LLM 定向改写，失败/缺省 → 模板（基于父任务原始问题，避免逐轮嵌套膨胀）
    task_by_id = {t.task_id: t for t in tasks}
    fixes: dict[int, str] = {}
    if gaps and llm is not None:
        fixes = _llm_fix_questions(llm, gaps, research_round)
    for i, gap in enumerate(gaps):
        if i in fixes:
            gap.fix_question = fixes[i]
        else:
            parent_q = _strip_nesting(task_by_id[gap.task_id].question) if gap.task_id in task_by_id else _strip_nesting(gap.description)
            gap.fix_question = f"定向补充（{_REASON_ZH.get(gap.reason, gap.reason)}）：{parent_q}"

    # high 缺口 → 定向新任务（每任务只取第一个 high 缺口，受 max_new 上限约束）
    new_tasks: list[ResearchTask] = []
    claimed_tasks: set[str] = set()
    for gap in gaps:
        if gap.severity != "high" or gap.task_id is None:
            continue
        if gap.task_id in claimed_tasks or gap.task_id not in task_by_id or len(new_tasks) >= max_new:
            continue
        parent = task_by_id[gap.task_id]
        new_tasks.append(
            ResearchTask(
                question=gap.fix_question or f"定向补充：{gap.description}",
                perspective=parent.perspective or "critique",
                required_evidence=list(_GAP_EVIDENCE.get(gap.reason, ["与结论直接相关的原文证据片段"])),
                success_criteria="获得可绑定 evidence_id 的定向证据并回填该缺口",
                round_created=research_round + 1,
                origin="gap",
                parent_task_id=parent.task_id,
                max_tool_calls=tool_budget,
                allowed_tools=list(_DEFAULT_ALLOWED_TOOLS),
            )
        )
        claimed_tasks.add(gap.task_id)

    sufficient = not any(g.severity == "high" for g in gaps) or (research_round + 1) > max_rounds
    report = GapReport(sufficient=sufficient, gaps=gaps, new_tasks=new_tasks)
    tracer.event(
        "node_end",
        node=_NODE,
        round=research_round,
        n_gaps=len(gaps),
        n_high=sum(1 for g in gaps if g.severity == "high"),
        n_new_tasks=len(new_tasks),
        sufficient=sufficient,
        stale_cutoff=stale_cutoff,
    )
    return report
