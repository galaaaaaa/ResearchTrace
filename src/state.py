"""LangGraph 全局状态与合并 Reducer。

设计要点：
- papers / evidence / findings / cards 使用去重合并 Reducer，支持 Researcher 并行分支安全写入；
- tasks / outline / draft / verification 等由单一节点整体覆写；
- 预算计数不进状态（并行分支写冲突），由 src/llm.py 的 TokenBudget 单例管理。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from .schemas import (
    ClaimRecord,
    ComparisonMatrix,
    ConflictReport,
    EvidenceRecord,
    Finding,
    Gap,
    OutlineSection,
    PaperCard,
    PaperRecord,
    ResearchBrief,
    ResearchTask,
    VerificationReport,
)


def _dedup_merge(old: Any, new: Any, key: str):
    """按 key 去重合并两个对象列表（保留先出现的）。任一端为 None 视为空表。"""
    items = [x for x in (old or [])] + [x for x in (new or [])]
    seen: set[str] = set()
    merged = []
    for item in items:
        k = getattr(item, key, None) or str(id(item))
        if k in seen:
            continue
        seen.add(k)
        merged.append(item)
    return merged


def merge_papers(old: list[PaperRecord] | None, new: list[PaperRecord] | None) -> list[PaperRecord]:
    return _dedup_merge(old, new, "paper_id")  # type: ignore[return-value]


def merge_evidence(old: list[EvidenceRecord] | None, new: list[EvidenceRecord] | None) -> list[EvidenceRecord]:
    return _dedup_merge(old, new, "evidence_id")  # type: ignore[return-value]


def merge_findings(old: list[Finding] | None, new: list[Finding] | None) -> list[Finding]:
    """同一 task_id 只保留最新 Finding（轮次推进时覆盖旧记录）。"""
    if not new:
        return list(old or [])
    by_task = {f.task_id: f for f in (old or [])}
    for f in new:
        by_task[f.task_id] = f
    return list(by_task.values())


def merge_cards(old: list[PaperCard] | None, new: list[PaperCard] | None) -> list[PaperCard]:
    if not new:
        return list(old or [])
    by_paper = {c.paper_id: c for c in (old or [])}
    for c in new:
        by_paper[c.paper_id] = c
    return list(by_paper.values())


class ResearchState(TypedDict, total=False):
    # ---- 输入 ----
    user_query: str
    papers_dir: str
    run_id: str

    # ---- Scope / Plan ----
    brief: ResearchBrief
    tasks: list[ResearchTask]  # 由 supervisor/gap 整体覆写
    current_task: dict  # Send fan-out 注入的当前任务（仅 researcher 节点读取）

    # ---- 并行研究产物（reducer 合并）----
    papers: Annotated[list[PaperRecord], merge_papers]
    evidence: Annotated[list[EvidenceRecord], merge_evidence]
    findings: Annotated[list[Finding], merge_findings]
    cards: Annotated[list[PaperCard], merge_cards]
    comparisons: Annotated[list[ComparisonMatrix], add]
    conflicts: Annotated[list[ConflictReport], add]
    tool_logs: Annotated[list, add]

    # ---- 分析与写作 ----
    gaps: list[Gap]
    gap_report: Any  # GapReport
    outline: list[OutlineSection]
    draft: str
    claims: list[ClaimRecord]

    # ---- 核验 ----
    verification: VerificationReport
    repair_tasks: list[ResearchTask]

    # ---- 循环控制 ----
    research_round: int
    repair_round: int
    budget: dict  # 快照（真实计数在 TokenBudget）

    # ---- 输出 ----
    report_md: str
    report_path: str
    audit_path: str
    errors: Annotated[list[str], add]
    final_status: str
