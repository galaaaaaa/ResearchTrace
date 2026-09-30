"""Analyst：方法对比矩阵构建。

- LLM（analyst 角色）按统一列对齐多篇论文卡片；强约束：只能使用给定卡片与证据信息，
  缺失填 "未报告"；不同数据集版本/指标口径/评测设置不能直接横向比较 → 写进 warnings；
- 代码再补规则告警：数据集无交集、同名指标在不同数据集上报告、"未报告" 占比 > 40%；
- LLM 失败 → 纯规则兜底矩阵（列 = 方法/数据集/指标/结果，值取卡片字段直拼）；
- 绝不抛异常到图节点层。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.llm import get_llm
from src.schemas import ComparisonMatrix, ComparisonRow, EvidenceRecord, PaperCard, PaperRecord, ResearchTask
from src.tracing import get_active_tracer
from src.utils import truncate

_MISSING = "未报告"
# 常见可比较指标名（用于"指标出现但数据集不同"的规则告警）
_METRIC_HINTS: tuple[str, ...] = (
    "accuracy", "准确率", "f1", "precision", "recall", "auc", "bleu", "rouge", "success rate", "pass@1",
)


class _AnalystRow(BaseModel):
    """LLM 输出的一行：paper_id → 列名 → 单元格值。"""

    paper_id: str
    values: dict[str, Any] = Field(default_factory=dict)


class _AnalystOut(BaseModel):
    """LLM 输出的矩阵结构。"""

    columns: list[str] = Field(default_factory=list)
    rows: list[_AnalystRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _norm_metric(name: str) -> str:
    return (name or "").strip().lower()


def _dataset_set(card: PaperCard) -> set[str]:
    return {d.strip().lower() for d in card.datasets if d.strip()}


def _metric_names(card: PaperCard) -> set[str]:
    """卡片 metrics + results 文本中出现的常见指标名。"""
    names = {_norm_metric(m) for m in card.metrics if m.strip()}
    text = (card.results or "").lower()
    names.update(h for h in _METRIC_HINTS if h in text)
    return names


def _rule_warnings(cards: list[PaperCard]) -> list[str]:
    """纯规则告警：数据集无交集 / 同名指标不同数据集 / 未报告占比（后者由调用方补充）。"""
    warnings: list[str] = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            a, b = cards[i], cards[j]
            da, db = _dataset_set(a), _dataset_set(b)
            if da and db and not (da & db):
                warnings.append(
                    f"论文 {a.paper_id} 与 {b.paper_id} 的数据集无交集"
                    f"（{'/'.join(sorted(da))} vs {'/'.join(sorted(db))}），结果不可直接横向比较"
                )
    # 同名指标出现在不同数据集上
    owner: dict[str, list[str]] = {}
    for card in cards:
        ds = "/".join(sorted(_dataset_set(card))) or _MISSING
        for metric in _metric_names(card):
            owner.setdefault(metric, []).append(ds)
    for metric, datasets in owner.items():
        unique = {d for d in datasets}
        if len(datasets) >= 2 and len(unique) > 1:
            warnings.append(f"指标 '{metric}' 在不同数据集上报告（{', '.join(sorted(unique))}），口径可能不一致")
    return warnings


def _missing_ratio_warning(rows: list[ComparisonRow]) -> str | None:
    """矩阵单元格中 '未报告' 占比 > 40% 时告警。"""
    cells = [v for row in rows for v in row.values.values()]
    if not cells:
        return None
    ratio = sum(1 for v in cells if str(v).strip() == _MISSING) / len(cells)
    if ratio > 0.4:
        return f"矩阵中 '{_MISSING}' 占比 {ratio:.0%} 超过 40%，对比价值有限"
    return None


def build_comparison(
    task: ResearchTask,
    papers: list[PaperRecord],
    evidence: list[EvidenceRecord],
    cards: list[PaperCard],
    *,
    llm: Any = None,
) -> ComparisonMatrix:
    """把同一任务下的论文卡片对齐为统一字段对比矩阵。

    Args:
        task: 当前研究子任务。
        papers: 论文元数据（用于回填行标题）。
        evidence: 证据列表（仅用于给 LLM 提供每篇论文的证据条数上下文）。
        cards: Reader 产出的论文卡片（矩阵的数据来源）。
        llm: 注入的 LLM 客户端（默认 get_llm("analyst")）。

    Returns:
        ComparisonMatrix；cards < 2 时返回空矩阵 + note；LLM 失败时返回规则兜底矩阵。
    """
    tracer = get_active_tracer()
    tracer.event("node_start", node="analyst", task_id=task.task_id, cards=len(cards))

    if len(cards) < 2:
        matrix = ComparisonMatrix(task_id=task.task_id, notes="论文不足 2 篇，无法对比")
        tracer.event("node_end", node="analyst", task_id=task.task_id, rows=0, skipped=True)
        return matrix

    titles = {p.paper_id: p.title for p in papers}
    evidence_counts: dict[str, int] = {}
    for ev in evidence:
        evidence_counts[ev.paper_id] = evidence_counts.get(ev.paper_id, 0) + 1

    rule_warnings = _rule_warnings(cards)
    try:
        matrix = _llm_matrix(task, cards, titles, evidence_counts, llm or get_llm("analyst"))
        matrix.task_id = task.task_id
    except Exception as exc:  # noqa: BLE001 —— LLM 失败 → 纯规则兜底
        tracer.event("tool_error", node="analyst", task_id=task.task_id, error=f"{type(exc).__name__}: {exc}")
        matrix = _fallback_matrix(task, cards, titles)
        matrix.notes = f"LLM 失败（{type(exc).__name__}），使用规则兜底矩阵"

    # 补代码规则告警（与 LLM warnings 去重合并）
    warnings = list(matrix.comparability_warnings)
    for w in rule_warnings:
        if w not in warnings:
            warnings.append(w)
    missing_warning = _missing_ratio_warning(matrix.rows)
    if missing_warning and missing_warning not in warnings:
        warnings.append(missing_warning)
    matrix.comparability_warnings = warnings

    # 缺失的卡片补兜底行，保证每篇论文都有行
    covered = {row.paper_id for row in matrix.rows}
    fallback = _fallback_matrix(task, cards, titles)
    for row in fallback.rows:
        if row.paper_id not in covered:
            matrix.rows.append(row)

    tracer.event(
        "node_end", node="analyst", task_id=task.task_id, rows=len(matrix.rows), warnings=len(warnings)
    )
    return matrix


def _llm_matrix(
    task: ResearchTask,
    cards: list[PaperCard],
    titles: dict[str, str],
    evidence_counts: dict[str, int],
    llm: Any,
) -> ComparisonMatrix:
    """调用 LLM 生成矩阵（强约束见 system 提示词）。"""
    card_lines: list[str] = []
    for c in cards:
        card_lines.append(
            f"- paper_id={c.paper_id} | 题名={truncate(titles.get(c.paper_id, c.paper_id), 80)} | 方法={c.method or 'null'}"
            f" | 数据集={c.datasets or []} | 指标={c.metrics or []} | 结果={truncate(c.results or 'null', 200)}"
            f" | 局限={c.limitations or []} | 消融支撑={c.is_ablation_supported}"
            f" | 证据条数={evidence_counts.get(c.paper_id, 0)}"
        )
    system = (
        "你是方法对比分析 Agent（Analyst）。硬性规则：\n"
        "1. 只能使用下方给定的论文卡片与证据信息填表，缺失信息一律填 \"未报告\"，禁止编造；\n"
        "2. 不同数据集版本 / 指标口径 / 评测设置的结果不能直接横向比较，必须写入 warnings；\n"
        "3. columns 覆盖方法、数据集、指标、结果、局限等维度；rows 的 paper_id 必须来自给定卡片，不可遗漏。"
    )
    prompt = (
        f"任务问题：{task.question}\n\n论文卡片：\n" + "\n".join(card_lines) + "\n\n请生成方法对比矩阵（columns/rows/warnings）。"
    )
    out = llm.chat_json(prompt, _AnalystOut, system=system, label="analyst:matrix")

    rows: list[ComparisonRow] = []
    for r in out.rows:
        if not str(r.paper_id).strip():
            continue
        values = {str(col): ("" if v is None else str(v)) for col, v in (r.values or {}).items()}
        rows.append(ComparisonRow(paper_id=str(r.paper_id), title=titles.get(r.paper_id, ""), values=values))
    # 列以 LLM 输出为准，缺 title 列不影响（title 存在行字段上）
    return ComparisonMatrix(
        task_id=task.task_id,
        columns=[str(c) for c in out.columns if str(c).strip()],
        rows=rows,
        comparability_warnings=[str(w) for w in out.warnings if str(w).strip()],
    )


def _fallback_matrix(task: ResearchTask, cards: list[PaperCard], titles: dict[str, str]) -> ComparisonMatrix:
    """纯规则兜底矩阵：列 = 方法/数据集/指标/结果，值取卡片字段直拼。"""
    columns = ["方法", "数据集", "指标", "结果"]
    rows = [
        ComparisonRow(
            paper_id=c.paper_id,
            title=titles.get(c.paper_id, ""),
            values={
                "方法": c.method or _MISSING,
                "数据集": "、".join(c.datasets) if c.datasets else _MISSING,
                "指标": "、".join(c.metrics) if c.metrics else _MISSING,
                "结果": c.results or _MISSING,
            },
        )
        for c in cards
    ]
    return ComparisonMatrix(task_id=task.task_id, columns=columns, rows=rows)
