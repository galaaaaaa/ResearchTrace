"""Critic：冲突检测（规则旗标 + LLM 深度发现，并列呈现不强求统一）。

- 规则旗标（不依赖 LLM）：无消融支撑的因果归因、同名指标数值方向相反、撤稿论文仍在引用；
- LLM（critic 角色）：寻找与主流结论相反的证据 / 小样本 / 数据污染 / 口径不一致 / 论文与代码
  不一致，宁缺毋滥，最多 5 条；
- 合并去重（topic 相似 > 0.8）；LLM 失败只返回规则旗标；绝不抛异常到图节点层。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.llm import get_llm
from src.schemas import ConflictReport, EvidenceRecord, PaperCard, PaperRecord, ResearchTask
from src.tracing import get_active_tracer
from src.utils import truncate

# 因果归因词（用于"无消融支撑的因果归因"旗标）
_CAUSAL_WORDS: tuple[str, ...] = (
    "因为", "导致", "由于", "因而", "所以", "because", "causes", "caused", "leads to", "lead to", "results in",
)
# 指标方向词（用于"同一指标数值方向相反"旗标，尽力而为）
_UP_WORDS: tuple[str, ...] = (
    "提高", "提升", "上升", "增加", "改善", "更高", "improve", "increase", "gain", "higher", "better",
)
_DOWN_WORDS: tuple[str, ...] = (
    "下降", "降低", "减少", "退化", "更低", "decrease", "drop", "lower", "worse", "degrad",
)
_METRIC_HINTS: tuple[str, ...] = (
    "accuracy", "准确率", "f1", "precision", "recall", "auc", "bleu", "rouge", "success rate", "pass@1",
)
_NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*%?")
_MAX_LLM_CONFLICTS = 5


class _LLMConflict(BaseModel):
    """LLM 输出的单条冲突。"""

    topic: str
    paper_ids: list[str] = Field(default_factory=list)
    description: str = ""
    possible_causes: list[str] = Field(default_factory=list)
    severity: str = "medium"


class _LLMConflicts(BaseModel):
    """LLM 输出的冲突集合。"""

    conflicts: list[_LLMConflict] = Field(default_factory=list)


def _sanitize_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in ("high", "medium", "low") else "medium"


def _metric_directions(cards: list[PaperCard]) -> dict[str, dict[str, tuple[int, str]]]:
    """从 results 文本抽数字 + 方向词，得到 {metric: {paper_id: (方向, 数字)}}。"""
    out: dict[str, dict[str, tuple[int, str]]] = {}
    for card in cards:
        text = (card.results or "")
        low = text.lower()
        metrics = {m.strip().lower() for m in card.metrics if m.strip()} | {h for h in _METRIC_HINTS if h in low}
        for metric in metrics:
            direction: int | None = None
            number = ""
            for m in re.finditer(re.escape(metric), low):
                window = low[max(0, m.start() - 40): m.end() + 80]
                has_up = any(w in window for w in _UP_WORDS)
                has_down = any(w in window for w in _DOWN_WORDS)
                if has_up and has_down:
                    continue  # 窗口内方向矛盾，跳过该次出现
                if has_up or has_down:
                    d = 1 if has_up else -1
                    if direction is not None and direction != d:
                        direction = None
                        break  # 论文内部方向矛盾，放弃该指标
                    direction = d
                    num = _NUM_RE.search(window)
                    if num and not number:
                        number = num.group(0).strip()
            if direction is not None:
                out.setdefault(metric, {})[card.paper_id] = (direction, number)
    return out


def _rule_flags(cards: list[PaperCard], papers: list[PaperRecord]) -> list[ConflictReport]:
    """规则旗标：因果无消融 / 指标方向相反 / 撤稿引用。"""
    flags: list[ConflictReport] = []
    for card in cards:
        low = (card.results or "").lower()
        if card.is_ablation_supported is False and any(w in low for w in _CAUSAL_WORDS):
            flags.append(
                ConflictReport(
                    topic="无消融支撑的因果归因",
                    paper_ids=[card.paper_id],
                    description=f"论文 {card.paper_id} 的结果表述含因果归因，但未报告消融实验支撑",
                    severity="medium",
                )
            )
    for metric, per_paper in _metric_directions(cards).items():
        if len({d for d, _ in per_paper.values()}) > 1:
            detail = "；".join(
                f"{pid} 报告 {metric} {'提高' if d > 0 else '下降'}({num or '数值未知'})" for pid, (d, num) in per_paper.items()
            )
            flags.append(
                ConflictReport(
                    topic=f"同一指标 '{metric}' 数值方向相反",
                    paper_ids=list(per_paper),
                    description=detail,
                    possible_causes=["实验设置或数据集版本不同", "指标口径不一致"],
                    severity="medium",
                )
            )
    for p in papers:
        if p.is_retracted:
            flags.append(
                ConflictReport(
                    topic="撤稿论文结论仍在引用",
                    paper_ids=[p.paper_id],
                    description=f"论文 {p.paper_id} 已被标记撤稿，但其结论仍出现在证据集中",
                    severity="high",
                )
            )
    return flags


def _topics_similar(a: str, b: str) -> bool:
    """topic 相似度 > 0.8（token_set_ratio/100）。"""
    if not a or not b:
        return False
    try:
        from rapidfuzz import fuzz

        return fuzz.token_set_ratio(a, b) / 100.0 > 0.8
    except Exception:  # noqa: BLE001
        return a.strip() == b.strip()


def find_conflicts(
    task: ResearchTask,
    papers: list[PaperRecord],
    evidence: list[EvidenceRecord],
    cards: list[PaperCard],
    *,
    llm: Any = None,
) -> list[ConflictReport]:
    """检测论文集内部的结论冲突（规则旗标 + LLM 发现，合并去重）。

    Args:
        task: 当前研究子任务。
        papers: 论文元数据（is_retracted 参与规则旗标）。
        evidence: 证据列表（摘要进入 LLM 输入；id 用于回填冲突的证据引用）。
        cards: Reader 产出的论文卡片。
        llm: 注入的 LLM 客户端（默认 get_llm("critic")）。

    Returns:
        list[ConflictReport]；LLM 失败时只返回规则旗标，绝不抛异常。
    """
    tracer = get_active_tracer()
    tracer.event("node_start", node="critic", task_id=task.task_id, cards=len(cards), evidence=len(evidence))

    conflicts = _rule_flags(cards, papers)

    try:
        llm_conflicts = _llm_conflicts(task, papers, evidence, cards, llm or get_llm("critic"))
        for c in llm_conflicts[:_MAX_LLM_CONFLICTS]:
            topic = str(c.topic or "").strip()
            if not topic:
                continue
            if any(_topics_similar(topic, existing.topic) for existing in conflicts):
                continue
            conflicts.append(
                ConflictReport(
                    topic=topic,
                    paper_ids=[str(p) for p in c.paper_ids if str(p).strip()],
                    description=str(c.description or ""),
                    possible_causes=[str(x) for x in c.possible_causes if str(x).strip()],
                    severity=_sanitize_severity(c.severity),
                )
            )
    except Exception as exc:  # noqa: BLE001 —— LLM 失败只返回规则旗标
        tracer.event("tool_error", node="critic", task_id=task.task_id, error=f"{type(exc).__name__}: {exc}")

    # evidence_ids 回填：冲突涉及论文的证据取前 3 条
    for conflict in conflicts:
        ids = [ev.evidence_id for ev in evidence if ev.paper_id in conflict.paper_ids][:3]
        conflict.evidence_ids = ids

    tracer.event("node_end", node="critic", task_id=task.task_id, conflicts=len(conflicts))
    return conflicts


def _llm_conflicts(
    task: ResearchTask,
    papers: list[PaperRecord],
    evidence: list[EvidenceRecord],
    cards: list[PaperCard],
    llm: Any,
) -> list[_LLMConflict]:
    """LLM 深度冲突发现（输入=卡片+证据摘要，每条截 300 字符）。"""
    titles = {p.paper_id: p.title for p in papers}
    card_lines: list[str] = []
    for c in cards:
        card_lines.append(
            f"- paper_id={c.paper_id} | 题名={truncate(titles.get(c.paper_id, c.paper_id), 80)}"
            f" | 方法={c.method or 'null'} | 结果={truncate(c.results or 'null', 200)}"
            f" | 局限={c.limitations or []} | 消融支撑={c.is_ablation_supported}"
        )
    evidence_lines = [
        f"- [{ev.paper_id} p{ev.page}] {truncate(ev.evidence_text, 300)}" for ev in evidence[:20]
    ]
    system = (
        "你是科研结论冲突检测 Agent（Critic）。寻找：与主流结论相反的证据、小样本风险、数据污染、"
        "指标口径不一致、论文与代码实现不一致、撤稿/勘误/后续否定结果。\n"
        "宁缺毋滥：只报告有给定材料支撑的冲突，最多 5 条；没有可靠冲突时返回空列表，禁止编造。"
    )
    prompt = (
        f"任务问题：{task.question}\n\n论文卡片：\n" + "\n".join(card_lines)
        + "\n\n证据摘要：\n" + ("\n".join(evidence_lines) if evidence_lines else "（无）")
        + "\n\n请输出 conflicts。"
    )
    out = llm.chat_json(prompt, _LLMConflicts, system=system, label="critic:conflicts")
    return list(out.conflicts)
