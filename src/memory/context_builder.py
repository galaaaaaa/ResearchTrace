"""Context Builder：token 预算内装配上下文（顺序严格遵循文档 §9）。

装配顺序：
1. 任务相关过滤：task 非空时按（perspective 对应证据类型 + question 关键词与
   claim_hint/evidence_text 的词重叠）打分排序；
2. 论文按 paper_id 去重；
3. 无 pdf_path 且 paper_type="blog" 的论文降级到“线索”区，不与核心证据混排；
4. 直接证据 + 冲突证据优先：conflicts 中出现过的 paper_id/evidence_ids 提权；
5. PaperCard 压缩为每篇 ≤240 字符（保留 paper_id）；
6. 按 estimate_tokens 截断到 token_budget（默认 budgets.context_token_budget=24000）：
   整条丢弃，绝不截断半条；evidence_id/paper_id/page 等引用身份永远完整。
"""
from __future__ import annotations

import re

from src.schemas import (
    ConflictReport,
    EvidenceRecord,
    Finding,
    PaperCard,
    PaperRecord,
    ResearchBrief,
    ResearchTask,
)
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import estimate_tokens, truncate

_QUOTE_CHARS = 500   # 单条证据“原文片段”的压缩上限（引用身份不受影响）
_CARD_CHARS = 240    # 单篇论文卡片（含 paper_id）的压缩上限
_HINT_CHARS = 120    # claim_hint 的压缩上限
_NOTE_RESERVE = 60   # 为附注预留的 token 余量

_CJK_RUN = re.compile(r"[一-鿿]+")
_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-]+")

# perspective → 证据类型关键词（中文子串匹配 / 英文按词匹配）
_PERSPECTIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "background": ("综述", "背景", "相关工作", "概述", "定义", "概念", "脉络", "survey", "overview", "review", "introduction"),
    "method": ("方法", "架构", "框架", "算法", "模型", "提出", "训练", "approach", "method", "architecture", "framework", "algorithm", "pipeline"),
    "experiment": ("实验", "结果", "消融", "数据集", "指标", "精度", "基线", "评估", "ablation", "dataset", "metric", "accuracy", "benchmark", "baseline"),
    "application": ("应用", "部署", "场景", "实践", "落地", "deployment", "practice", "industry", "case study"),
    "critique": ("局限", "不足", "缺陷", "批评", "风险", "失败", "缺点", "争议", "limitation", "drawback", "fail"),
}


def _tokens(text: str) -> set[str]:
    """中英混合粗分词：ASCII 词（≥2 字符）+ CJK 二元组（长度 1 的 run 保留单字）。"""
    if not text:
        return set()
    tokens = set(_ASCII_WORD.findall(text.lower()))
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def _score_evidence(ev: EvidenceRecord, task: ResearchTask | None, conflict_ev_ids: set[str]) -> float:
    """证据相关性打分：冲突提权 > 任务相关（视角关键词 + 问题词重叠）> 直接证据 > 基础分。"""
    score = float(ev.relevance_score or 0.0)
    if ev.evidence_id in conflict_ev_ids:
        score += 1000.0  # 冲突证据必须优先呈现，不允许被预算挤掉
    if task is None:
        return score
    hay = f"{ev.claim_hint or ''}\n{ev.evidence_text or ''}"
    hay_tokens = _tokens(hay)
    score += 2.0 * len(_tokens(task.question) & hay_tokens)
    hits = 0
    for kw in _PERSPECTIVE_KEYWORDS.get(task.perspective, ()):
        if kw.isascii():
            if kw in hay_tokens:
                hits += 1
        elif kw in hay:
            hits += 1
    score += 3.0 * hits
    if ev.page is not None and ev.evidence_text:
        score += 2.0  # 直接证据：定位到原文页码的片段
    return score


def _format_header(task: ResearchTask | None, brief: ResearchBrief | None) -> str:
    lines: list[str] = []
    if task is not None:
        lines.append("## 研究任务")
        lines.append(f"- task_id: {task.task_id}")
        lines.append(f"- 问题: {task.question}")
        lines.append(f"- 视角: {task.perspective}")
        if task.required_evidence:
            lines.append(f"- 所需证据: {'；'.join(task.required_evidence)}")
        if task.success_criteria:
            lines.append(f"- 成功标准: {task.success_criteria}")
    if brief is not None:
        lines.append("## 研究范围")
        lines.append(f"- 目标: {brief.objective}")
        if brief.out_of_scope:
            lines.append(f"- 排除范围: {'；'.join(brief.out_of_scope)}")
        time_desc = brief.time_range or (
            f"{brief.year_from or ''}-{brief.year_to or ''}" if (brief.year_from or brief.year_to) else None
        )
        if time_desc:
            lines.append(f"- 时间范围: {time_desc}")
        if brief.disciplines:
            lines.append(f"- 学科: {'；'.join(brief.disciplines)}")
        if brief.paper_types:
            lines.append(f"- 论文类型: {'；'.join(brief.paper_types)}")
        if brief.core_questions:
            lines.append("- 核心问题:")
            lines.extend(f"  {i + 1}. {q}" for i, q in enumerate(brief.core_questions))
        if brief.criteria:
            lines.append("- 判断标准: " + "；".join(f"{k}={v}" for k, v in brief.criteria.items()))
        if brief.assumptions:
            lines.append(f"- 默认假设: {'；'.join(brief.assumptions)}")
        lines.append(f"- 深度: {brief.depth}")
    return "\n".join(lines)


def _format_card_line(paper: PaperRecord, card: PaperCard | None) -> str:
    """论文 + 卡片压缩为一行，整行 ≤240 字符且保留 paper_id。"""
    meta = [str(paper.year)] if paper.year else []
    if paper.venue:
        meta.append(paper.venue)
    meta.append(paper.paper_type)
    line = f"- [{paper.paper_id}] {paper.title}（{'，'.join(meta)}）"
    if card is not None:
        parts: list[str] = []
        if card.motivation:
            parts.append(f"动机:{card.motivation}")
        if card.method:
            parts.append(f"方法:{card.method}")
        if card.datasets:
            parts.append(f"数据:{'、'.join(card.datasets)}")
        if card.metrics:
            parts.append(f"指标:{'、'.join(card.metrics)}")
        if card.results:
            parts.append(f"结论:{card.results}")
        if card.limitations:
            parts.append(f"局限:{'；'.join(card.limitations)}")
        if card.is_ablation_supported is not None:
            parts.append(f"消融支撑:{'是' if card.is_ablation_supported else '否'}")
        body = " ｜".join(parts) or (card.notes or "（卡片无内容）")
    else:
        body = f"（暂无卡片）摘要:{truncate(paper.abstract or '', 80)}"
    return truncate(f"{line}｜{body}", _CARD_CHARS)


def _format_evidence_line(ev: EvidenceRecord, conflict_ev_ids: set[str]) -> str:
    """单条证据一行主体 + 一行原文片段；引用身份（evidence_id/paper_id/page）不截断。"""
    page = f"p.{ev.page}" if ev.page is not None else "p.?"
    head = f"- [{ev.evidence_id}] 论文 {ev.paper_id} 页码 {page}"
    if ev.section:
        head += f"（{ev.section}）"
    if ev.evidence_id in conflict_ev_ids:
        head += " ｜冲突相关"
    quote = truncate(ev.evidence_text or "", _QUOTE_CHARS)
    hint = truncate(ev.claim_hint or "", _HINT_CHARS)
    extra = "" if ev.modality == "text" else f" ｜模态:{ev.modality}"
    if ev.figure_path:
        extra += f" ｜图:{ev.figure_path}"
    return f"{head}\n  原文: {quote}{extra} ｜结论提示: {hint}"


def _format_conflict_line(conflict: ConflictReport) -> str:
    parts = [f"- [{conflict.conflict_id}] {conflict.topic}（严重度:{conflict.severity}）"]
    if conflict.description:
        parts.append(conflict.description)
    if conflict.paper_ids:
        parts.append("涉及论文:" + "、".join(conflict.paper_ids))
    if conflict.evidence_ids:
        parts.append("涉及证据:" + "、".join(conflict.evidence_ids))
    if conflict.possible_causes:
        parts.append("可能原因:" + "；".join(conflict.possible_causes))
    return " ｜".join(parts)


def _format_lead_line(paper: PaperRecord) -> str:
    src = f" 来源:{paper.source_url}" if paper.source_url else ""
    return f"- [{paper.paper_id}] {paper.title}（{paper.paper_type}，无原文 PDF，仅作线索，不作为直接证据）{src}"


def build_context(
    *,
    task: ResearchTask | None = None,
    brief: ResearchBrief | None = None,
    evidence: list[EvidenceRecord] | None = None,
    papers: list[PaperRecord] | None = None,
    cards: list[PaperCard] | None = None,
    conflicts: list[ConflictReport] | None = None,
    extra: str | None = None,
    token_budget: int | None = None,
) -> str:
    """装配顺序：任务相关过滤→去重→过滤无原文→直接+冲突证据优先→摘要压缩→token 截断（绝不截断引用身份）。"""
    try:
        return _build(
            task=task,
            brief=brief,
            evidence=evidence or [],
            papers=papers or [],
            cards=cards or [],
            conflicts=conflicts or [],
            extra=extra,
            token_budget=token_budget,
        )
    except Exception as ex:  # 装配失败降级为最小上下文，不崩调用方
        get_active_tracer().event("warning", node="context_builder", error=str(ex))
        question = task.question if task is not None else "（无任务）"
        return f"## 研究任务\n- {question}\n\n## 附注\n- 上下文装配失败，已降级为最小上下文。\n"


def _build(
    *,
    task: ResearchTask | None,
    brief: ResearchBrief | None,
    evidence: list[EvidenceRecord],
    papers: list[PaperRecord],
    cards: list[PaperCard],
    conflicts: list[ConflictReport],
    extra: str | None,
    token_budget: int | None,
) -> str:
    if token_budget is None:
        token_budget = int(get_settings().budget("context_token_budget", 24000) or 24000)
    budget = max(token_budget, 200)

    # 1) 去重：evidence / papers 按 id 保留首个；cards 按 paper_id 保留首个
    uniq_ev: dict[str, EvidenceRecord] = {}
    for ev in evidence:
        uniq_ev.setdefault(ev.evidence_id, ev)
    uniq_paper: dict[str, PaperRecord] = {}
    for p in papers:
        uniq_paper.setdefault(p.paper_id, p)
    card_map: dict[str, PaperCard] = {}
    for c in cards:
        card_map.setdefault(c.paper_id, c)

    conflict_ev_ids = {eid for c in conflicts for eid in c.evidence_ids}

    # 2) 任务相关打分排序（稳定排序：同分保持输入顺序）
    scored = sorted(uniq_ev.values(), key=lambda e: _score_evidence(e, task, conflict_ev_ids), reverse=True)

    # 3) blog 且无原文 PDF 的论文降级到“线索”区
    core_papers = [p for p in uniq_paper.values() if not (p.paper_type == "blog" and not p.pdf_path)]
    lead_papers = [p for p in uniq_paper.values() if p.paper_type == "blog" and not p.pdf_path]

    # 4) 渲染各行
    header = _format_header(task, brief)
    cf_lines = [_format_conflict_line(c) for c in conflicts]
    ev_lines = [_format_evidence_line(e, conflict_ev_ids) for e in scored]
    card_lines = [_format_card_line(p, card_map.get(p.paper_id)) for p in core_papers]
    lead_lines = [_format_lead_line(p) for p in lead_papers]

    # 5) 预算分配：头部 → 冲突点（优先，至多 1/4 预算）→ 关键证据 → 卡片 → 线索 → 附注。
    #    逐条贪心：放不下就整条丢弃，绝不截断半条。
    used = estimate_tokens(header)

    kept_cf: list[str] = []
    dropped_cf = 0
    cf_cap = budget // 4
    cf_used = 0
    for line in cf_lines:
        t = estimate_tokens(line) + 1
        if cf_used + t <= cf_cap:
            kept_cf.append(line)
            cf_used += t
        else:
            dropped_cf += 1
    used += cf_used

    reserve = max(200, budget * 15 // 100)  # 为卡片/线索/附注预留
    kept_ev: list[str] = []
    dropped_ev = 0
    ev_used = 0
    for line in ev_lines:
        t = estimate_tokens(line) + 1
        if ev_used + t <= budget - used - reserve:
            kept_ev.append(line)
            ev_used += t
        else:
            dropped_ev += 1
    used += ev_used

    kept_cards: list[str] = []
    dropped_cards = 0
    for line in card_lines:
        t = estimate_tokens(line) + 1
        if used + t <= budget - _NOTE_RESERVE:
            kept_cards.append(line)
            used += t
        else:
            dropped_cards += 1

    kept_leads: list[str] = []
    dropped_leads = 0
    for line in lead_lines:
        t = estimate_tokens(line) + 1
        if used + t <= budget - _NOTE_RESERVE:
            kept_leads.append(line)
            used += t
        else:
            dropped_leads += 1

    notes: list[str] = [f"- token 预算 {budget}"]
    if extra:
        notes.append(f"- {extra}")
    drops: list[str] = []
    if dropped_cf:
        drops.append(f"{dropped_cf} 条冲突")
    if dropped_ev:
        drops.append(f"{dropped_ev} 条证据")
    if dropped_cards:
        drops.append(f"{dropped_cards} 篇论文卡片")
    if dropped_leads:
        drops.append(f"{dropped_leads} 条线索")
    if drops:
        notes.append("- 已按预算整条丢弃（未截断任何条目）：" + "、".join(drops) + "，未纳入部分不参与本次推理。")

    parts: list[str] = []
    if header:
        parts.append(header)
    if kept_cards:
        parts.append("## 论文卡片（压缩）\n" + "\n".join(kept_cards))
    if kept_ev:
        parts.append("## 关键证据\n" + "\n".join(kept_ev))
    if kept_cf:
        parts.append("## 冲突点\n" + "\n".join(kept_cf))
    if kept_leads:
        parts.append("## 线索（无原文来源，不作为直接证据）\n" + "\n".join(kept_leads))
    parts.append("## 附注\n" + "\n".join(notes))
    return "\n\n".join(parts) + "\n"


def compress_findings(findings: list[Finding], max_chars: int = 2000) -> str:
    """Finding 列表 → 每任务一行“task_id | 状态 | 摘要 | 工具调用数”，总长截断到 max_chars。"""
    if not findings or max_chars <= 0:
        return ""
    lines: list[str] = []
    used = 0
    dropped = 0
    for f in findings:
        summary = truncate((f.summary or "").replace("\n", " "), 120)
        if f.error:
            summary = f"{summary}（错误:{truncate(f.error, 40)}）"
        line = f"{f.task_id} | {f.status} | {summary} | {f.tool_calls} 次工具调用"
        need = len(line) + (1 if lines else 0)
        if used + need > max_chars - 16:  # 预留截断提示
            dropped += 1
            continue
        lines.append(line)
        used += need
    out = "\n".join(lines)
    if dropped:
        note = f"…（其余 {dropped} 条省略）"
        out = f"{out}\n{note}" if len(out) + 1 + len(note) <= max_chars else truncate(out, max_chars)
    return out if len(out) <= max_chars else truncate(out, max_chars)
