"""Perspective Planner：STORM 式多视角问题分解（文档 §7.2）。

流程：
1. 无种子论文且允许联网时，懒加载检索工具搜 3-5 篇综述/高引用入口论文
   （工具未实现或失败 → 空列表继续，绝不崩）；
2. 一次 chat_json：brief 摘要 + 种子论文题名 → 分析维度 + 五视角子问题草案；
3. 后处理：perspective 归一化（非法值按问题关键词猜）、证据/成功标准模板补全、
   token_set_ratio 语义去重、五视角缺失补模板问题、截断到 max_tasks；
4. LLM 失败 → 按五视角各生成一个模板任务（objective 填充），tracer 记 error。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from src.llm import get_llm
from src.schemas import PaperRecord, ResearchBrief, ResearchTask
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import norm_title

_NODE = "planner"

#: 五类视角（与 schemas.Perspective 一致）
_PERSPECTIVES: tuple[str, ...] = ("background", "method", "experiment", "application", "critique")

#: 各视角缺省所需证据类型
_PERSPECTIVE_EVIDENCE: dict[str, list[str]] = {
    "background": ["综述式的问题定义与研究脉络", "代表工作的原文描述"],
    "method": ["方法机制描述（原文片段）", "与已有方法的机制差异说明"],
    "experiment": ["基准数据集与指标（含表格原文）", "消融或对照实验结果"],
    "application": ["真实场景部署或案例描述", "落地条件（数据/算力/合规）"],
    "critique": ["反例或负结果论文", "无消融支撑的归因点", "撤稿/勘误记录"],
}

#: 各视角缺省成功标准
_PERSPECTIVE_SUCCESS: dict[str, str] = {
    "background": "能说清该方向的问题定义、发展脉络与代表工作",
    "method": "能比较至少 2 种方法的机制、适用条件与代价",
    "experiment": "能给出统一口径的定量对比，并标注评测设置差异",
    "application": "能列出适用场景、部署条件与已知失败案例",
    "critique": "至少指出 1 个主流结论的反例、无消融支撑的归因或撤稿记录",
}

#: 五视角缺失时补的模板问题（{objective} 占位）
_PERSPECTIVE_QUESTION: dict[str, str] = {
    "background": "{objective}的问题定义、发展脉络与代表工作是什么？",
    "method": "{objective}中的主流方法各自解决什么问题、机制差异与代价是什么？",
    "experiment": "{objective}的主流方法在哪些基准上评测、指标口径是否一致？",
    "application": "{objective}的方法在哪些真实场景落地、需要什么部署条件？",
    "critique": "针对{objective}的主流结论，存在哪些反例、无消融支撑的归因或撤稿记录？",
}

#: perspective 非法时按问题关键词猜（顺序即优先级，命中即返回）
_PERSPECTIVE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("method", ("对比", "比较", "方法", "机制", "compare", "method", "mechanism")),
    ("experiment", ("基准", "评测", "实验", "消融", "benchmark", "evaluation", "ablation")),
    ("critique", ("风险", "局限", "争议", "反例", "撤稿", "失败", "risk", "limitation", "controvers", "retract")),
    ("application", ("应用", "落地", "部署", "场景", "application", "deployment")),
)

#: perspective 中英同义别名（模型输出常见叫法）
_PERSPECTIVE_ALIASES: tuple[tuple[str, str], ...] = (
    ("背景", "background"),
    ("综述", "background"),
    ("方法", "method"),
    ("实验", "experiment"),
    ("评测", "experiment"),
    ("应用", "application"),
    ("批判", "critique"),
    ("批评", "critique"),
    ("争议", "critique"),
)

#: 问题语义去重阈值（token_set_ratio / 100 > 阈值 → 丢后出现的）
_DEDUP_THRESHOLD = 0.8

#: 与 schemas.ResearchTask 默认白名单一致
_DEFAULT_ALLOWED_TOOLS: list[str] = [
    "search_arxiv",
    "search_semantic_scholar",
    "search_web_literature",
    "traverse_citations",
    "download_pdf",
    "read_pdf",
]


class _DraftTask(BaseModel):
    """LLM 草案中的子问题（perspective 故意为 str，宽松接收后统一归一化）。"""

    question: str
    perspective: str = "background"
    required_evidence: list[str] = Field(default_factory=list)
    success_criteria: str | None = None


class _PlannerDraft(BaseModel):
    dimensions: list[str] = Field(default_factory=list)
    tasks: list[_DraftTask] = Field(default_factory=list)


_PLANNER_SYSTEM = """#FAKE:planner
你是 Perspective Planner（orchestrator 角色），复现 STORM 的多视角提问思想：先从种子论文中提炼领域常见分析维度，再生成互不重复的研究子问题。

五类视角（perspective 只能取以下之一）及各自要什么证据：
- background（背景）：问题定义、发展脉络、代表工作；需要综述式证据与研究脉络描述；
- method（方法）：机制差异、适用条件、代价；需要方法机制描述原文与机制对比；
- experiment（实验）：基准、指标、评测口径；需要基准数据集/指标表格原文与消融、对照实验结果；
- application（应用）：落地场景与部署条件；需要真实部署案例与落地条件描述；
- critique（批判）：反例、无消融的归因、撤稿/勘误；需要反例或负结果论文、无消融支撑的归因点、撤稿/勘误记录。

输出 JSON 字段：
- dimensions：从种子论文提炼的分析维度（3-8 个短语）；
- tasks：子问题列表，每项含 question / perspective / required_evidence / success_criteria。

要求：
1. 每个问题必须可检索、可判定完成，不要写成宽泛口号；
2. 问题之间语义不重复；
3. 五类视角尽量都有覆盖（除非目标明显不适用）；
4. required_evidence 写明证据类型（如"消融结果表格原文"），success_criteria 写明完成标准。"""


# --------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------
def _normalize_perspective(value: str, question: str) -> str:
    """把任意 perspective 输出归一化为五类合法值；非法时按问题关键词猜。"""
    v = (value or "").strip().lower()
    if v in _PERSPECTIVES:
        return v
    for alias, target in _PERSPECTIVE_ALIASES:
        if alias in v:
            return target
    q = (question or "").lower()
    for target, keywords in _PERSPECTIVE_KEYWORDS:
        if any(k.lower() in q for k in keywords):
            return target
    return "background"


def _is_duplicate(question: str, existing: list[str]) -> bool:
    """token_set_ratio 与已有问题任一相似度 > 0.8 → 视为重复。"""
    return any(fuzz.token_set_ratio(question, other) / 100.0 > _DEDUP_THRESHOLD for other in existing)


def _search_seed_papers(brief: ResearchBrief) -> list[PaperRecord]:
    """懒加载检索工具，搜 3-5 篇综述/高引用入口论文；任何失败返回空列表。"""
    tracer = get_active_tracer()
    base = (brief.objective or "").strip() or (brief.core_questions[0] if brief.core_questions else "")
    if not base:
        return []
    try:
        from src.tools.arxiv_tool import search_arxiv
        from src.tools.semantic_scholar_tool import search_semantic_scholar
    except Exception as exc:  # noqa: BLE001 —— 工具模块缺失不应中断规划
        tracer.event("error", node=_NODE, error=f"种子检索工具导入失败: {type(exc).__name__}: {exc}")
        return []

    found: dict[str, PaperRecord] = {}
    year = None
    if brief.year_from or brief.year_to:
        year = f"{brief.year_from or ''}-{brief.year_to or ''}".strip("-") or None
    for suffix in ("survey", "review"):
        query = f"{base} {suffix}"
        for source in ("arxiv", "s2"):
            try:
                if source == "arxiv":
                    results = search_arxiv(
                        query,
                        max_results=3,
                        date_from=f"{brief.year_from}-01-01" if brief.year_from else None,
                        date_to=f"{brief.year_to}-12-31" if brief.year_to else None,
                    )
                else:
                    results = search_semantic_scholar(query, max_results=3, year=year)
                for p in results or []:
                    key = p.paper_id or norm_title(p.title)
                    if key and key not in found:
                        found[key] = p
            except Exception as exc:  # noqa: BLE001 —— 含 NotImplementedError（工具未实现）
                tracer.event("error", node=_NODE, error=f"种子检索失败（{source}/{suffix}）: {type(exc).__name__}: {exc}")
    # 综述优先，其次按被引量降序，最多 5 篇
    ranked = sorted(found.values(), key=lambda p: (p.paper_type != "survey", -(p.citation_count or 0)))
    return ranked[:5]


def _planner_prompt(brief: ResearchBrief, seeds: list[PaperRecord]) -> str:
    lines = [
        "研究简报摘要：",
        f"- 研究目标：{brief.objective}",
        f"- 时间范围：{brief.time_range or '未指定'}",
        f"- 学科范围：{'、'.join(brief.disciplines) or '未指定'}",
        f"- 论文类型：{'、'.join(brief.paper_types) or '未指定'}",
        f"- 深度：{brief.depth}",
    ]
    if brief.core_questions:
        lines.append("- 核心问题：")
        lines.extend(f"  {i + 1}. {q}" for i, q in enumerate(brief.core_questions))
    if brief.criteria:
        lines.append(f"- 判断标准：{brief.criteria}")
    if seeds:
        lines.append("种子论文（综述/高引用入口，供提炼维度，不要照抄题名当问题）：")
        lines.extend(
            f"  {i + 1}. {p.title}（{p.year or '年份未知'}, {p.paper_type}, 被引 {p.citation_count or 0}）"
            for i, p in enumerate(seeds)
        )
    else:
        lines.append("种子论文：无（离线或检索失败），请直接依据研究简报分解。")
    lines.append("请输出 dimensions 与 tasks。")
    return "\n".join(lines)


def _to_task(
    draft: _DraftTask,
    *,
    round_no: int,
    tool_budget: int,
) -> ResearchTask:
    """把草案转为 ResearchTask：归一化 perspective 并补全证据/成功标准模板。"""
    perspective = _normalize_perspective(draft.perspective, draft.question)
    return ResearchTask(
        question=draft.question,
        perspective=perspective,
        required_evidence=list(draft.required_evidence) or list(_PERSPECTIVE_EVIDENCE[perspective]),
        success_criteria=draft.success_criteria or _PERSPECTIVE_SUCCESS[perspective],
        round_created=round_no,
        origin="planner",
        max_tool_calls=tool_budget,
        allowed_tools=list(_DEFAULT_ALLOWED_TOOLS),
    )


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def run_perspective_planner(
    brief: ResearchBrief,
    *,
    llm: Any = None,
    seed_papers: list[PaperRecord] | None = None,
    max_tasks: int | None = None,
    round_no: int = 0,
    online: bool = True,
) -> list[ResearchTask]:
    """把 ResearchBrief 分解为多视角子任务列表。

    Args:
        brief: Scope Agent 产出的研究简报。
        llm: LLM 客户端（duck-typing，需提供 chat_json）；缺省 get_llm("orchestrator")。
        seed_papers: 显式传入的种子论文；None 且 online=True 时函数内自动检索。
        max_tasks: 任务数上限；缺省取 budgets.max_tasks_initial。
        round_no: 当前研究轮次，写入 round_created。
        online: 是否允许联网检索种子论文。

    Returns:
        list[ResearchTask]。LLM 失败时返回五视角模板任务兜底，绝不抛出异常。
    """
    tracer = get_active_tracer()
    tracer.event("node_start", node=_NODE, round=round_no, online=online, has_seed=seed_papers is not None)
    settings = get_settings()
    limit = int(max_tasks if max_tasks is not None else settings.budget("max_tasks_initial", 8))
    limit = max(1, limit)  # 至少 1 个任务，避免空计划让图循环空转
    tool_budget = int(settings.budget("max_tool_calls_per_researcher", 5))

    if seed_papers is not None:
        seeds = list(seed_papers)
    elif online:
        seeds = _search_seed_papers(brief)
    else:
        seeds = []
    if seeds:
        tracer.event("seed_papers", node=_NODE, n=len(seeds))

    client = llm if llm is not None else get_llm("orchestrator")
    raw_tasks: list[_DraftTask] = []
    degraded = False
    try:
        draft = client.chat_json(
            _planner_prompt(brief, seeds),
            _PlannerDraft,
            system=_PLANNER_SYSTEM + f"\n\n本轮最多生成 {limit} 个任务。",
            label="planner",
        )
        raw_tasks = list(draft.tasks)
    except Exception as exc:  # noqa: BLE001 —— 规划失败走模板兜底
        degraded = True
        tracer.event("error", node=_NODE, error=f"LLM 解析失败（{type(exc).__name__}: {str(exc)[:200]}），使用五视角模板兜底")

    objective = brief.objective or (brief.core_questions[0] if brief.core_questions else "该研究方向")

    # 去重 + 转换
    tasks: list[ResearchTask] = []
    seen_questions: list[str] = []
    for d in raw_tasks:
        question = (d.question or "").strip()
        if not question or _is_duplicate(question, seen_questions):
            continue
        seen_questions.append(question)
        d.question = question
        tasks.append(_to_task(d, round_no=round_no, tool_budget=tool_budget))

    # 五视角缺失且有余量 → 补模板问题
    present = {t.perspective for t in tasks}
    for perspective in _PERSPECTIVES:
        if perspective in present or len(tasks) >= limit:
            continue
        question = _PERSPECTIVE_QUESTION[perspective].format(objective=objective)
        if _is_duplicate(question, seen_questions):
            continue
        seen_questions.append(question)
        tasks.append(
            _to_task(
                _DraftTask(question=question, perspective=perspective),
                round_no=round_no,
                tool_budget=tool_budget,
            )
        )
        tracer.event("perspective_backfill", node=_NODE, perspective=perspective)

    # LLM 完全失败/空输出且上面没补出任何任务 → 五视角模板兜底
    if not tasks:
        for perspective in _PERSPECTIVES:
            tasks.append(
                _to_task(
                    _DraftTask(
                        question=_PERSPECTIVE_QUESTION[perspective].format(objective=objective),
                        perspective=perspective,
                    ),
                    round_no=round_no,
                    tool_budget=tool_budget,
                )
            )

    tasks = tasks[:limit]
    tracer.event(
        "node_end",
        node=_NODE,
        n_tasks=len(tasks),
        perspectives=sorted({t.perspective for t in tasks}),
        degraded=degraded,
    )
    return tasks
