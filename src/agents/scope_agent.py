"""Scope Agent：把用户原始问题固化成 ResearchBrief（研究范围与判断标准）。

设计要点（文档 §7.1）：
- 只有"研究范围会显著改变结果"时才置 clarification_needed 并给出澄清问题；
  其余情况一律填默认值，并把采用的默认假设写入 assumptions；
- year_from / year_to 与 time_range 的一致性由代码兜底：缺省 year_to=当前年、
  year_from=year_to-5，time_range 缺省时由年份推导；
- LLM 失败（含 JSONValidationError）时返回规则兜底的默认 Brief 并记 trace error，
  绝不让整个流程因 scope 失败而中断。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.llm import get_llm
from src.schemas import ResearchBrief
from src.tracing import get_active_tracer

_NODE = "scope"

#: 默认判断标准：效果 / 成本 / 数据规模 / 可复现性（文档 §7.1）
_DEFAULT_CRITERIA: dict[str, str] = {
    "效果": "与基线相比的提升幅度及统计显著性",
    "成本": "训练/推理算力与数据获取成本",
    "数据规模": "训练与评测数据的量级与构成",
    "可复现性": "代码、超参与随机种子是否公开，结果能否复现",
}

_SCOPE_SYSTEM = """#FAKE:scope
你是科研助手系统的 Scope Agent（orchestrator 角色），负责把用户的原始问题转写为一份可执行的研究简报（Research Brief）。

输出字段：
- objective：研究目标（一句话，尽量保留用户原始意图）
- out_of_scope：明确不研究的内容
- time_range：时间范围描述（如 "2021-2026"）
- year_from / year_to：时间范围上下界（整数年份，必须与 time_range 一致）
- disciplines：学科范围
- paper_types：论文类型（survey / full-paper / preprint 等）
- core_questions：需要回答的核心问题（3-6 个，必须可检索、可判定完成）
- output_format：输出格式（默认 markdown）
- depth：quick / standard / deep
- criteria：判断标准字典，至少覆盖 效果 / 成本 / 数据规模 / 可复现性
- language：报告语言（默认 zh）
- clarification_needed / clarification_question：是否需要向用户澄清及澄清问题
- assumptions：未询问用户时填入的默认假设（字符串列表）

关键规则：
1. 只有当"研究范围会显著改变结果"时才置 clarification_needed=true 并给出 clarification_question，
   例如：综述全领域 vs 单方法深挖、时间范围跨代际、学科口径完全不同；
2. 其余情况一律 clarification_needed=false，把采用的默认值写入 assumptions，
   例如："默认近 5 年"、"默认聚焦方法与实验证据"、"默认以同行评审论文为主"；
3. 用户未指定时间范围时按近 5 年填写 year_from/year_to；
4. 不要发明用户没有提到的限制，也不要过度收窄范围。"""


def _finalize_years(brief: ResearchBrief, added: list[str]) -> None:
    """代码兜底：保证 year_from/year_to/time_range 三者一致且完整。"""
    now_year = datetime.now().year
    if brief.year_to is None:
        brief.year_to = now_year
    if brief.year_from is None:
        brief.year_from = brief.year_to - 5
        added.append("默认近 5 年（未指定时间范围）")
    if brief.year_from > brief.year_to:
        brief.year_from, brief.year_to = brief.year_to, brief.year_from
        added.append("year_from 晚于 year_to，已自动对调")
    if not brief.time_range:
        brief.time_range = f"{brief.year_from}-{brief.year_to}"


def _fallback_brief(user_query: str, papers_dir: str | None, reason: str) -> ResearchBrief:
    """LLM 失败时的规则兜底 Brief：以用户问题本身为唯一核心问题。"""
    now_year = datetime.now().year
    return ResearchBrief(
        objective=user_query,
        core_questions=[user_query],
        year_from=now_year - 5,
        year_to=now_year,
        time_range=f"{now_year - 5}-{now_year}",
        paper_types=["survey", "full-paper", "preprint"] + (["local"] if papers_dir else []),
        papers_dir=papers_dir,
        criteria=dict(_DEFAULT_CRITERIA),
        assumptions=[f"scope 降级：{reason}", "默认近 5 年", "默认聚焦方法与实验证据"],
    )


def run_scope(
    user_query: str,
    *,
    papers_dir: str | None = None,
    llm: Any = None,
) -> ResearchBrief:
    """把用户问题转写为 ResearchBrief。

    Args:
        user_query: 用户原始问题。
        papers_dir: 本地论文目录（非空时写入 brief.papers_dir，并在 paper_types
            中补充 "local"，提示研究优先覆盖本地论文）。
        llm: LLM 客户端（duck-typing，需提供 chat_json）；缺省 get_llm("orchestrator")。

    Returns:
        ResearchBrief。LLM 失败时返回规则兜底的默认 Brief（assumptions 注明 scope 降级），
        绝不抛出异常中断流程。
    """
    tracer = get_active_tracer()
    tracer.event("node_start", node=_NODE, user_query=user_query)
    client = llm if llm is not None else get_llm("orchestrator")

    system = _SCOPE_SYSTEM
    if papers_dir:
        system += (
            f"\n\n用户提供了本地论文目录：{papers_dir}。研究需优先覆盖这些本地论文，"
            "paper_types 中应包含 \"local\"，并在 assumptions 中注明该默认。"
        )
    prompt = f"用户问题：{user_query}\n当前年份：{datetime.now().year}。"

    try:
        brief = client.chat_json(prompt, ResearchBrief, system=system, label="scope")
        brief.objective = brief.objective or user_query
        added: list[str] = []
        if papers_dir:
            brief.papers_dir = papers_dir
            if not any("local" in (t or "").lower() for t in brief.paper_types):
                brief.paper_types.append("local")
                added.append("用户提供了本地论文目录，paper_types 已补充 local")
        if not brief.core_questions:
            brief.core_questions = [brief.objective]
            added.append("LLM 未给出核心问题，已以研究目标兜底")
        _finalize_years(brief, added)
        if not brief.criteria:
            brief.criteria = dict(_DEFAULT_CRITERIA)
            added.append("默认聚焦方法与实验证据（判断标准：效果/成本/数据规模/可复现性）")
        for note in added:
            if note not in brief.assumptions:
                brief.assumptions.append(note)
        tracer.event(
            "node_end",
            node=_NODE,
            objective=brief.objective,
            n_questions=len(brief.core_questions),
            clarification_needed=brief.clarification_needed,
        )
        return brief
    except Exception as exc:  # noqa: BLE001 —— scope 绝不中断整个流程
        reason = f"LLM 解析失败（{type(exc).__name__}: {str(exc)[:200]}）"
        tracer.event("error", node=_NODE, error=reason)
        brief = _fallback_brief(user_query, papers_dir, reason)
        tracer.event("node_end", node=_NODE, objective=brief.objective, degraded=True)
        return brief
