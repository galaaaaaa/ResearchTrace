"""Supervisor：纯规则的任务治理（不读论文、不调 LLM，文档 §7.3）。

职责：
- 判定任务完成情况（Finding status ∈ {done, partial} 视为完成，failed 视为已处理，
  交给 Gap Analyzer 决策是否重派）；
- 每轮从 pending 任务中按 perspective 轮转交错取前 max_concurrent 个，
  保证并行分支视角多样；
- detect_stagnation：最近两次搜索词高度重复时提示提前停止（搜索层面复用）。

注意：next_batch 是纯选择、不改任务状态；由图编排层负责把选中任务标记为 running。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from rapidfuzz import fuzz

from src.schemas import Finding, ResearchTask
from src.settings import get_settings
from src.tracing import get_active_tracer

_NODE = "supervisor"

#: 视为"已处理"的 Finding 状态（done/partial=完成；failed=已处理，交 Gap Analyzer 决策）
_PROCESSED_STATUSES: tuple[str, ...] = ("done", "partial", "failed")

_CJK_CHAR = re.compile(r"([一-鿿])")


class Supervisor:
    """任务治理器：完成判定 + 并发批次派发。纯规则，不调用 LLM。"""

    def __init__(self, *, budgets: Any = None):
        """Args:
            budgets: 预算来源。None → 全局 get_settings()；dict → 取 defaults 键；
                也接受提供 budget(key, default) 方法的对象（如 Settings 实例）。
        """
        self._budgets = budgets

    # ---- 预算读取 ----
    def _budget(self, key: str, default: Any) -> Any:
        b = self._budgets
        if b is None:
            return get_settings().budget(key, default)
        if isinstance(b, dict):
            return b.get("defaults", b).get(key, default)
        getter = getattr(b, "budget", None)
        return getter(key, default) if callable(getter) else default

    # ---- 完成判定 ----
    @staticmethod
    def _processed_task_ids(findings: Iterable[Finding]) -> set[str]:
        return {f.task_id for f in findings if f.status in _PROCESSED_STATUSES}

    def next_batch(
        self,
        tasks: list[ResearchTask],
        findings: list[Finding],
        research_round: int,
        *,
        max_concurrent: int | None = None,
    ) -> list[ResearchTask]:
        """选出本轮派发的任务批次。

        已完成（done/partial）或已处理（failed）的任务不再派发；剩余 pending 任务
        按 perspective 分组后轮转交错（round-robin），取前 max_concurrent 个，
        保证同一批并行分支视角尽量不同。

        Args:
            tasks: 全部任务（含各轮 planner/gap 产物）。
            findings: 当前全部 Finding（按 task_id 匹配）。
            research_round: 当前研究轮次（仅用于追踪）。
            max_concurrent: 并发上限；缺省取 budgets.max_concurrent_researchers。

        Returns:
            本轮派发的 ResearchTask 列表（不修改任务对象状态）。
        """
        tracer = get_active_tracer()
        tracer.event("node_start", node=_NODE, round=research_round, n_tasks=len(tasks))
        limit = int(
            max_concurrent if max_concurrent is not None else self._budget("max_concurrent_researchers", 4)
        )
        processed = self._processed_task_ids(findings)
        pending = [t for t in tasks if t.task_id not in processed]

        # 按 perspective 分组（保持任务原顺序），再按组轮转交错
        groups: dict[str, list[ResearchTask]] = {}
        for t in pending:
            groups.setdefault(t.perspective, []).append(t)
        interleaved: list[ResearchTask] = []
        if groups:
            columns = list(groups.values())
            for i in range(max(len(col) for col in columns)):
                for col in columns:
                    if i < len(col):
                        interleaved.append(col[i])

        batch = interleaved[: max(0, limit)]
        tracer.event(
            "node_end",
            node=_NODE,
            round=research_round,
            n_pending=len(pending),
            n_batch=len(batch),
            perspectives=sorted({t.perspective for t in batch}),
        )
        return batch

    def all_done(self, tasks: list[ResearchTask], findings: list[Finding]) -> bool:
        """所有任务都已有已处理的 Finding（done/partial/failed）。

        空任务列表视为全部完成（无可派发内容，图应直接进入下一阶段）。
        """
        processed = self._processed_task_ids(findings)
        return all(t.task_id in processed for t in tasks)


# --------------------------------------------------------------------------
# 搜索停滞检测（模块级，供 Researcher/Searcher 复用）
# --------------------------------------------------------------------------
def _prep_query(query: str) -> str:
    """相似度预处理：CJK 逐字切开（补 whitespace 分词短板）、压缩空白、小写。"""
    spaced = _CJK_CHAR.sub(r" \1 ", query or "")
    return re.sub(r"\s+", " ", spaced).strip().lower()


def detect_stagnation(queries: list[str], *, threshold: float | None = None) -> bool:
    """最近两次搜索词高度重复 → True（提示提前停止，避免同义查询刷预算）。

    相似度取 token_set_ratio 与 partial_token_set_ratio 的较大值：
    前者覆盖同词重排，后者覆盖形态变体/包含关系（如 "grpo forget" → "grpo forgetting"
    没有引入任何新检索意图）。中文按字符切词后比较。

    Args:
        queries: 按时间顺序累计的查询词列表（取最后两条比较）。
        threshold: 相似度阈值（0-1）；缺省取 budgets.similar_query_stop。
    """
    if threshold is None:
        threshold = float(get_settings().budget("similar_query_stop", 0.85))
    if len(queries) < 2:
        return False
    a, b = _prep_query(queries[-2]), _prep_query(queries[-1])
    if not a or not b:
        return False
    score = max(
        fuzz.token_set_ratio(a, b) / 100.0,
        fuzz.partial_token_set_ratio(a, b) / 100.0,
    )
    return score >= threshold
