"""LangGraph 图编排：Scope → Plan → Parallel Research → Gap → Outline → Write → Verify → Repair/Finalize。

可靠性设计：
- 固定状态图负责主流程；Researcher 内部保留受限 ReAct 循环（见 agents/searcher.py）；
- research_round ≤ 2、repair_round ≤ 2，达到上限仍缺证据 → 报告中明确标注"证据不足/结论冲突"，不再循环；
- 所有节点包一层 node 装饰器：trace + 异常兜底（researcher 崩溃时写入 failed Finding 防止死循环）；
- 并行 Researcher 通过 Send fan-out，payload 自包含（worker 读不到共享 channel，已在集成时验证）。
"""

from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .llm import get_global_budget, get_llm
from .schemas import (
    ClaimRecord,
    EvidenceRecord,
    Finding,
    PaperCard,
    PaperRecord,
    ResearchBrief,
    ResearchTask,
)
from .settings import get_settings
from .state import ResearchState
from .tracing import get_active_tracer
from .utils import hash_file, now_iso, truncate


# --------------------------------------------------------------------------
# 节点装饰器：trace + 兜底
# --------------------------------------------------------------------------
def node(fn: Callable) -> Callable:
    name = fn.__name__

    @wraps(fn)
    def wrapped(state: dict) -> dict:
        tracer = get_active_tracer()
        tracer.event("node_start", node=name)
        t0 = time.time()
        try:
            update = fn(state) or {}
        except Exception as exc:  # noqa: BLE001
            tracer.event("node_error", node=name, error=f"{type(exc).__name__}: {exc}")
            update = {"errors": [f"{name}: {type(exc).__name__}: {truncate(str(exc), 300)}"]}
            # researcher 崩溃必须落 failed Finding，否则任务永远 pending → 死循环
            task = state.get("current_task") if isinstance(state, dict) else None
            if isinstance(task, dict) and task.get("task_id"):
                update["findings"] = [
                    Finding(task_id=task["task_id"], agent="researcher", status="failed", error=truncate(str(exc), 300))
                ]
        update.setdefault("errors", [])
        tracer.event("node_end", node=name, duration_ms=int((time.time() - t0) * 1000), keys=sorted(update.keys()))
        return update

    return wrapped


# --------------------------------------------------------------------------
# 本地 PDF 扫描（离线种子论文）
# --------------------------------------------------------------------------
def scan_local_pdfs(papers_dir: str | None) -> list[PaperRecord]:
    if not papers_dir:
        return []
    root = Path(papers_dir)
    if not root.exists():
        return []
    papers: list[PaperRecord] = []
    for pdf in sorted(root.glob("*.pdf")):
        digest = hash_file(pdf)[:12]
        title = pdf.stem.replace("_", " ").replace("-", " ")
        try:  # extract_title 若可用则覆盖文件名猜测
            from .tools.pdf_tool import extract_title

            better = extract_title(str(pdf))
            if better:
                title = better
        except Exception:  # noqa: BLE001  # NotImplementedError / 解析失败
            pass
        papers.append(
            PaperRecord(
                paper_id=f"sha256:{digest}",
                title=title,
                source_url=f"file://{pdf.resolve()}",
                pdf_path=str(pdf.resolve()),
                source_api="local",
                paper_type="full-paper",
            )
        )
    return papers


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------
def build_graph(*, settings=None):
    s = settings or get_settings()
    budgets = s.budgets.get("defaults", {})

    # ---- Scope ----
    @node
    def scope(state: ResearchState) -> dict:
        from .agents.scope_agent import run_scope

        brief = run_scope(state["user_query"], papers_dir=state.get("papers_dir"), llm=get_llm("orchestrator"))

        # 知识库预热（跨 run 知识传递）：按用户问题检索历史证据，配套论文（自带
        # pdf_path，不重复下载）注入 state；searcher 对 existing 去重 → Reader 不重读。
        # KB 不可用/无命中 → 空注入，研究照常从零。
        kb_evs: list = []
        kb_papers: list = []
        try:
            from .memory.vector_store import prime_from_kb

            kb_evs, kb_papers = prime_from_kb(state["user_query"])
            if kb_evs:
                get_active_tracer().event(
                    "kb_prime", node="scope", evidence=len(kb_evs), papers=len(kb_papers)
                )
        except Exception as exc:  # noqa: BLE001 —— 预热失败不影响研究
            get_active_tracer().event("tool_error", node="scope", tool="kb_prime", error=str(exc)[:200])
        return {"brief": brief, "evidence": kb_evs, "papers": kb_papers}

    # ---- Perspective Planner ----
    @node
    def planner(state: ResearchState) -> dict:
        from .agents.perspective_planner import run_perspective_planner

        brief: ResearchBrief = state["brief"]
        seeds = scan_local_pdfs(state.get("papers_dir"))
        online = bool(s.source("arxiv", "enabled", True) or s.source("semantic_scholar", "enabled", True))
        tasks = run_perspective_planner(
            brief, seed_papers=seeds or None, llm=get_llm("orchestrator"),
            max_tasks=int(budgets.get("max_tasks_initial", 8)), online=online,
        )
        return {"tasks": tasks, "papers": seeds}

    # ---- Supervisor（派发路由）----
    def dispatch(state: ResearchState):
        from .agents.supervisor import Supervisor

        sup = Supervisor(budgets=s.budgets)
        tasks: list[ResearchTask] = state.get("tasks") or []
        findings = state.get("findings") or []
        batch = sup.next_batch(tasks, findings, state.get("research_round", 0))
        if not batch or get_global_budget().exceeded:
            return "gap_analyzer"
        brief = state.get("brief")
        papers = state.get("papers") or []
        payload_base = {
            "user_query": state.get("user_query", ""),
            "run_id": state.get("run_id", ""),
            "brief": brief.model_dump() if isinstance(brief, ResearchBrief) else (brief or None),
            "existing_papers": [p.model_dump() if isinstance(p, PaperRecord) else p for p in papers],
        }
        return [
            Send("researcher", {**payload_base, "current_task": t.model_dump()})
            for t in batch
        ]

    # ---- Researcher（并行执行）----
    @node
    def researcher(state: ResearchState) -> dict:
        from .agents.analyst import build_comparison
        from .agents.critic import find_conflicts
        from .agents.reader import Reader
        from .agents.searcher import Searcher

        task = ResearchTask.model_validate(state["current_task"])
        brief = ResearchBrief.model_validate(state["brief"]) if state.get("brief") else None
        existing = [p if isinstance(p, PaperRecord) else PaperRecord.model_validate(p) for p in (state.get("existing_papers") or [])]

        # 1) 受限 ReAct 搜索（工具不可用/离线时返回空结果，不崩）
        searcher = Searcher(budgets=s.budgets)
        sr = searcher.discover(task, brief, existing_papers=existing)
        get_active_tracer().event("search_done", node="researcher", task_id=task.task_id,
                                  papers=len(sr.papers), tool_calls=len(sr.tool_logs), reason=sr.stopped_reason)

        # 2) 精读候选：新发现（含 PDF）优先；离线/搜索空时退回本地论文（题名关键词匹配）
        candidates = [p for p in sr.papers if p.pdf_path]
        if not candidates and existing:
            kws = {w.lower() for w in task.question.split() if len(w) > 3}
            scored = [
                (len(kws & {w.lower() for w in p.title.split()}), p)
                for p in existing if p.pdf_path
            ]
            candidates = [p for score, p in sorted(scored, key=lambda x: -x[0]) if score > 0] or [
                p for p in existing if p.pdf_path
            ][:2]
        max_read = 4
        read_papers: list[PaperRecord] = []
        evidence: list[EvidenceRecord] = []
        cards: list[PaperCard] = []
        reader = Reader(budgets=s.budgets)
        for paper in candidates[:max_read]:
            try:
                rr = reader.read_paper(paper, task, llm=get_llm("researcher"))
            except Exception as exc:  # noqa: BLE001
                get_active_tracer().event("reader_error", node="researcher", paper_id=paper.paper_id, error=str(exc))
                continue
            read_papers.append(paper)
            evidence.extend(rr.evidence)
            cards.append(rr.card)

        # 3) 分析与批判（按视角）
        comparisons, conflicts = [], []
        try:
            if task.perspective in ("method", "experiment") and len(cards) >= 2:
                comparisons.append(build_comparison(task, read_papers, evidence, cards, llm=get_llm("analyst")))
            if task.perspective == "critique" and cards:
                conflicts = find_conflicts(task, read_papers, evidence, cards, llm=get_llm("critic"))
        except Exception as exc:  # noqa: BLE001
            get_active_tracer().event("analysis_error", node="researcher", task_id=task.task_id, error=str(exc))

        status = "done" if evidence else ("partial" if read_papers else "failed")
        summary = (
            f"[{task.perspective}] {truncate(task.question, 60)}：发现 {len(sr.papers)} 篇 / 精读 {len(read_papers)} 篇 / "
            f"证据 {len(evidence)} 条 / 停止原因 {sr.stopped_reason}"
        )
        finding = Finding(
            task_id=task.task_id, agent="researcher", summary=summary,
            paper_ids=[p.paper_id for p in read_papers or sr.papers],
            evidence_ids=[e.evidence_id for e in evidence],
            notes=list(sr.queries), tool_calls=len(sr.tool_logs), status=status,
        )
        return {
            "papers": sr.papers, "evidence": evidence, "cards": cards, "findings": [finding],
            "comparisons": comparisons, "conflicts": conflicts, "tool_logs": list(sr.tool_logs),
        }

    # ---- Gap Analyzer ----
    @node
    def gap_analyzer(state: ResearchState) -> dict:
        from .agents.gap_analyzer import analyze_gaps

        tasks = state.get("tasks") or []
        papers = state.get("papers") or []
        evidence = state.get("evidence") or []
        findings = state.get("findings") or []
        rr = state.get("research_round", 0)
        report = analyze_gaps(
            tasks, papers, evidence, findings, llm=get_llm("fast"),
            research_round=rr, max_rounds=int(budgets.get("max_research_rounds", 2)),
            max_new_tasks=int(budgets.get("max_tasks_per_round", 4)),
        )
        get_active_tracer().event("gap_report", node="gap_analyzer", sufficient=report.sufficient,
                                  gaps=len(report.gaps), new_tasks=len(report.new_tasks))
        update: dict[str, Any] = {
            "gaps": report.gaps, "gap_report": report.model_dump(),
            "research_round": rr + (0 if report.sufficient else 1),
        }
        if not report.sufficient and report.new_tasks:
            update["tasks"] = tasks + report.new_tasks  # 定向补充任务，回到 Supervisor
        return update

    def route_gap(state: ResearchState) -> str:
        report = state.get("gap_report") or {}
        if report.get("sufficient", True):
            return "write"
        # 上限兜底：预算耗尽或无新任务时不再循环，报告标注"证据不足"
        if not report.get("new_tasks") or get_global_budget().exceeded:
            return "write"
        return "research_more"

    # ---- Outline & Writer ----
    @node
    def outline(state: ResearchState) -> dict:
        from .agents.writer import generate_outline

        sections = generate_outline(
            state["brief"], state.get("tasks") or [], state.get("evidence") or [],
            state.get("papers") or [], gaps=state.get("gaps") or [], llm=get_llm("orchestrator"),
        )
        return {"outline": sections}

    @node
    def writer(state: ResearchState) -> dict:
        from .agents.writer import write_report

        repair_hints: list[str] = []
        if state.get("repair_round", 0) > 0 and state.get("verification") is not None:
            from .agents.verifier import build_repair_hints

            repair_hints = build_repair_hints(state["verification"], state.get("claims") or [])
        draft = write_report(
            state["brief"], state.get("outline") or [], state.get("evidence") or [],
            state.get("papers") or [], state.get("cards") or [],
            comparisons=state.get("comparisons") or [], conflicts=state.get("conflicts") or [],
            gaps=state.get("gaps") or [], llm=get_llm("writer"), repair_hints=repair_hints,
        )
        return {"draft": draft.markdown, "claims": draft.claims}

    # ---- Verifier ----
    @node
    def verifier(state: ResearchState) -> dict:
        from .agents.verifier import verify

        report = verify(
            state.get("claims") or [], state.get("evidence") or [], state.get("papers") or [],
            state.get("draft", ""), llm=get_llm("judge"),
            max_judge_calls=int(budgets.get("max_judge_calls_per_verification", 80)),
        )
        return {"verification": report}

    def route_verification(state: ResearchState) -> str:
        from .agents.verifier import route_verification as route

        return route(
            state.get("verification"),
            repair_round=state.get("repair_round", 0),
            max_repair_rounds=int(budgets.get("max_repair_rounds", 2)),
        )

    # ---- Targeted Research（修复轮）----
    @node
    def targeted_research(state: ResearchState) -> dict:
        from .agents.verifier import build_repair_tasks

        repair_tasks = build_repair_tasks(state["verification"], state.get("claims") or [],
                                          max_tasks=int(budgets.get("max_tasks_per_round", 4)))
        rr = state.get("repair_round", 0) + 1
        get_active_tracer().event("repair_dispatch", node="targeted_research", tasks=len(repair_tasks), repair_round=rr)
        return {"tasks": (state.get("tasks") or []) + repair_tasks, "repair_round": rr}

    # ---- Finalize ----
    @node
    def finalize(state: ResearchState) -> dict:
        from .agents.writer import render_final
        from .memory.evidence_store import EvidenceStore

        s = get_settings()
        run_id = state.get("run_id") or now_iso()
        claims: list[ClaimRecord] = state.get("claims") or []
        papers: list[PaperRecord] = state.get("papers") or []
        evidence = state.get("evidence") or []
        verification = state.get("verification")

        # 渲染最终报告（内部键 → 编号 + 参考文献）
        final_md = render_final(state.get("draft", ""), claims, papers)

        # 追加审计与缺口说明（文档要求：证据不足必须明确标注，不得静默）
        if verification is not None:
            counts: dict[str, int] = {}
            for c in claims:
                counts[c.status] = counts.get(c.status, 0) + 1
            final_md += (
                "\n\n---\n\n## 引用审计摘要\n\n"
                f"- 结论状态分布：{json.dumps(counts, ensure_ascii=False)}\n"
                f"- 覆盖率核验未带引用的事实句：{len(verification.coverage_missing)}\n"
                f"- 未解释冲突：{len(verification.conflicts)}\n"
                f"- 审计详情：`{run_id}.json`\n"
            )
        gaps = state.get("gaps") or []
        if gaps:
            final_md += "\n\n## 证据缺口（未解决，明确标注）\n\n" + "\n".join(
                f"- [{g.severity}] {g.description}" + (f"（{g.fix_question}）" if g.fix_question else "") for g in gaps
            )
        final_md += f"\n\n<!-- run_id: {run_id} 生成于 {now_iso()} -->\n"

        report_path = s.reports_dir / f"{run_id}.md"
        report_path.write_text(final_md, encoding="utf-8")

        # 持久化证据记忆
        try:
            store = EvidenceStore(s.db_path)
            store.upsert_papers(papers)
            store.upsert_evidence_many(evidence)
            for card in state.get("cards") or []:
                store.upsert_card(card)
            for claim in claims:
                store.upsert_claim(claim)
            for f in state.get("findings") or []:
                store.add_finding(f)
            store.close()
        except Exception as exc:  # noqa: BLE001
            get_active_tracer().event("persist_error", node="finalize", error=str(exc))

        # 向量知识库写入（Milvus Lite + bge-m3，best-effort：不可用/失败静默跳过，
        # 证据问答的既有路径不受影响）
        try:
            from .memory import vector_store

            titles = {p.paper_id: p.title or "" for p in papers}
            vector_store.upsert_evidence(
                [e.model_dump() for e in evidence], run_id=run_id, paper_titles=titles
            )
        except Exception as exc:  # noqa: BLE001
            get_active_tracer().event("persist_error", node="finalize", error=f"kb: {exc}")

        audit = {
            "run_id": run_id,
            "created_at": now_iso(),
            "user_query": state.get("user_query"),
            "brief": state["brief"].model_dump() if state.get("brief") else None,
            "research_round": state.get("research_round", 0),
            "repair_round": state.get("repair_round", 0),
            "tasks": [t.model_dump() for t in (state.get("tasks") or [])],
            "papers": [p.model_dump() for p in papers],
            "evidence": [e.model_dump() for e in evidence],
            "findings": [f.model_dump() for f in (state.get("findings") or [])],
            "claims": [c.model_dump() for c in claims],
            "verification": verification.model_dump() if verification else None,
            "conflicts": [c.model_dump() for c in (state.get("conflicts") or [])],
            "gaps": [g.model_dump() for g in gaps],
            "budget": get_global_budget().snapshot(),
            "errors": state.get("errors") or [],
            "report_path": str(report_path),
            "trace_path": str(s.traces_dir / f"{run_id}.jsonl"),
        }
        audit_path = s.audit_dir / f"{run_id}.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "report_md": final_md, "report_path": str(report_path),
            "audit_path": str(audit_path), "final_status": "done",
        }

    # ---- 组装 ----
    builder = StateGraph(ResearchState)
    builder.add_node("scope", scope)
    builder.add_node("planner", planner)
    builder.add_node("supervisor", lambda state: {})  # 派发逻辑全部在 dispatch 路由里
    builder.add_node("researcher", researcher)
    builder.add_node("gap_analyzer", gap_analyzer)
    builder.add_node("outline", outline)
    builder.add_node("writer", writer)
    builder.add_node("verifier", verifier)
    builder.add_node("targeted_research", targeted_research)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "scope")
    builder.add_edge("scope", "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges("supervisor", dispatch, ["researcher", "gap_analyzer"])
    builder.add_edge("researcher", "supervisor")  # 批次完成后回到 Supervisor（多批派发/排空）
    builder.add_conditional_edges("gap_analyzer", route_gap, {"research_more": "supervisor", "write": "outline"})
    builder.add_edge("outline", "writer")
    builder.add_edge("writer", "verifier")
    builder.add_conditional_edges("verifier", route_verification, {"repair": "targeted_research", "finalize": "finalize"})
    builder.add_edge("targeted_research", "supervisor")  # 修复任务走同一研究管线（最终仍经 gap→write）
    builder.add_edge("finalize", END)
    return builder.compile()
