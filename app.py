#!/usr/bin/env python
"""科研助手 Agent CLI。

用法：
    python app.py research "VLM 后训练方法（SFT/DPO/GRPO）的对比与争议" \
        [--papers-dir data/papers] [--fake] [--max-concurrency 4]
    python app.py audit outputs/audit/run_xxx.json
    python app.py eval outputs/audit/run_xxx.json --trace data/traces/run_xxx.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import reset_global_budget  # noqa: E402
from src.settings import enable_fake_mode, get_settings  # noqa: E402
from src.tracing import JsonlTracer, set_active_tracer  # noqa: E402
from src.utils import new_id, truncate  # noqa: E402


def _progress_line(node: str, state: dict) -> str:
    papers = len(state.get("papers") or [])
    evidence = len(state.get("evidence") or [])
    tasks = len(state.get("tasks") or [])
    findings = len(state.get("findings") or [])
    rr = state.get("research_round", 0)
    pr = state.get("repair_round", 0)
    if node == "researcher":
        cur = state.get("current_task") or {}
        q = truncate(cur.get("question", ""), 40)
        return f"  ├ 研究: {q}（{cur.get('perspective', '?')}）"
    if node == "writer":
        return f"  ├ 写作: draft {len(state.get('draft') or '')} 字符 / claims {len(state.get('claims') or [])} 条"
    if node == "verifier":
        v = state.get("verification")
        if v is not None:
            return f"  ├ 核验: passed={getattr(v, 'passed', '?')} missing={len(getattr(v, 'coverage_missing', []) or [])}"
    if node == "gap_analyzer":
        return f"  ├ 缺口分析: gaps={len(state.get('gaps') or [])}"
    return f"  ├ {node}: papers={papers} evidence={evidence} tasks={tasks}/{findings} rounds={rr}/{pr}"


def cmd_research(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.fake:
        enable_fake_mode()
    if args.no_download:
        settings.sources.setdefault("search", {})["download_pdfs"] = False

    run_id = new_id("run")
    tracer = JsonlTracer(run_id, settings.traces_dir / f"{run_id}.jsonl")
    set_active_tracer(tracer)
    budget = reset_global_budget()

    from src.graph import build_graph

    graph = build_graph(settings=settings)
    init_state = {"user_query": args.query, "papers_dir": args.papers_dir, "run_id": run_id}
    config = {
        "max_concurrency": args.max_concurrency or int(settings.budget("max_concurrent_researchers", 4)),
        "recursion_limit": 300,
    }

    print(f"▶ 开始研究：{args.query}")
    print(f"  run_id={run_id} fake={args.fake} papers_dir={args.papers_dir or '-'} 并发={config['max_concurrency']}")

    final_state: dict = {}
    try:
        for state in graph.stream(init_state, config=config, stream_mode="values"):
            final_state = state
            # 每个超步输出一行进度
            print(_progress_line("-", state), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ 图执行失败：{type(exc).__name__}: {exc}")
        tracer.event("graph_error", error=str(exc))
        tracer.close()
        return 1

    tracer.close()
    report_path = final_state.get("report_path")
    audit_path = final_state.get("audit_path")
    errors = final_state.get("errors") or []
    verification = final_state.get("verification")

    print("\n" + "=" * 60)
    print(f"✔ 完成  run_id={run_id}")
    print(f"  报告: {report_path or '(未生成)'}")
    print(f"  审计: {audit_path or '(未生成)'}")
    print(f"  Trace: {settings.traces_dir / (run_id + '.jsonl')}")
    print(f"  论文 {len(final_state.get('papers') or [])} 篇 / 证据 {len(final_state.get('evidence') or [])} 条 / "
          f"结论 {len(final_state.get('claims') or [])} 条")
    if verification is not None:
        counts: dict[str, int] = {}
        for c in final_state.get("claims") or []:
            counts[c.status] = counts.get(c.status, 0) + 1
        print(f"  结论状态: {json.dumps(counts, ensure_ascii=False)}")
        print(f"  引用核验: passed={getattr(verification, 'passed', None)}, "
              f"coverage_missing={len(verification.coverage_missing)}, conflicts={len(verification.conflicts)}")
    print(f"  预算: {json.dumps(budget.snapshot(), ensure_ascii=False)}")
    if errors:
        print(f"  ⚠ 错误 {len(errors)} 条（详见 trace）：{errors[:3]}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    claims = data.get("claims") or []
    counts: dict[str, int] = {}
    for c in claims:
        counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1
    v = data.get("verification") or {}
    print(f"run {data.get('run_id')}  query: {truncate(data.get('user_query') or '', 80)}")
    print(f"论文 {len(data.get('papers') or [])} / 证据 {len(data.get('evidence') or [])} / 结论 {len(claims)}")
    print(f"结论状态: {json.dumps(counts, ensure_ascii=False)}")
    print(f"核验: passed={v.get('passed')} coverage_missing={len(v.get('coverage_missing') or [])} "
          f"conflicts={len(v.get('conflicts') or [])}")
    for c in claims:
        if c.get("status") in ("unsupported", "metadata_error", "conflicted"):
            print(f"  [{c.get('status')}] {truncate(c.get('claim_text', ''), 100)}")
            if c.get("verifier_note"):
                print(f"      └ {truncate(c['verifier_note'], 120)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from src.eval.agent_eval import evaluate as eval_agent

    data = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    from src.eval.citation_eval import evaluate as eval_citation

    metrics = eval_citation(data, gold=json.loads(Path(args.gold).read_text(encoding="utf-8")) if args.gold else None)
    print("引用质量指标:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.trace:
        print("\nAgent 行为指标:")
        print(json.dumps(eval_agent(args.trace), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="科研助手 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_research = sub.add_parser("research", help="执行一次完整研究流程")
    p_research.add_argument("query", help="研究问题（自然语言）")
    p_research.add_argument("--papers-dir", default=None, help="本地 PDF 目录（可选，作为种子论文）")
    p_research.add_argument("--fake", action="store_true", help="离线 fake 模式（不发网络请求）")
    p_research.add_argument("--no-download", action="store_true", help="禁用 PDF 下载（只检索元数据）")
    p_research.add_argument("--max-concurrency", type=int, default=None, help="并行 Researcher 上限")
    p_research.set_defaults(func=cmd_research)

    p_audit = sub.add_parser("audit", help="打印审计包摘要")
    p_audit.add_argument("audit", help="outputs/audit/*.json 路径")
    p_audit.set_defaults(func=cmd_audit)

    p_eval = sub.add_parser("eval", help="从审计包/trace 计算指标")
    p_eval.add_argument("audit", help="审计 JSON 路径")
    p_eval.add_argument("--gold", default=None, help="金标 key_points JSON（可选）")
    p_eval.add_argument("--trace", default=None, help="trace JSONL 路径（可选，输出 agent 行为指标）")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
