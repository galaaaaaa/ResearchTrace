"""Agent 行为评测：解析 JSONL trace → 节点/工具/LLM/搜索行为指标。

用法：
    python -m src.eval.agent_eval data/traces/x.jsonl

指标：各 kind 事件计数、llm_call 总 token、每节点平均耗时、tool_error 率、
相邻搜索查询相似度 > 0.85 的比例（repeated_search_rate）、是否正常 finalize（stop_correctness）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:  # rapidfuzz 是声明依赖
    from rapidfuzz import fuzz

    def _query_sim(a: str, b: str) -> float:
        return fuzz.ratio(a.lower(), b.lower()) / 100.0

except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _query_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()


_SIMILAR_THRESHOLD = 0.85  # budgets.yaml: similar_query_stop


def _load_events(trace_path: str | Path) -> list[dict]:
    events: list[dict] = []
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
            if isinstance(ev, dict):
                events.append(ev)
        except (json.JSONDecodeError, ValueError):
            continue
    return events


def evaluate(trace_path: str | Path) -> dict:
    """解析 JSONL trace，输出 Agent 行为指标（解析失败行跳过，绝不抛异常）。"""
    events = _load_events(trace_path)
    kind_counts: Counter[str] = Counter(str(e.get("kind") or "?") for e in events)

    llm_events = [e for e in events if e.get("kind") == "llm_call"]
    input_tokens = sum(int(e.get("input_tokens") or 0) for e in llm_events)
    output_tokens = sum(int(e.get("output_tokens") or 0) for e in llm_events)

    node_durations: dict[str, list[float]] = defaultdict(list)
    for e in events:
        node, dur = e.get("node"), e.get("duration_ms")
        if node and dur is not None:
            try:
                node_durations[str(node)].append(float(dur))
            except (TypeError, ValueError):
                pass
    node_avg = {k: round(sum(v) / len(v), 1) for k, v in sorted(node_durations.items())}

    tool_events = [e for e in events if str(e.get("kind") or "").startswith("tool")]
    tool_errors = sum(1 for e in tool_events if e.get("ok") is False or e.get("error"))
    tool_error_rate = round(tool_errors / len(tool_events), 4) if tool_events else 0.0

    queries: list[str] = []
    for e in tool_events:
        name = str(e.get("tool") or e.get("name") or "")
        if "search" not in name.lower():
            continue
        args = e.get("args") if isinstance(e.get("args"), dict) else {}
        q = args.get("query") or e.get("query")
        if q:
            queries.append(str(q))
    n_queries = len(queries)
    similar_pairs = sum(1 for a, b in zip(queries, queries[1:]) if _query_sim(a, b) > _SIMILAR_THRESHOLD)
    repeated_search_rate = round(similar_pairs / (n_queries - 1), 4) if n_queries > 1 else 0.0

    budget_exhausted = any(e.get("kind") == "budget_exceeded" for e in events) or any(
        "预算已耗尽" in str(e.get("error") or "") for e in events
    )
    finalized = any(e.get("kind") in ("run_end", "finalize") for e in events)
    stop_correctness = finalized and not budget_exhausted

    result: dict[str, Any] = {
        "n_events": len(events),
        "kind_counts": dict(kind_counts),
        "llm_calls": len(llm_events),
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "node_avg_duration_ms": node_avg,
        "tool_calls": len(tool_events),
        "tool_errors": tool_errors,
        "tool_error_rate": tool_error_rate,
        "n_search_queries": n_queries,
        "repeated_search_rate": repeated_search_rate,
        "budget_exhausted": budget_exhausted,
        "finalized": finalized,
        "stop_correctness": stop_correctness,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 行为评测（JSONL trace → 指标）")
    parser.add_argument("trace", help="trace JSONL 路径")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args.trace), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
