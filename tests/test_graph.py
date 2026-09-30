"""全图 fake 模式冒烟：验证 Scope→…→Finalize 管线连通、循环有界、产物落盘。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.settings import enable_fake_mode


@pytest.fixture
def fake_env(tmp_path, two_synthetic_pdfs, offline_sources, monkeypatch):
    """fake LLM + 离线数据源 + 本地合成论文目录。"""
    from src.settings import get_settings

    enable_fake_mode()
    s = get_settings()
    monkeypatch.setattr(s, "reports_dir", tmp_path / "reports")
    monkeypatch.setattr(s, "audit_dir", tmp_path / "audit")
    monkeypatch.setattr(s, "traces_dir", tmp_path / "traces")
    monkeypatch.setattr(s, "db_path", tmp_path / "indexes" / "ev.sqlite3")
    for d in (s.reports_dir, s.audit_dir, s.traces_dir):
        d.mkdir(parents=True, exist_ok=True)
    from src.llm import reset_global_budget

    reset_global_budget()
    return s


def test_full_graph_fake_mode(fake_env, two_synthetic_pdfs):
    from src.graph import build_graph
    from src.tracing import JsonlTracer, set_active_tracer
    from src.utils import new_id

    run_id = new_id("run")
    tracer = JsonlTracer(run_id, fake_env.traces_dir / f"{run_id}.jsonl")
    set_active_tracer(tracer)

    graph = build_graph(settings=fake_env)
    final = {}
    for state in graph.stream(
        {"user_query": "VLM 后训练方法 GRPO 与 DPO 的对比", "papers_dir": str(two_synthetic_pdfs), "run_id": run_id},
        config={"max_concurrency": 3, "recursion_limit": 300},
        stream_mode="values",
    ):
        final = state
    tracer.close()

    # 主链路完整走完
    assert final.get("final_status") == "done"
    assert final.get("brief") is not None
    assert final.get("tasks"), "planner 必须产出任务（fake 模式走模板兜底）"
    assert final.get("findings"), "researcher 必须产出 Finding"

    # 循环有界（research_round ≤ 2 + 初始轮）
    assert final.get("research_round", 0) <= 3

    # 本地 PDF 进入论文池
    paper_ids = [p.paper_id for p in final.get("papers") or []]
    assert any(pid.startswith("sha256:") for pid in paper_ids), "本地 PDF 应作为种子论文进入"

    # 产物落盘
    report = Path(final["report_path"])
    audit = Path(final["audit_path"])
    assert report.exists() and report.stat().st_size > 0
    assert audit.exists()
    data = json.loads(audit.read_text(encoding="utf-8"))
    for key in ("run_id", "brief", "tasks", "papers", "evidence", "claims", "verification", "budget", "trace_path"):
        assert key in data, f"审计包缺字段 {key}"

    # trace 记录了节点事件
    trace_lines = Path(data["trace_path"]).read_text(encoding="utf-8").strip().splitlines()
    kinds = {json.loads(line)["kind"] for line in trace_lines}
    assert "node_start" in kinds and "node_end" in kinds
