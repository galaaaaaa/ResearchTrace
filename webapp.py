#!/usr/bin/env python
"""科研助手 Agent — 轻量 Web 界面。

用法：
    python webapp.py [--host 127.0.0.1] [--port 8300]

行为：
- POST /api/research 在后台线程执行完整研究图（同一时刻只允许一个运行中的任务，
  因为 TokenBudget 与 SQLite 证据库是进程级全局资源）；
- 进度快照取自 LangGraph values 流 + trace JSONL 尾部（节点级事件）；
- 历史 run 从 outputs/audit 扫描，报告/审计直接读盘；
- 默认只绑 127.0.0.1；--host 0.0.0.0 可局域网访问（无鉴权，请勿暴露公网）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.settings import enable_fake_mode, get_settings  # noqa: E402
from src.tracing import JsonlTracer, set_active_tracer  # noqa: E402
from src.utils import new_id, truncate  # noqa: E402

app = FastAPI(title="科研助手 Agent", docs_url=None, redoc_url=None, openapi_url=None)

_RUNS_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}  # run_id → 运行快照（仅本进程内活跃 run）
_LOG_KEEP = 80


# --------------------------------------------------------------------------
# 后台执行
# --------------------------------------------------------------------------
def _phase_of(state: dict) -> str:
    """从图状态推断当前阶段（values 流不含节点名，用状态特征近似）。"""
    if state.get("final_status"):
        return "完成"
    if state.get("verification") is not None:
        return "引用核验"
    if state.get("draft"):
        return "撰写报告"
    if state.get("outline"):
        return "生成提纲"
    if state.get("gaps"):
        return "缺口分析"
    if state.get("findings"):
        return f"研究执行（第 {state.get('research_round', 0) + 1} 轮）"
    if state.get("tasks"):
        return "任务规划完成"
    return "范围界定"


def _snapshot_from_state(state: dict) -> dict:
    return {
        "phase": _phase_of(state),
        "papers": len(state.get("papers") or []),
        "evidence": len(state.get("evidence") or []),
        "tasks": len(state.get("tasks") or []),
        "findings": len(state.get("findings") or []),
        "research_round": state.get("research_round", 0),
        "repair_round": state.get("repair_round", 0),
        "current_task": truncate(((state.get("current_task") or {}).get("question") or ""), 60),
    }


def _run_research(run_id: str, req: "ResearchRequest") -> None:
    """后台线程：完整执行一次研究图，快照写入 _RUNS[run_id]。"""
    snap = _RUNS[run_id]
    logs: deque[str] = snap["log"]
    settings = get_settings()
    if req.fake:
        enable_fake_mode()
    if req.no_download:
        settings.sources.setdefault("search", {})["download_pdfs"] = False

    tracer = JsonlTracer(run_id, settings.traces_dir / f"{run_id}.jsonl")
    set_active_tracer(tracer)
    from src.llm import reset_global_budget

    budget = reset_global_budget()

    from src.graph import build_graph

    graph = build_graph(settings=settings)
    init_state = {"user_query": req.question, "papers_dir": req.papers_dir or None, "run_id": run_id}
    config = {
        "max_concurrency": req.max_concurrency or int(settings.budget("max_concurrent_researchers", 4)),
        "recursion_limit": 300,
    }
    logs.append(f"▶ 开始研究：{req.question}")
    logs.append(f"  fake={req.fake} papers_dir={req.papers_dir or '-'} 并发={config['max_concurrency']}")
    try:
        final_state: dict = {}
        for state in graph.stream(init_state, config=config, stream_mode="values"):
            final_state = state
            with _RUNS_LOCK:
                snap.update(_snapshot_from_state(state))
        with _RUNS_LOCK:
            snap["status"] = "done"
            snap["report_path"] = str(final_state.get("report_path") or "")
            snap["audit_path"] = str(final_state.get("audit_path") or "")
            counts: dict[str, int] = {}
            for c in final_state.get("claims") or []:
                counts[c.status] = counts.get(c.status, 0) + 1
            snap["claim_status"] = counts
            snap["budget"] = budget.snapshot()
            logs.append(
                f"✔ 完成：论文 {snap['papers']} / 证据 {snap['evidence']} / 结论 {len(final_state.get('claims') or [])}"
            )
    except Exception as exc:  # noqa: BLE001
        with _RUNS_LOCK:
            snap["status"] = "error"
            snap["error"] = f"{type(exc).__name__}: {exc}"
        try:
            tracer.event("graph_error", error=str(exc))
        except Exception:  # noqa: BLE001
            pass
        logs.append(f"✗ 执行失败：{snap['error']}")
    finally:
        tracer.close()
        with _RUNS_LOCK:
            snap["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 请求/响应模型
# --------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    question: str
    papers_dir: str | None = "data/papers"
    no_download: bool = False
    fake: bool = False
    max_concurrency: int | None = None


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"

    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    run_id: str | None = None  # 提供则基于该 run 的证据问答（RAG），否则自由对话
    web_search: bool = False  # 自由对话时联网搜索（WEB_SEARCH_PROVIDER 选服务商）；run_id 优先
    use_kb: bool = False  # 自由对话时查向量知识库（Milvus+bge-m3 跨 run 全库）；优先级 run_id > 联网 > 知识库
    top_k: int = 8


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.post("/api/research")
def start_research(req: ResearchRequest) -> dict:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "研究问题不能为空")
    with _RUNS_LOCK:
        running = [r for r in _RUNS.values() if r.get("status") == "running"]
        if running:
            raise HTTPException(409, f"已有运行中的任务 {running[0]['run_id']}（预算与证据库为全局资源，请等待完成）")
        run_id = new_id("run")
        _RUNS[run_id] = {
            "run_id": run_id,
            "question": question,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "启动中",
            "papers": 0,
            "evidence": 0,
            "tasks": 0,
            "findings": 0,
            "research_round": 0,
            "repair_round": 0,
            "current_task": "",
            "log": deque(maxlen=_LOG_KEEP),
            "fake": req.fake,
        }
    threading.Thread(target=_run_research, args=(run_id, req), daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/runs")
def list_runs() -> dict:
    settings = get_settings()
    items: list[dict] = []
    seen: set[str] = set()
    with _RUNS_LOCK:
        for rid, snap in _RUNS.items():
            seen.add(rid)
            items.append(
                {
                    "run_id": rid,
                    "question": snap["question"],
                    "status": snap["status"],
                    "created_at": snap["started_at"],
                    "fake": snap.get("fake", False),
                }
            )
    # 历史 run：从审计包扫描（服务器重启后仍可见）
    for path in sorted(settings.audit_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        rid = path.stem
        if rid in seen:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        items.append(
            {
                "run_id": rid,
                "question": truncate(data.get("user_query") or "", 80),
                "status": "done",
                "created_at": data.get("created_at") or time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)
                ),
                "fake": False,
            }
        )
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return {"runs": items}


@app.get("/api/runs/{run_id}")
def run_status(run_id: str) -> dict:
    with _RUNS_LOCK:
        snap = _RUNS.get(run_id)
        if snap is not None:
            return {**{k: v for k, v in snap.items() if k != "log"}, "log": list(snap["log"])}
    # 非本进程 run：从审计包构造摘要
    audit = _load_audit(run_id)
    if audit is None:
        raise HTTPException(404, f"未找到 run {run_id}")
    claims = audit.get("claims") or []
    counts: dict[str, int] = {}
    for c in claims:
        counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1
    v = audit.get("verification") or {}
    return {
        "run_id": run_id,
        "question": audit.get("user_query"),
        "status": "done",
        "created_at": audit.get("created_at"),
        "phase": "完成",
        "papers": len(audit.get("papers") or []),
        "evidence": len(audit.get("evidence") or []),
        "tasks": len(audit.get("tasks") or []),
        "findings": len(audit.get("findings") or []),
        "research_round": audit.get("research_round", 0),
        "repair_round": audit.get("repair_round", 0),
        "claim_status": counts,
        "verification": {
            "passed": v.get("passed"),
            "coverage_missing": len(v.get("coverage_missing") or []),
            "conflicts": len(v.get("conflicts") or []),
        },
        "budget": audit.get("budget"),
        "report_path": audit.get("report_path"),
        "log": [],
    }


@app.get("/api/runs/{run_id}/report")
def run_report(run_id: str) -> dict:
    settings = get_settings()
    path = settings.reports_dir / f"{run_id}.md"
    if not path.exists():
        raise HTTPException(404, "报告尚未生成")
    return {"markdown": path.read_text(encoding="utf-8")}


@app.get("/api/runs/{run_id}/audit")
def run_audit(run_id: str) -> dict:
    audit = _load_audit(run_id)
    if audit is None:
        raise HTTPException(404, f"未找到 run {run_id} 的审计包")
    claims = audit.get("claims") or []
    v = audit.get("verification") or {}
    flagged = [
        {"status": c.get("status"), "claim": truncate(c.get("claim_text", ""), 120), "note": truncate(c.get("verifier_note") or "", 160)}
        for c in claims
        if c.get("status") in ("unsupported", "metadata_error", "conflicted")
    ][:40]
    return {
        "run_id": run_id,
        "papers": len(audit.get("papers") or []),
        "evidence": len(audit.get("evidence") or []),
        "claims": len(claims),
        "claim_status": _claim_counts(claims),
        "verification": {
            "passed": v.get("passed"),
            "coverage_missing": len(v.get("coverage_missing") or []),
            "conflicts": len(v.get("conflicts") or []),
        },
        "gaps": [
            {"severity": g.get("severity"), "description": truncate(g.get("description") or "", 140)}
            for g in (audit.get("gaps") or [])
        ][:30],
        "flagged": flagged,
        "budget": audit.get("budget"),
    }


@app.get("/api/runs/{run_id}/trace")
def run_trace(run_id: str, tail: int = 40) -> dict:
    """trace JSONL 尾部事件（节点级进度）。"""
    settings = get_settings()
    path = settings.traces_dir / f"{run_id}.jsonl"
    if not path.exists():
        return {"events": []}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"events": []}
    events = []
    for line in lines[-max(1, min(tail, 200)) :]:
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        kind = d.get("kind") or ""
        label = d.get("label") or d.get("node") or ""
        if kind in ("node_start", "node_end", "llm_call", "tool_call", "error", "tool_error"):
            events.append(
                {
                    "ts": (d.get("ts") or "")[11:19],
                    "kind": kind,
                    "who": label,
                    "detail": truncate(
                        str({k: v for k, v in d.items() if k in ("task_id", "tool", "papers", "n_evidence", "error")}),
                        90,
                    ),
                }
            )
    return {"events": events}


# --------------------------------------------------------------------------
# 对话 / 证据问答
# --------------------------------------------------------------------------
_CHAT_FREE_SYSTEM = (
    "你是“科研助手 Agent”的对话模式：可以普通聊天、解释概念、帮用户梳理研究问题与思路。"
    "当用户询问具体论文的数据/结论等细节时，提醒对方可在左侧选择对应的历史 run 并把本对话框"
    "顶部的语境切换为“证据问答”，即可获得带论文页码引用、可回溯原文的回答。用中文回答。"
)

_CHAT_GROUNDED_SYSTEM = "你是严谨的证据问答助手：只依据用户消息末尾给出的证据片段作答。"

# 实测（deepseek-flash / v4-pro）得出的关键布局：证据片段放在最后一条用户消息末尾（近因），
# 并要求结构化输出“- 结论 [n]”（结论须贴近原文，可逐字摘录英文原句）——长 system 前缀会让模型
# 过度保守整体拒答，无约束短提示则诱发编造引文，此布局两者都避免。
_CHAT_GROUNDED_REQ = (
    "\n\n（要求：逐条读完下列全部片段后作答。回答格式：每个要点一行“- 结论 [n]”，"
    "其中结论必须贴近片段原文（可摘录英文原句或忠实翻译，禁止编造引文），[n] 为来源片段编号；"
    "不同片段结论冲突时并列呈现；所有片段都无关时才回复“当前证据集中没有足够信息”；"
    "禁止使用片段之外的知识填充。）"
)

# 联网模式提示词：与证据问答同一布局（材料放用户消息末尾 + 结构化 [n] 输出），
# 但允许“（补充）”行列出搜索之外的知识——搜索是辅助信源，不整体禁用模型常识。
_CHAT_WEB_SYSTEM = "你是严谨的联网问答助手：优先依据用户消息末尾的联网搜索结果作答，并按要求标注来源编号。"

_CHAT_WEB_REQ = (
    "\n\n（要求：逐条读完下列联网搜索结果后作答。来自搜索结果的要点每行“- 结论 [n]”，"
    "结论须贴近该条结果原文（禁止编造引文），[n] 为结果编号，同一结论可标多个编号；"
    "需要补充结果之外的背景知识时另起“- （补充）…”行且不加编号；不同结果冲突时并列呈现；"
    "今天是 {today}，回答注意信息时效（各条发布日期已标注）；"
    "所有结果都与问题无关时，明确说明搜索结果无法回答该问题，再视需要用补充行作答。）"
)

# Reader 失败桩 / VLM 提示前缀（与 writer._STUB_CLAIM_RE 同源）：不是真实要点，不展示
_STUB_HINT_PREFIXES = ("（LLM 失败", "LLM 摘要失败", "[VLM 解读", "VLM 解读", "图表证据提示（", "表格证据提示（", "（本片段为")


def _norm_token(t: str) -> str:
    """轻量英文词干归一（无词干时 temperature≠temperatures 会整段漏检）。"""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def _rerank_with_dense(fill: list[tuple[float, dict]], queries: list[str], k: int) -> list[tuple[float, dict]]:
    """向量重排（可选）：embedding 可用时对候选短名单做 RRF 融合，词法序保底。

    与 pdf_tool._fuse_with_dense 同款 RRF(k=60)：语义近邻（同义改写/跨语言）补词法失配；
    向量侧不可用/失败原样返回，问答行为不变。
    """
    from src.tools import embedding as emb

    if not fill or not emb.embedding_available():
        return fill
    q_list = [q for q in (queries or []) if q and q.strip()]
    if not q_list:
        return fill
    shortlist = fill[: max(3 * k, k)]
    texts = [
        f"{ev.get('claim_hint') or ''} {ev.get('evidence_text') or ''}"[:2000]
        for _s, ev in shortlist
    ]
    vecs = emb.embed_texts(q_list + texts)
    if not vecs:
        return fill
    q_vecs, t_vecs = vecs[: len(q_list)], vecs[len(q_list) :]
    if not any(q_vecs):
        return fill
    dense_order = sorted(
        range(len(shortlist)), key=lambda i: -max(emb.cosine(qv, t_vecs[i]) for qv in q_vecs)
    )
    k_rrf = 60
    scores: dict[int, float] = {}
    for ranks in (list(range(len(shortlist))), dense_order):
        for r, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + r + 1)
    reordered = [shortlist[i] for i in sorted(scores, key=lambda i: -scores[i])]
    return reordered + fill[len(shortlist):]


def _rank_evidence(
    evs: list[dict],
    queries: list[str],
    k: int,
    paper_titles: dict[str, str] | None = None,
) -> list[tuple[float, dict]]:
    """Okapi BM25 检索（复用 pdf_tool 中英混合分词 + 英文词干归一）。

    - 先按 (paper_id, page, 文本前缀) 去重（多轮研究会重复收集同一段落）；
    - 多查询取每个文档上的最高分（中文原问 + 英文扩展查询，缓解中英跨语言失配）；
    - 论文标题命中查询词的文档 ×1.3（问 DPO 时优先 DPO 论文而非顺带提 DPO 的他文）；
    - text/table 证据优先，figure（VLM 二手中文解读，与中文问题字面虚高重合）只补位。
    """
    from src.tools.pdf_tool import tokenize

    def _tok(text: str) -> list[str]:
        return [_norm_token(t) for t in tokenize(text)]

    seen: set[tuple] = set()
    uniq: list[dict] = []
    for ev in evs or []:
        key = (ev.get("paper_id"), ev.get("page"), (ev.get("evidence_text") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ev)
    q_tokens_list = [_tok(q) for q in queries if q and q.strip()]
    if not q_tokens_list or not uniq:
        return []
    all_q = {t for toks in q_tokens_list for t in toks if len(t) >= 2}
    title_tokens: dict[str, set[str]] = {}
    for pid, title in (paper_titles or {}).items():
        title_tokens[pid] = {t for t in _tok(title or "") if len(t) >= 2}
    docs = [_tok(f"{ev.get('claim_hint') or ''} {ev.get('evidence_text') or ''}") for ev in uniq]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df: dict[str, int] = {}
    for toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    k1, b = 1.5, 0.75
    scored: list[tuple[float, dict]] = []
    for ev, toks in zip(uniq, docs):
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        norm = k1 * (1 - b + b * len(toks) / avgdl)
        best = 0.0
        for q_tokens in q_tokens_list:
            s = 0.0
            for qt in q_tokens:
                f = tf.get(qt, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5))
                s += idf * f * (k1 + 1) / (f + norm)
            best = max(best, s)
        if all_q & title_tokens.get(ev.get("paper_id") or "", set()):
            best *= 1.3
        scored.append((best, ev))
    scored.sort(key=lambda x: x[0], reverse=True)
    text_hits = [(s, ev) for s, ev in scored if s > 0 and ev.get("modality") not in ("figure",)]
    figure_hits = [(s * 0.6, ev) for s, ev in scored if s > 0 and ev.get("modality") == "figure"]
    fill = (text_hits + figure_hits) if len(text_hits) < k else text_hits
    fill.sort(key=lambda x: x[0], reverse=True)
    return _rerank_with_dense(fill, queries, k)[:k]


def _highlight_terms(text: str, terms: list[str]) -> tuple[str, list[str]]:
    """把查询词直接标进片段文本（【】包裹），并返回命中的词——降低弱模型的定位负担。"""
    hits: list[str] = []
    for t in sorted({t for t in terms if len(t) >= 3}, key=len, reverse=True):
        pattern = re.compile(re.escape(t), re.IGNORECASE)
        if pattern.search(text):
            if t not in hits:
                hits.append(t)
            text = pattern.sub(lambda m: f"【{m.group(0)}】", text)
    return text, hits


def _expand_query(llm, question: str) -> str:
    """把中文问题译成英文检索关键词（证据原文多为英文）；失败返回空串，退化为单查询。"""
    try:
        out = llm.chat(
            prompt=f"把下面的问题翻译成适合学术论文全文检索的英文查询。只输出空格分隔的英文关键词"
            f"（不超过 15 个词，含关键数字与术语，不要解释、不要引号）：\n{truncate(question, 300)}",
            label="chat:query_expand",
        )
        return " ".join(out.strip().split())[:300]
    except Exception:  # noqa: BLE001
        return ""


def _chat_llm():
    """对话客户端：默认与主模型一致；设 CHAT_MODEL 可单独换更强模型。

    证据问答对模型阅读能力敏感——实测 deepseek-flash 倾向整体拒答或凭记忆作答，
    deepseek-v4-pro 才能逐条读片段并标注 [n]；故留此开关（.env 里配 CHAT_MODEL=…）。
    """
    from src.llm import get_llm

    llm = get_llm("chat")  # models.yaml 未定义 chat 角色时自动回落 _default
    override = os.environ.get("CHAT_MODEL", "").strip()
    if override:
        llm.model = override
    return llm


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """对话：run_id 为空 → 自由对话；指定 run → 基于其证据库的 RAG 问答（回答带 [n] 引用）。"""
    msgs = [m.model_dump() for m in req.messages if (m.content or "").strip()][-12:]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "最后一条消息必须是用户提问")

    with _RUNS_LOCK:
        if any(r.get("status") == "running" for r in _RUNS.values()):
            raise HTTPException(409, "研究任务运行中（共享 LLM 预算），请等它完成后再对话")

    from src.llm import reset_global_budget

    try:
        llm = _chat_llm()
        if llm.budget.exceeded:
            reset_global_budget()  # 上一个研究 run 用完预算不应阻塞交互对话
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"模型初始化失败：{exc}")

    citations: list[dict] = []
    grounded = req.run_id is not None
    web_mode = bool(req.web_search) and not grounded  # 证据问答（run_id）优先于联网
    kb_mode = bool(req.use_kb) and not grounded and not web_mode  # 知识库再次于联网
    system = _CHAT_FREE_SYSTEM
    if grounded:
        question = msgs[-1]["content"]
        audit = _load_audit(req.run_id)
        if audit is None:
            raise HTTPException(404, f"未找到 run {req.run_id}")
        evs = audit.get("evidence") or []
        if not evs:
            return {"reply": "该研究 run 没有收集到证据，无法进行证据问答。", "citations": [], "grounded": True}
        queries = [question] + [q for q in [_expand_query(llm, question)] if q]
        papers = {p.get("paper_id"): p for p in audit.get("papers") or []}
        picked = _rank_evidence(evs, queries, max(4, min(req.top_k, 12)),
                                paper_titles={pid: p.get("title") or "" for pid, p in papers.items()})
        if not picked:
            picked = [(0.0, evs[0])]
        # 查询里的 ASCII 词（原问 + 英文扩展）用于片段内高亮
        hl_terms = [w for q in queries for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", q)]
        lines: list[str] = []
        for i, (score, ev) in enumerate(picked, 1):
            p = papers.get(ev.get("paper_id")) or {}
            title = truncate(p.get("title") or ev.get("paper_id") or "未知论文", 90)
            page = ev.get("page")
            loc = f"第 {page} 页" if page else "页码未知"
            if ev.get("modality") == "figure":
                loc += "（图表证据：模型对图的解读，须对照原文）"
            hint = (ev.get("claim_hint") or "").strip()
            head = f"（要点：{truncate(hint, 100)}）" if hint and not hint.startswith(_STUB_HINT_PREFIXES) else ""
            text = truncate((ev.get("evidence_text") or "").strip(), 600)
            text, hits = _highlight_terms(text, hl_terms)
            tag = f"（命中词：{', '.join(hits[:6])}）" if hits else "（无命中词）"
            lines.append(f"[{i}] 《{title}》{loc}{tag}{head}：{text}")
            citations.append(
                {
                    "n": i,
                    "title": title,
                    "page": page,
                    "paper_id": ev.get("paper_id"),
                    "snippet": truncate(text, 200),
                }
            )
        # 证据片段放最后一条用户消息末尾（近因效应），配合结构化输出要求
        weak = picked[0][0] < 3.0  # 最高相关度也偏低 → 提醒模型大概率答不了
        msgs[-1] = {
            **msgs[-1],
            "content": (
                question
                + _CHAT_GROUNDED_REQ
                + ("\n\n（注意：以下证据与问题的整体相关度较低，若无直接对应内容，直接回答证据不足。）"
                   if weak else "")
                + f"\n\n（原研究问题：{truncate(audit.get('user_query') or '', 140)}）"
                + "\n\n证据片段：\n"
                + "\n\n".join(lines)
            ),
        }
        system = _CHAT_GROUNDED_SYSTEM
    elif web_mode:
        # 自由对话 + 联网开关：先检索再作答（密钥缺失 400 引导配置；搜索失败 502）
        question = msgs[-1]["content"]
        from src.tools.web_search import WebSearchError, WebSearchNotConfigured, search_web

        try:
            hits = search_web(question, count=8)
        except WebSearchNotConfigured:
            raise HTTPException(
                400,
                "联网搜索未配置：请在项目根目录 .env 任选一家服务商并重启 webapp——"
                "AnySearch：WEB_SEARCH_PROVIDER=anysearch + ANYSEARCH_API_KEY=<anysearch.com 控制台密钥>；"
                "智谱：ZHIPUAI_API_KEY=<智谱密钥>（默认）；"
                "阿里云 IQS：WEB_SEARCH_PROVIDER=iqs + IQS_API_KEY=<阿里云密钥>"
                "（可选 WEB_SEARCH_ENGINE=search_std|search_pro|search_pro_sogou|search_pro_quark）",
            )
        except WebSearchError as exc:
            raise HTTPException(502, f"联网搜索失败：{exc}")
        if hits:
            lines: list[str] = []
            for i, h in enumerate(hits, 1):
                meta = "，".join(x for x in (h["media"], h["publish_date"]) if x)
                lines.append(
                    f"[{i}] 《{h['title']}》{f'（{meta}）' if meta else ''}：{truncate(h['content'], 400)}"
                )
                citations.append(
                    {
                        "n": i,
                        "title": truncate(h["title"], 90),
                        "url": h["url"],
                        "media": h["media"] or None,
                        "date": h["publish_date"],
                        "snippet": truncate(h["content"], 200),
                    }
                )
            # 搜索结果同样放用户消息末尾（与证据问答同布局，{today} 提示时效）
            msgs[-1] = {
                **msgs[-1],
                "content": (
                    question
                    + _CHAT_WEB_REQ.format(today=time.strftime("%Y-%m-%d"))
                    + "\n\n联网搜索结果：\n"
                    + "\n\n".join(lines)
                ),
            }
            system = _CHAT_WEB_SYSTEM
        else:
            # 搜索正常但无结果：降级为普通对话，并明确告知（不冒充联网作答）
            msgs[-1] = {
                **msgs[-1],
                "content": question + "\n\n（注意：刚才的联网搜索没有返回任何结果，本条按普通对话作答，不要使用编号引用。）",
            }
    elif kb_mode:
        # 知识库模式：跨 run 全库向量检索（Milvus + bge-m3），片段布局与防线同证据问答
        question = msgs[-1]["content"]
        from src.memory import vector_store

        if not vector_store.kb_available():
            return {
                "reply": "📚 向量知识库不可用：需要 pymilvus（`uv pip install 'pymilvus[milvus_lite]'`）"
                         "与 .env 的 EMBEDDING_* 配置（bge-m3 网关），并确认已运行 scripts/backfill_kb.py。"
                         "当前问题可关闭 📚 后直接提问。",
                "citations": [], "kb": True,
            }
        hits = vector_store.search_kb(question, k=max(4, min(req.top_k, 12)))
        if not hits:
            return {
                "reply": "知识库中没有检索到与该问题相关的证据（库内是历史研究 run 的积累）。"
                         "可换更贴近已研究主题的问法，或关闭 📚 走普通对话 / 🌐 联网搜索。",
                "citations": [], "kb": True,
            }
        lines: list[str] = []
        for i, h in enumerate(hits, 1):
            title = truncate(h.get("title") or h.get("paper_id") or "未知论文", 90)
            page = h.get("page_out")
            loc = f"第 {page} 页" if page else "页码未知"
            if h.get("modality") in ("figure", "table"):
                loc += "（图表证据：模型对图的解读，须对照原文）"
            hint = (h.get("claim_hint") or "").strip()
            head = f"（要点：{truncate(hint, 100)}）" if hint and not hint.startswith(_STUB_HINT_PREFIXES) else ""
            lines.append(f"[{i}] 《{title}》{loc}{head}：{truncate((h.get('evidence_text') or '').strip(), 600)}")
            citations.append(
                {
                    "n": i, "title": title, "page": page, "paper_id": h.get("paper_id"),
                    "run_id": h.get("run_id"), "snippet": truncate(h.get("evidence_text") or "", 200),
                }
            )
        msgs[-1] = {
            **msgs[-1],
            "content": (
                question
                + _CHAT_GROUNDED_REQ
                + "\n\n（以下片段来自历史研究积累的知识库，可能出自不同 run 的不同论文。）"
                + "\n\n证据片段：\n"
                + "\n\n".join(lines)
            ),
        }
        system = _CHAT_GROUNDED_SYSTEM

    try:
        reply = llm.chat(messages=msgs, system=system, label="chat")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"模型调用失败：{exc}")
    if not (reply or "").strip():
        raise HTTPException(502, "模型返回空回复")
    reply = reply.strip()
    if grounded or kb_mode:
        # 去掉混进引用列表的“没有足够信息 [n]”行（拒答与引用不得同行出现）
        cleaned = "\n".join(
            l for l in reply.split("\n") if not re.match(r"^\s*[-*]?\s*当前证据集中没有足够信息\s*\[\d+\]", l)
        ).strip()
        if cleaned:
            reply = cleaned

    # 证据/联网问答的程序化防线（与 Verifier 同一哲学：无引用 = 不可信）：
    # 1) 回答里一个 [n] 都没有 → 带拒绝理由重写一次（拒绝语同样加在用户消息末尾）；
    # 2) 证据模式重写仍无 → 不作为答案展示，按“证据不足”处理，原稿折叠进 unverified；
    #    联网模式重写仍无 → 保留回答但加“未标注来源”警示横幅（搜索是辅助信源，不整体否决）。
    if (grounded or web_mode or kb_mode) and citations and not re.search(r"\[\d+\]", reply):
        if grounded or kb_mode:
            rejection = (
                "（【系统拒绝】你上一稿没有标注任何 [n] 证据引用。请重写：只使用上方证据片段中的信息，"
                "按“- 结论 [n]”格式；证据里没有的数字与结论一律删除；若确实无法回答，"
                "只回复“当前证据集中没有足够信息”。）"
            )
        else:
            rejection = (
                "（【系统拒绝】你上一稿没有标注任何 [n] 来源编号。请重写：凡来自上方搜索结果的"
                "事实性要点必须按“- 结论 [n]”标注编号；搜索之外的知识只能放“- （补充）…”行；"
                "若搜索结果确实都与问题无关，先明确说明再作答。）"
            )
        retry_msgs = [
            *msgs[:-1],
            {**msgs[-1], "content": msgs[-1]["content"] + "\n\n" + rejection},
        ]
        try:
            retried = llm.chat(messages=retry_msgs, system=system, label="chat:retry")
            if retried and retried.strip():
                reply = retried.strip()
        except Exception:  # noqa: BLE001 —— 重试失败沿用原回答
            pass
    unverified: str | None = None
    if (grounded or kb_mode) and not re.search(r"\[\d+\]", reply) and "没有足够信息" not in reply:
        unverified = reply
        source_desc = "本 run 的" if grounded else "知识库召回的"
        reply = (
            "⚠ **按“证据不足”处理**：模型连续两次未在回答中标注证据引用（其内容可能来自模型自身知识而非"
            f"{source_desc} {len(citations)} 条召回证据），不予采信、已折叠在下方“未验证回答”中。\n\n"
            "**与该问题最接近的证据片段**（编号即页码来源，可对照 PDF 原文）：\n"
            + "\n".join(
                f"- [{c['n']}] 《{c['title']}》"
                + (f"第 {c['page']} 页" if c.get("page") else "页码未知")
                + f"：{c['snippet'][:80]}"
                for c in citations
            )
            + ("\n\n建议：换一个更贴近该 run 研究范围的问法，或直接查阅上述片段对应原文。" if grounded
               else "\n\n建议：换一个更贴近已研究主题的问法，或查阅上述片段对应的论文原文。")
        )
    elif web_mode and citations and not re.search(r"\[\d+\]", reply):
        reply = (
            "⚠ **注意**：以下回答未标注任何搜索来源编号（[n]），内容可能主要来自模型自身知识而非本次"
            "搜索结果，请自行核实关键事实。\n\n---\n\n" + reply
        )
    return {
        "reply": reply,
        "citations": citations,
        "grounded": grounded,
        "web_search": web_mode,
        "kb": kb_mode,
        "unverified": unverified,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.exception_handler(404)
async def not_found_handler(request, exc):  # noqa: ANN001
    return JSONResponse({"detail": getattr(exc, "detail", "Not Found")}, status_code=404)


def _claim_counts(claims: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in claims:
        counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1
    return counts


def _load_audit(run_id: str) -> dict | None:
    settings = get_settings()
    path = settings.audit_dir / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="科研助手 Agent Web 界面")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址（0.0.0.0 可局域网访问）")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()

    import uvicorn

    print(f"▶ 科研助手 Web 界面: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
