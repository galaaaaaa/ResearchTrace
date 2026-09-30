"""Searcher：受限 ReAct 检索循环。

设计要点：
- 工具白名单 + 每任务调用硬顶（max_tool_calls_per_researcher），LLM 迭代上限 = 2*max_calls+2；
- 无效动作（自创工具名）不消耗工具预算：在下一轮用户消息末尾注入纠正反馈（附可用工具清单，
  促使模型自我纠正），连续 3 次才出局——实测弱模型幻觉工具名频率高，烧预算+两振出局会让
  任务空手而归；
- 中文查询（任务问题或模型给的 query）自动译成英文检索式再进搜索工具（arXiv/S2 对中文
  几乎零命中；翻译 LLM 失败时原样透传）；
- LLM 失败的兜底检索轮换引擎：arxiv 用同一查询搜过则改试 semantic_scholar；
- 工具桩未实现（NotImplementedError/ImportError）→ 记 ToolCallLog(ok=False, error="unavailable")
  并从后续轮次的可用工具中剔除，流程继续；
- 相邻两次搜索词相似度超过 similar_query_stop → 停滞提前停止（懒导入 supervisor.detect_stagnation，
  不可用时用 rapidfuzz token_set_ratio 自行实现同逻辑）；
- 循环结束后按引用数对无 PDF 的 arXiv 论文补下载（受 max_pdfs_downloaded_per_task 约束）；
- 任何未预期异常都被内部消化，返回已收集的部分 SearchResult（stopped_reason="error:..."），绝不向上抛。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.llm import get_llm
from src.schemas import PaperRecord, ResearchBrief, ResearchTask, ToolCallLog
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import norm_title, truncate

# 模块级哨兵：注册表中"已知默认实现不可用"的标记
_UNAVAILABLE: Any = object()

# 搜索类工具（其 query 参与停滞检测；web_literature 兜底轮换也按此序）
_SEARCH_TOOLS: tuple[str, ...] = ("search_arxiv", "search_semantic_scholar", "search_web_literature")

# 工具签名说明（进入 system 提示词）
_TOOL_SPECS: dict[str, str] = {
    "search_arxiv": "search_arxiv(query, date_from='YYYY-MM-DD', date_to='YYYY-MM-DD', max_results=int) -> list[PaperRecord]",
    "search_semantic_scholar": "search_semantic_scholar(query, max_results=int, year='2020-2024'|'2020-'|'-2024') -> list[PaperRecord]",
    "search_web_literature": "search_web_literature(query, max_results=int) -> list[PaperRecord]（学术网页垂直域检索：arXiv API 不可用、或需要非 arXiv 来源（技术报告/博客/数据集）时用）",
    "traverse_citations": "traverse_citations(paper_id, direction='citing'|'cited', limit=int) -> list[PaperRecord]",
    "lookup_crossref": "lookup_crossref(title_or_doi) -> PaperRecord | None",
    "download_pdf": "download_pdf(paper_id) -> pdf_path",
}


class SearchResult(BaseModel):
    """Searcher 单任务产出：去重后的论文、完整工具日志、搜索词与下载结果。"""

    papers: list[PaperRecord] = Field(default_factory=list)
    tool_logs: list[ToolCallLog] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    downloads: dict[str, str] = Field(default_factory=dict, description="paper_id → pdf_path")
    stopped_reason: str = "budget"


class ReactStep(BaseModel):
    """ReAct 单步决策：正常动作或 action=finish。"""

    thought: str = ""
    action: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


_CJK_RE = re.compile(r"[一-鿿]")


@dataclass
class _LoopCtx:
    """单次 discover 的循环上下文（工具注册表、去重键、结果容器）。"""

    task: ResearchTask
    brief: ResearchBrief
    result: SearchResult
    registry: dict[str, Callable[..., Any]] = field(default_factory=dict)
    available: list[str] = field(default_factory=lambda: list(_TOOL_SPECS))
    seen_ids: set[str] = field(default_factory=set)
    seen_titles: set[str] = field(default_factory=set)
    translated: dict[str, str] = field(default_factory=dict)  # 中文查询 → 英文检索式（缓存）


def _to_int(value: Any, default: int | None) -> int | None:
    """宽松整数转换（LLM 可能给字符串数字）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_paper_list(ret: Any) -> list[PaperRecord]:
    """把工具返回值规整为 PaperRecord 列表（兼容 None/单对象/列表）。"""
    if ret is None:
        return []
    if isinstance(ret, PaperRecord):
        return [ret]
    if isinstance(ret, (list, tuple)):
        return [p for p in ret if isinstance(p, PaperRecord)]
    return []


# 模型常见的自创动作名 → 真实工具（含语义别名与子串关键词）
_ACTION_ALIASES = {
    "search": "search_arxiv", "search_paper": "search_arxiv", "search_papers": "search_arxiv",
    "search_literature": "search_web_literature", "search_web": "search_web_literature",
    "web_search": "search_web_literature", "web_search_literature": "search_web_literature",
    "search_web_papers": "search_web_literature",
    "arxiv": "search_arxiv", "arxiv_search": "search_arxiv", "arxiv_api": "search_arxiv",
    "search_scholar": "search_semantic_scholar", "scholar": "search_semantic_scholar",
    "semantic_scholar": "search_semantic_scholar", "s2": "search_semantic_scholar",
    "citations": "traverse_citations", "citation_traverse": "traverse_citations", "traverse": "traverse_citations",
    "citation_search": "traverse_citations", "references": "traverse_citations",
    "lookup": "lookup_crossref", "crossref": "lookup_crossref", "verify_doi": "lookup_crossref",
    "doi_lookup": "lookup_crossref", "metadata": "lookup_crossref",
    "download": "download_pdf", "fetch_pdf": "download_pdf", "get_pdf": "download_pdf",
    "pdf": "download_pdf", "download_paper": "download_pdf",
    "stop": "finish", "done": "finish", "submit": "finish", "结束": "finish", "完成": "finish",
}
_ACTION_KEYWORDS = (
    ("arxiv", "search_arxiv"), ("scholar", "search_semantic_scholar"),
    ("web", "search_web_literature"), ("网页", "search_web_literature"), ("联网", "search_web_literature"),
    ("citation", "traverse_citations"), ("引用", "traverse_citations"),
    ("crossref", "lookup_crossref"), ("doi", "lookup_crossref"), ("元数据", "lookup_crossref"),
    ("pdf", "download_pdf"), ("下载", "download_pdf"),
    ("finish", "finish"), ("停止", "finish"), ("结束", "finish"), ("完成", "finish"), ("足够", "finish"),
    ("检索", "search_arxiv"), ("文献", "search_arxiv"), ("搜索", "search_arxiv"), ("search", "search_arxiv"),
)


def _normalize_action(raw: str, available: set[str]) -> str:
    """把模型自创/中文/变形动作名归一到真实工具名；无法归一时原样返回（走 invalid_action 路径）。"""
    a = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not a or a in available or a == "finish":
        return a
    if a in _ACTION_ALIASES:
        mapped = _ACTION_ALIASES[a]
        return mapped if mapped == "finish" or mapped in available else a
    lowered = (raw or "").lower()
    for keyword, mapped in _ACTION_KEYWORDS:  # 中文/描述式动作：按关键词命中
        if keyword in lowered:
            return mapped if mapped == "finish" or mapped in available else a
    return a


def _load_default_tool(name: str) -> Callable[..., Any]:
    """按工具名懒导入 src.tools 下的默认实现；模块缺失时抛 ImportError。"""
    if name == "search_arxiv":
        from src.tools.arxiv_tool import search_arxiv

        return search_arxiv
    if name in ("search_semantic_scholar", "traverse_citations"):
        from src.tools import semantic_scholar_tool

        return getattr(semantic_scholar_tool, name)
    if name == "lookup_crossref":
        from src.tools.crossref_tool import lookup_crossref

        return lookup_crossref
    if name == "search_web_literature":
        from src.tools.web_search import search_web_literature

        return search_web_literature
    if name == "download_pdf":
        from src.tools.pdf_download_tool import fetch_paper_pdf

        # 通用下载器：arXiv 优先，非 arXiv 的 source_url PDF 直链（校园订阅 IP
        # 实测可抓 Nature/Springer）也吃；落地页不硬来
        return fetch_paper_pdf
    raise KeyError(f"未知工具: {name}")


class Searcher:
    """受限 ReAct 搜索 Agent：每轮由 LLM 决定一个工具调用，硬顶预算内最大化论文发现。"""

    def __init__(self, *, budgets: dict[str, Any] | None = None, max_calls: int | None = None):
        self._settings = get_settings()
        self._budget_overrides: dict[str, Any] = dict(budgets or {})
        if max_calls is not None:
            self._budget_overrides["max_tool_calls_per_researcher"] = max_calls
        self.max_calls = int(self._budget("max_tool_calls_per_researcher", 5))

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    def discover(
        self,
        task: ResearchTask,
        brief: ResearchBrief,
        *,
        existing_papers: list[PaperRecord] | None = None,
        tool_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> SearchResult:
        """执行一次受限检索循环。

        Args:
            task: 当前研究子任务（question / required_evidence）。
            brief: 研究范围（年份约束会作为搜索工具的默认 date/year 参数）。
            existing_papers: 已有论文（只参与去重，不进入返回的 papers）。
            tool_registry: 测试注入的假工具（dict[name, callable]，覆盖默认实现）。

        Returns:
            SearchResult；任何内部异常都会被消化为 stopped_reason="error:..."。
        """
        tracer = get_active_tracer()
        tracer.event("node_start", node="searcher", task_id=task.task_id, max_calls=self.max_calls)
        result = SearchResult()
        try:
            ctx = _LoopCtx(
                task=task,
                brief=brief,
                result=result,
                registry=dict(tool_registry or {}),
                seen_ids={(p.paper_id or "").strip() for p in (existing_papers or []) if p.paper_id},
                seen_titles={norm_title(p.title) for p in (existing_papers or []) if norm_title(p.title)},
            )
            self._run_loop(ctx)
            self._download_missing(ctx)
            self._truncate_papers(ctx)
        except Exception as exc:  # noqa: BLE001 —— 绝不向图节点层抛出
            result.stopped_reason = f"error:{type(exc).__name__}: {exc}"
            tracer.event("tool_error", node="searcher", task_id=task.task_id, error=result.stopped_reason)
        tracer.event(
            "node_end",
            node="searcher",
            task_id=task.task_id,
            papers=len(result.papers),
            tool_calls=len(result.tool_logs),
            stopped_reason=result.stopped_reason,
        )
        return result

    # ------------------------------------------------------------------
    # ReAct 主循环
    # ------------------------------------------------------------------
    def _run_loop(self, ctx: _LoopCtx) -> None:
        tracer = get_active_tracer()
        llm = get_llm("researcher")
        executed = 0
        consecutive_bad = 0
        stopped = "budget"
        correction = ""  # 上一轮无效动作的纠正反馈（附在用户消息末尾，近因位置）
        max_iters = self.max_calls * 2 + 2

        for _round in range(max_iters):
            if executed >= self.max_calls:
                stopped = "budget"
                break
            try:
                step = llm.chat_json(
                    self._user_prompt(ctx.task, correction),
                    ReactStep,
                    system=self._system_prompt(ctx.task, ctx.brief, ctx.available, ctx.result.papers),
                    label="searcher:react",
                )
            except Exception as exc:  # noqa: BLE001 —— LLM 失败走默认搜索词重试
                tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, error=f"react_llm:{exc}")
                step = None

            action = _normalize_action((step.action or "").strip(), set(ctx.available)) if step is not None else ""
            if step is None:
                # LLM 客观失败（超时/解析错误）：兜底检索一次（真实搜索，计预算）
                consecutive_bad += 1
                if consecutive_bad > 2:
                    stopped = "error:llm_failure"
                    break
                ok = self._fallback_search(ctx, llm)
                executed += 1
                if not ok:
                    stopped = "error:fallback_search_failed"
                    break
                continue
            if action != "finish" and action not in ctx.available:
                # 无效动作（模型幻觉工具名）：不烧预算，注入纠正反馈后重试
                consecutive_bad += 1
                tracer.event(
                    "tool_error", node="searcher", task_id=ctx.task.task_id, error=f"invalid_action:{action}"
                )
                if consecutive_bad > 2:
                    stopped = f"error:invalid_action:{action}"
                    break
                correction = (
                    f"\n\n（系统提示：你上一轮的 action {step.action!r} 不是可用工具，该轮已被拒绝且不消耗预算。"
                    f"可用工具仅限：{', '.join(ctx.available)} 或 finish。请严格按其中一个名字重新输出 JSON。）"
                )
                continue

            consecutive_bad = 0
            correction = ""
            if action == "finish":
                stopped = "finished"
                break

            args = dict(step.args or {})
            if action in _SEARCH_TOOLS:
                query = self._english_query(llm, ctx, str(args.get("query") or ctx.task.question))
                args["query"] = query
                if ctx.result.queries and self._is_stagnant(query, ctx.result.queries[-1]):
                    stopped = "stagnation"
                    break
            self._call_tool(ctx, action, args)
            executed += 1

        ctx.result.stopped_reason = stopped

    def _english_query(self, llm: Any, ctx: _LoopCtx, query: str) -> str:
        """中文查询 → 英文检索式（arXiv/S2 对中文几乎零命中）；非中文/翻译失败原样返回。"""
        q = (query or "").strip()
        if not q or not _CJK_RE.search(q):
            return q
        if q in ctx.translated:
            return ctx.translated[q]
        en = ""
        try:
            out = llm.chat(
                prompt="把下面的问题翻译成适合学术论文检索的英文关键词查询（空格分隔，不超过 12 个词，"
                f"保留术语与缩写，只输出查询本身）：\n{truncate(q, 300)}",
                label="searcher:translate",
            )
            en = " ".join(out.strip().split())[:200]
        except Exception:  # noqa: BLE001 —— 翻译失败不阻塞检索
            en = ""
        if not en or _CJK_RE.search(en):
            en = q
        ctx.translated[q] = en
        get_active_tracer().event("tool_call", tool="translate_query", src=truncate(q, 60), dst=truncate(en, 60))
        return en

    def _fallback_search(self, ctx: _LoopCtx, llm: Any) -> bool:
        """LLM 失败时的兜底：任务问题（中文自动译英）在未用过该查询的引擎上检索一次。"""
        query = self._english_query(llm, ctx, ctx.task.question)
        for name in _SEARCH_TOOLS:
            if name not in ctx.available:
                continue
            if query in ctx.result.queries and any(
                log.tool == name and log.ok and log.args.get("query") == query for log in ctx.result.tool_logs
            ):
                continue  # 该引擎已用同一查询成功搜过 → 换下一个引擎
            return self._call_tool(ctx, name, {"query": query})
        return False

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------
    def _call_tool(self, ctx: _LoopCtx, name: str, args: dict[str, Any]) -> bool:
        """执行一个工具调用：记日志、合并论文、剔除不可用工具。返回是否成功。"""
        tracer = get_active_tracer()
        fn = ctx.registry.get(name, _UNAVAILABLE)
        if fn is _UNAVAILABLE or fn is None:
            try:
                fn = _load_default_tool(name)
                ctx.registry[name] = fn
            except (ImportError, KeyError):
                ctx.registry[name] = _UNAVAILABLE
                self._disable(ctx, name)
                self._log(ctx, name, dict(args), ok=False, error="unavailable", duration_ms=0, summary="")
                tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, tool=name, error="unavailable")
                return False

        t0 = time.time()
        try:
            if name == "download_pdf":
                return self._exec_download(ctx, name, dict(args), fn, t0)
            kwargs = self._prepare_args(name, dict(args), ctx.task, ctx.brief)
            if kwargs is None:
                self._log(ctx, name, dict(args), ok=False, error="missing required arg", duration_ms=int((time.time() - t0) * 1000), summary="")
                tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, tool=name, error="missing required arg")
                return False
            ret = fn(**kwargs)
            papers = _as_paper_list(ret)
            added = self._merge_papers(ctx, papers, query=str(kwargs.get("query") or ""), source=name)
            if name in _SEARCH_TOOLS:
                ctx.result.queries.append(str(kwargs.get("query") or ctx.task.question))
            summary = "；".join(truncate(p.title, 60) for p in papers[:3])
            duration_ms = int((time.time() - t0) * 1000)
            self._log(ctx, name, kwargs, ok=True, error=None, duration_ms=duration_ms, summary=summary or f"{len(papers)} papers")
            tracer.event(
                "tool_call", node="searcher", task_id=ctx.task.task_id, tool=name, ok=True,
                papers=len(papers), papers_added=added, duration_ms=duration_ms,
            )
            return True
        except NotImplementedError:
            duration_ms = int((time.time() - t0) * 1000)
            self._disable(ctx, name)
            self._log(ctx, name, dict(args), ok=False, error="unavailable", duration_ms=duration_ms, summary="")
            tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, tool=name, error="unavailable")
            return False
        except Exception as exc:  # noqa: BLE001 —— 工具失败只记日志，不中断循环
            duration_ms = int((time.time() - t0) * 1000)
            self._log(ctx, name, dict(args), ok=False, error=f"{type(exc).__name__}: {exc}", duration_ms=duration_ms, summary="")
            tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, tool=name, error=f"{type(exc).__name__}: {exc}")
            return False

    def _exec_download(
        self, ctx: _LoopCtx, name: str, args: dict[str, Any], fn: Callable[..., Any], t0: float
    ) -> bool:
        """执行 download_pdf：定位论文 → 下载 → 回填 pdf_path 与 downloads。"""
        tracer = get_active_tracer()
        paper_id = str(args.get("paper_id") or "")
        paper = next((p for p in ctx.result.papers if p.paper_id == paper_id), None)
        if paper is None:
            self._log(ctx, name, {"paper_id": paper_id}, ok=False, error="paper not found",
                      duration_ms=int((time.time() - t0) * 1000), summary=f"paper_id={paper_id} 不在已发现列表")
            tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, tool=name, error="paper not found")
            return False
        path = fn(paper)
        duration_ms = int((time.time() - t0) * 1000)
        if path:
            paper.pdf_path = str(path)
            ctx.result.downloads[paper.paper_id] = str(path)
            self._log(ctx, name, {"paper_id": paper_id}, ok=True, error=None, duration_ms=duration_ms,
                      summary=truncate(str(path), 120))
            tracer.event("tool_call", node="searcher", task_id=ctx.task.task_id, tool=name, ok=True, paper_id=paper_id)
            return True
        self._log(ctx, name, {"paper_id": paper_id}, ok=False, error="download_failed", duration_ms=duration_ms, summary="")
        tracer.event("tool_error", node="searcher", task_id=ctx.task.task_id, tool=name, error="download_failed")
        return False

    def _prepare_args(
        self, name: str, args: dict[str, Any], task: ResearchTask, brief: ResearchBrief
    ) -> dict[str, Any] | None:
        """把 LLM 给的 args 映射为工具关键字参数；brief 年份约束作默认值；缺必填参数返回 None。"""
        query = str(args.get("query") or task.question)
        year_from, year_to = brief.year_from, brief.year_to
        if name == "search_arxiv":
            return {
                "query": query,
                "date_from": str(args.get("date_from") or (f"{year_from}-01-01" if year_from else "") or None),
                "date_to": str(args.get("date_to") or (f"{year_to}-12-31" if year_to else "") or None),
                "max_results": _to_int(args.get("max_results"), None),
            }
        if name == "search_web_literature":
            return {
                "query": query,
                "max_results": _to_int(args.get("max_results"), None),
            }
        if name == "search_semantic_scholar":
            if year_from and year_to:
                default_year = f"{year_from}-{year_to}"
            elif year_from:
                default_year = f"{year_from}-"
            elif year_to:
                default_year = f"-{year_to}"
            else:
                default_year = None
            return {
                "query": query,
                "max_results": _to_int(args.get("max_results"), None),
                "year": str(args.get("year") or default_year or "") or None,
            }
        if name == "traverse_citations":
            paper_id = str(args.get("paper_id") or "").strip()
            if not paper_id:
                return None
            return {
                "paper_id": paper_id,
                "direction": str(args.get("direction") or "citing"),
                "limit": _to_int(args.get("limit"), 8) or 8,
            }
        if name == "lookup_crossref":
            title_or_doi = str(args.get("title_or_doi") or args.get("query") or "").strip()
            if not title_or_doi:
                return None
            return {"title_or_doi": title_or_doi}
        return None

    # ------------------------------------------------------------------
    # 论文合并与后处理
    # ------------------------------------------------------------------
    def _merge_papers(self, ctx: _LoopCtx, papers: list[PaperRecord], *, query: str, source: str) -> int:
        """按 paper_id + norm_title 两级去重合并；重复记录用于补全已有字段。返回新增数。"""
        added = 0
        for p in papers:
            pid = (p.paper_id or "").strip()
            nt = norm_title(p.title)
            if (pid and pid in ctx.seen_ids) or (nt and nt in ctx.seen_titles):
                self._enrich_existing(ctx.result.papers, pid, nt, p)
                continue
            if pid:
                ctx.seen_ids.add(pid)
            if nt:
                ctx.seen_titles.add(nt)
            if query and not p.retrieval_query:
                p.retrieval_query = query
            if source and not p.source_api:
                p.source_api = source
            self._inherit_pdf_path(p)
            ctx.result.papers.append(p)
            added += 1
        return added

    @staticmethod
    def _inherit_pdf_path(paper: PaperRecord) -> None:
        """跨 run 复用已下载的 PDF（sqlite 记忆的 find_paper 归一匹配 arxiv/doi/title）。

        同一论文不再重复下载与重扫；文件不存在则不继承（防陈旧路径）。
        继承失败=退正常下载路径，绝不抛异常。
        """
        if paper.pdf_path:
            return
        # 只信 arxiv_id / doi 这类强标识做跨 run 继承：title_norm 压平成纯 [a-z0-9] 后
        # 不同论文可碰撞（对抗评审确认），仅凭标题匹配可能拿错 PDF
        if not (paper.arxiv_id or paper.doi):
            return
        try:
            from pathlib import Path

            from src.memory.evidence_store import EvidenceStore

            store = EvidenceStore(get_settings().db_path)
            try:
                if paper.arxiv_id:
                    known = store.find_paper(arxiv_id=paper.arxiv_id)
                else:
                    known = store.find_paper(doi=paper.doi)
            finally:
                store.close()
            if known is not None and known.pdf_path and Path(known.pdf_path).exists():
                paper.pdf_path = known.pdf_path
                get_active_tracer().event(
                    "tool_call", node="searcher", tool="inherit_pdf",
                    paper_id=paper.paper_id, path=known.pdf_path,
                )
        except Exception:  # noqa: BLE001 —— 继承失败=正常下载路径
            pass

    @staticmethod
    def _enrich_existing(existing: list[PaperRecord], pid: str, nt: str, cand: PaperRecord) -> None:
        """用重复命中的一条记录补全已有记录的缺失元数据（轻量合并）。"""
        for p in existing:
            if (pid and p.paper_id == pid) or (nt and norm_title(p.title) == nt):
                if p.citation_count is None and cand.citation_count is not None:
                    p.citation_count = cand.citation_count
                if not p.doi and cand.doi:
                    p.doi = cand.doi
                if not p.venue and cand.venue:
                    p.venue = cand.venue
                if not p.abstract and cand.abstract:
                    p.abstract = cand.abstract
                if not p.arxiv_id and cand.arxiv_id:
                    p.arxiv_id = cand.arxiv_id
                return

    def _download_missing(self, ctx: _LoopCtx) -> None:
        """循环结束后：对无 pdf_path 的 arXiv 论文按引用数/相关度排序补下载。"""
        tracer = get_active_tracer()
        if not bool(self._settings.source("search", "download_pdfs", True)):
            return
        if "download_pdf" not in ctx.available:
            return
        limit = int(self._budget("max_pdfs_downloaded_per_task", 5))
        candidates = [p for p in ctx.result.papers if p.arxiv_id and not p.pdf_path]
        # 引用数降序；同引用数保持发现顺序（相关度代理）
        candidates.sort(key=lambda p: -(p.citation_count or 0))
        for paper in candidates[:limit]:
            self._call_tool(ctx, "download_pdf", {"paper_id": paper.paper_id})

    def _truncate_papers(self, ctx: _LoopCtx) -> None:
        """papers 截断到 max_papers_per_task，downloads 同步裁剪。"""
        limit = int(self._budget("max_papers_per_task", 8))
        if len(ctx.result.papers) > limit:
            keep_ids = {p.paper_id for p in ctx.result.papers[:limit]}
            ctx.result.downloads = {k: v for k, v in ctx.result.downloads.items() if k in keep_ids}
            ctx.result.papers = ctx.result.papers[:limit]

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _is_stagnant(self, new_query: str, last_query: str) -> bool:
        """相邻两次搜索词是否高度重复（>= similar_query_stop）。"""
        if not last_query or not new_query:
            return False
        threshold = float(self._budget("similar_query_stop", 0.85))
        try:
            from src.agents.supervisor import detect_stagnation  # 并行开发中，签名未定

            return bool(detect_stagnation([last_query, new_query], threshold))
        except Exception:  # noqa: BLE001 —— 导入/签名不符时退化为本地实现
            try:
                from rapidfuzz import fuzz

                return fuzz.token_set_ratio(last_query, new_query) / 100.0 >= threshold
            except Exception:  # noqa: BLE001
                return last_query.strip().lower() == new_query.strip().lower()

    def _budget(self, key: str, default: Any = None) -> Any:
        if key in self._budget_overrides:
            return self._budget_overrides[key]
        return self._settings.budget(key, default)

    @staticmethod
    def _disable(ctx: _LoopCtx, name: str) -> None:
        if name in ctx.available:
            ctx.available.remove(name)

    @staticmethod
    def _log(
        ctx: _LoopCtx, tool: str, args: dict[str, Any], *, ok: bool, error: str | None, duration_ms: int, summary: str
    ) -> None:
        ctx.result.tool_logs.append(
            ToolCallLog(tool=tool, args=dict(args), ok=ok, duration_ms=duration_ms, result_summary=summary, error=error)
        )

    # ------------------------------------------------------------------
    # 提示词
    # ------------------------------------------------------------------
    def _system_prompt(
        self, task: ResearchTask, brief: ResearchBrief, available: list[str], papers: list[PaperRecord]
    ) -> str:
        tool_lines = "\n".join(f"- {_TOOL_SPECS[n]}" for n in available)
        scope_lines: list[str] = []
        if brief.objective:
            scope_lines.append(f"研究目标：{brief.objective}")
        if brief.year_from or brief.year_to:
            scope_lines.append(f"年份范围：{brief.year_from or '…'}–{brief.year_to or '…'}（作为 date/year 参数传入搜索工具）")
        if brief.out_of_scope:
            scope_lines.append(f"超出范围（不要检索）：{'；'.join(brief.out_of_scope[:5])}")
        if brief.core_questions:
            scope_lines.append(f"核心问题：{'；'.join(brief.core_questions[:5])}")
        paper_lines = "\n".join(f"- {truncate(p.title, 80)} | {p.paper_id} | {p.year}" for p in papers[:12]) or "（暂无）"
        return (
            "你是受限 ReAct 论文检索 Agent。\n"
            "每轮只输出一个 JSON 对象：{\"thought\": \"...\", \"action\": \"<工具名>\", \"args\": {...}}；"
            "信息足够或继续搜索无益时输出 {\"action\": \"finish\", \"reason\": \"...\"}。\n"
            "action 必须逐字使用「可用工具」中列出的名字（或 finish），禁止自创工具名、"
            "禁止用中文描述代替工具名；query 优先用英文检索式；不要重复与已执行高度相似的 query。\n\n"
            f"可用工具：\n{tool_lines}\n\n"
            f"任务问题：{task.question}\n"
            + ("\n".join(scope_lines) + "\n" if scope_lines else "")
            + f"\n已发现论文（题名 | id | 年份）：\n{paper_lines}"
        )

    @staticmethod
    def _user_prompt(task: ResearchTask, correction: str = "") -> str:
        return f"任务问题：{task.question}\n请输出下一步动作的 JSON。" + correction
