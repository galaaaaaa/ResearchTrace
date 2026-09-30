"""Reader：PDF 精读 + 证据抽取 + 论文卡片组装。

流程（每步对未实现的工具桩优雅降级，绝不抛异常到图节点层）：
1. 无 pdf_path → 直接 partial；
2. 撤稿检查（check_retraction）→ 命中则 failed 返回；
3. pdf_tool：parse_pdf + BM25 retrieve_chunks（主问题 k=8 + required_evidence 探测词各 k=3）；
4. paperqa 可选分支：ask_papers 的 citations 作为候选片段，与 BM25 合并去重；
5. 两段式抽取（对照 paper-qa 的"小上下文+扁平输出"设计，嵌套大 schema 一次通过率过低）：
   5a. 选择批：每批 ≤_SELECT_BATCH 片段一次调用，输出仅 selected 扁平数组，
       单批失败只降级该批（保留 BM25 序前 2 条为桩），不影响其他批；
   5b. 卡片独立调用：仅喂已选中片段原文，失败只影响卡片字段，不动证据；
6. 图表多模态（尽力而为）：detect_figure_regions + analyze_figure（VLM 解读标注"须对照原文"）。

核心约束：evidence_text 必须逐字来自候选片段原文；卡片字段无对应内容时填 null。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.llm import get_llm
from src.schemas import EvidenceRecord, PaperCard, PaperRecord, ResearchTask
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import clamp, now_iso, truncate


class ReaderResult(BaseModel):
    """Reader 单篇论文产出：证据列表 + 论文卡片 + 工具调用计数 + 过程备注。"""

    evidence: list[EvidenceRecord] = Field(default_factory=list)
    card: PaperCard
    tool_calls: int = 0
    notes: list[str] = Field(default_factory=list)


class _Selected(BaseModel):
    """LLM 选中的候选片段引用（index 指向提示词中的候选编号）。"""

    index: int
    claim_hint: str = ""
    relevance: float = 0.5


class _CardOut(BaseModel):
    """LLM 抽取的卡片字段（无对应内容时保持 null/空）。"""

    motivation: str | None = None
    method: str | None = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: str | None = None
    limitations: list[str] = Field(default_factory=list)
    is_ablation_supported: bool | None = None


class _Selection(BaseModel):
    """一批候选片段的选择结果（单层扁平数组）。

    旧版是 selected+card 两层嵌套的 _Extraction 一次调用吃全部片段，
    glm/deepseek 真实 run 中 60/60 批次失败全部降级为桩；对照 paper-qa
    docs.py aget_evidence 的"每 chunk 独立小调用 + 扁平输出"后拆成两段。
    """

    selected: list[_Selected] = Field(default_factory=list)


# 选择批大小：对照 paper-qa 保持小上下文（6 片段 × ~800 字），
# 大到 24 片段的嵌套输出在 mid-tier 模型上不可靠（实测全灭）
_SELECT_BATCH = 6


def _dedup_snippets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """候选片段去重（前 160 字符的空白归一化键）并规范为 {page, section, text}。"""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for sn in items:
        text = str(sn.get("text") or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", "", text[:160]).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"page": sn.get("page"), "section": sn.get("section"), "text": text})
    return out


class Reader:
    """PDF 精读 Agent：输入 PaperRecord + ResearchTask，输出证据与论文卡片。"""

    def __init__(self, *, budgets: dict[str, Any] | None = None):
        self._settings = get_settings()
        self._budget_overrides: dict[str, Any] = dict(budgets or {})

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    def read_paper(
        self, paper: PaperRecord, task: ResearchTask, *, llm: Any = None, prefer_paperqa: bool = True
    ) -> ReaderResult:
        """精读一篇论文并抽取证据与卡片。

        Args:
            paper: 目标论文（需有 pdf_path 才能进入解析流程）。
            task: 当前研究子任务（question 驱动 BM25 检索与证据挑选）。
            llm: 注入的 LLM 客户端（默认 get_llm("researcher")，测试用假客户端）。
            prefer_paperqa: 是否优先尝试 PaperQA2 后端（可用时与 BM25 合并）。

        Returns:
            ReaderResult；任何内部异常都会被消化为 read_status="failed" + notes 记录。
        """
        tracer = get_active_tracer()
        tracer.event("node_start", node="reader", paper_id=paper.paper_id, task_id=task.task_id)
        card = PaperCard(paper_id=paper.paper_id)
        result = ReaderResult(card=card)
        try:
            self._read_inner(paper, task, result, llm or get_llm("researcher"), prefer_paperqa)
        except Exception as exc:  # noqa: BLE001 —— 绝不向图节点层抛出
            result.card.read_status = "failed"
            result.notes.append(f"未预期异常: {type(exc).__name__}: {exc}")
            tracer.event("tool_error", node="reader", paper_id=paper.paper_id, error=f"{type(exc).__name__}: {exc}")
        result.card.notes = "；".join(result.notes) if result.notes else None
        tracer.event(
            "node_end",
            node="reader",
            paper_id=paper.paper_id,
            task_id=task.task_id,
            status=result.card.read_status,
            evidence=len(result.evidence),
            tool_calls=result.tool_calls,
        )
        return result

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def _read_inner(self, paper: PaperRecord, task: ResearchTask, result: ReaderResult, llm: Any, prefer_paperqa: bool) -> None:
        card = result.card
        if not paper.pdf_path:
            card.read_status = "partial"
            result.notes.append("no pdf")
            return

        # 1) 撤稿检查（工具未实现时跳过）；命中撤稿直接失败返回
        if self._check_retraction(paper, result):
            return

        # 2) pdf_tool 解析 + BM25 检索
        doc, snippets = self._collect_snippets_pdf(paper, task, result)

        # 3) paperqa 可选分支
        if prefer_paperqa:
            snippets.extend(self._collect_snippets_paperqa(paper, task, result))
        snippets = _dedup_snippets(snippets)[:24]

        # 4) 两段式 LLM 抽取：选择批（证据）→ 卡片（独立，失败不动证据）
        picked = self._select_evidence(paper, task, snippets, result, llm)
        for sn, sel in picked:
            self._append_text_evidence(paper, task, result, sn, sel.claim_hint, sel.relevance)
        self._fill_card(paper, task, picked, result, llm, card)

        # 5) 图表多模态（尽力而为，失败静默跳过）
        self._read_figures(paper, task, result)

        # 6) 扫描件与终态判定
        if doc is not None and getattr(doc, "is_scanned", False):
            result.notes.append("扫描件，证据覆盖可能不全")
            if card.read_status == "ok":
                card.read_status = "partial"
        if not result.evidence and card.read_status == "ok":
            card.read_status = "partial"
            result.notes.append("未抽取到任何证据")

    # ------------------------------------------------------------------
    # 各阶段实现
    # ------------------------------------------------------------------
    @staticmethod
    def _check_retraction(paper: PaperRecord, result: ReaderResult) -> bool:
        """撤稿检查：is_retracted=True → failed 并返回 True；NotImplementedError → 跳过。"""
        try:
            from src.tools.citation_tool import check_retraction
        except (ImportError, NotImplementedError):
            return False
        try:
            status = check_retraction(paper)
        except NotImplementedError:
            return False
        result.tool_calls += 1
        if status is not None and status.is_retracted:
            result.card.read_status = "failed"
            result.notes.append(f"论文已撤稿：{status.notice or 'retraction notice'}")
            return True
        return False

    def _collect_snippets_pdf(
        self, paper: PaperRecord, task: ResearchTask, result: ReaderResult
    ) -> tuple[Any, list[dict[str, Any]]]:
        """parse_pdf + retrieve_chunks；pdf_tool 未实现 → partial + note。"""
        doc = None
        snippets: list[dict[str, Any]] = []
        try:
            from src.tools import pdf_tool
        except (ImportError, NotImplementedError):
            result.card.read_status = "partial"
            result.notes.append("pdf_tool 未实现，跳过本地解析")
            return doc, snippets
        try:
            doc = pdf_tool.parse_pdf(
                paper.pdf_path, paper_id=paper.paper_id, max_pages=int(self._budget("max_pdf_pages", 80))
            )
            result.tool_calls += 1
        except NotImplementedError:
            result.card.read_status = "partial"
            result.notes.append("pdf_tool 未实现，跳过本地解析")
            return None, snippets
        except Exception as exc:  # noqa: BLE001
            result.card.read_status = "partial"
            result.notes.append(f"parse_pdf 失败: {type(exc).__name__}: {exc}")
            return None, snippets
        try:
            snippets.extend(pdf_tool.retrieve_chunks(doc, task.question, k=8))
            result.tool_calls += 1
            for probe in (task.required_evidence or [])[:4]:
                snippets.extend(pdf_tool.retrieve_chunks(doc, str(probe), k=3))
                result.tool_calls += 1
        except NotImplementedError:
            result.notes.append("retrieve_chunks 未实现，仅使用其他候选片段")
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"retrieve_chunks 失败: {type(exc).__name__}: {exc}")
        return doc, snippets

    @staticmethod
    def _collect_snippets_paperqa(paper: PaperRecord, task: ResearchTask, result: ReaderResult) -> list[dict[str, Any]]:
        """PaperQA2 分支：citations(page/text) 作为候选片段；不可用时安静返回空。"""
        try:
            from src.tools import paperqa_tool
        except (ImportError, NotImplementedError):
            return []
        try:
            if not paperqa_tool.paperqa_available():
                return []
            answer = paperqa_tool.ask_papers(task.question, [paper.pdf_path])
            result.tool_calls += 1
        except NotImplementedError:
            return []
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"paperqa 调用失败已回退: {type(exc).__name__}: {exc}")
            return []
        snippets: list[dict[str, Any]] = []
        for cit in (answer or {}).get("citations") or []:
            text = str(cit.get("text") or "").strip()
            if text:
                snippets.append({"page": cit.get("page"), "section": None, "text": text})
        return snippets

    def _select_evidence(
        self, paper: PaperRecord, task: ResearchTask, snippets: list[dict[str, Any]], result: ReaderResult, llm: Any
    ) -> list[tuple[dict[str, Any], _Selected]]:
        """选择批抽取：每批 _SELECT_BATCH 条片段一次 chat_json，输出仅 selected 扁平数组。

        单批失败只降级该批（保留 BM25 序前 2 条为桩），其余批不受影响——
        对照 paper-qa docs.py aget_evidence 的单条失败隔离设计，
        替代旧版"一次嵌套大调用失败则全军覆没"。
        """
        if not snippets:
            result.notes.append("无候选证据片段，跳过 LLM 抽取")
            return []
        system = (
            "你是论文精读 Agent。硬性规则：\n"
            "1. selected 中每个条目的 index 填候选片段的整数编号（如 0、3，不带 # 号），"
            "claim_hint 用一句中文概括该片段对任务问题支撑的结论，不得改写原文事实；\n"
            "2. 只选与任务问题直接相关的片段；都不相关则 selected 为空数组；\n"
            "3. relevance 取 0-1。"
        )
        picked: list[tuple[dict[str, Any], _Selected]] = []
        for start in range(0, len(snippets), _SELECT_BATCH):
            batch = snippets[start : start + _SELECT_BATCH]
            try:
                sel = llm.chat_json(
                    self._build_selection_prompt(paper, task, batch, base_index=start),
                    _Selection,
                    system=system,
                    label="reader:select",
                )
            except Exception as exc:  # noqa: BLE001 —— 本批失败只降级本批
                result.card.read_status = "partial"
                result.notes.append(
                    f"选择批失败降级（#{start}-#{start + len(batch) - 1}）: "
                    f"{type(exc).__name__}: {truncate(str(exc), 120)}"
                )
                for sn in batch[:2]:
                    self._append_text_evidence(paper, task, result, sn, "（LLM 失败，自动保留的相关片段）", 0.4)
                continue
            for s in sel.selected:
                if 0 <= s.index - start < len(batch):
                    picked.append((snippets[s.index], s))
        return picked

    def _fill_card(
        self,
        paper: PaperRecord,
        task: ResearchTask,
        picked: list[tuple[dict[str, Any], _Selected]],
        result: ReaderResult,
        llm: Any,
        card: PaperCard,
    ) -> None:
        """卡片独立抽取：仅喂已选中片段原文；失败只影响卡片字段，不动证据。"""
        if not picked:
            return
        system = (
            "你是论文精读 Agent。卡片字段（motivation/method/datasets/metrics/results/limitations/"
            "is_ablation_supported）必须来自给定片段原文；无对应内容时填 null 或空列表，禁止编造。"
        )
        try:
            out = llm.chat_json(
                self._build_card_prompt(paper, task, picked), _CardOut, system=system, label="reader:card"
            )
        except Exception as exc:  # noqa: BLE001 —— 卡片失败不影响证据
            result.notes.append(f"卡片抽取失败，卡片字段留空: {type(exc).__name__}: {truncate(str(exc), 120)}")
            return
        card.motivation = out.motivation
        card.method = out.method
        card.datasets = list(out.datasets)
        card.metrics = list(out.metrics)
        card.results = out.results
        card.limitations = list(out.limitations)
        card.is_ablation_supported = out.is_ablation_supported

    @staticmethod
    def _build_selection_prompt(
        paper: PaperRecord, task: ResearchTask, batch: list[dict[str, Any]], *, base_index: int
    ) -> str:
        lines = [
            f"论文：{truncate(paper.title, 120)}（{paper.year or '年份未知'}，id={paper.paper_id}）",
            f"任务问题：{task.question}",
        ]
        if task.required_evidence:
            lines.append(f"需要的证据类型：{'；'.join(str(x) for x in task.required_evidence[:6])}")
        lines.append("")
        lines.append("候选证据片段（#编号 | page=页码 | section=章节）：")
        for i, sn in enumerate(batch):
            lines.append(f"#{base_index + i} | page={sn.get('page')} | section={sn.get('section') or '-'}")
            lines.append(truncate(sn.get("text") or "", 800))
        lines.append("")
        lines.append("请选出与任务问题直接相关的片段（selected）。")
        return "\n".join(lines)

    @staticmethod
    def _build_card_prompt(
        paper: PaperRecord, task: ResearchTask, picked: list[tuple[dict[str, Any], _Selected]]
    ) -> str:
        lines = [
            f"论文：{truncate(paper.title, 120)}（{paper.year or '年份未知'}）",
            f"任务问题：{task.question}",
            "",
            "已确认相关的证据片段原文：",
        ]
        for sn, _sel in picked[:8]:
            lines.append(f"--- page={sn.get('page')} ---")
            lines.append(truncate(sn.get("text") or "", 600))
        lines.append("")
        lines.append("请依据以上原文填写论文卡片字段。")
        return "\n".join(lines)

    @staticmethod
    def _append_text_evidence(
        paper: PaperRecord,
        task: ResearchTask,
        result: ReaderResult,
        snippet: dict[str, Any],
        claim_hint: str,
        relevance: float,
    ) -> None:
        """把一个文本片段组装为 EvidenceRecord 并回填卡片引用。"""
        ev = EvidenceRecord(
            paper_id=paper.paper_id,
            task_id=task.task_id,
            claim_hint=claim_hint or "（未提供结论提示）",
            evidence_text=truncate(str(snippet.get("text") or ""), 1200),
            page=snippet.get("page"),
            section=snippet.get("section"),
            modality="text",
            relevance_score=clamp(float(relevance or 0.5), 0.0, 1.0),
            source_url=paper.source_url or "",
            extractor="reader",
            created_at=now_iso(),
        )
        result.evidence.append(ev)
        result.card.evidence_ids.append(ev.evidence_id)

    def _read_figures(self, paper: PaperRecord, task: ResearchTask, result: ReaderResult) -> None:
        """图表多模态阅读：VLM 解读作为辅助证据（标注"须对照原文"），失败静默跳过。"""
        try:
            from src.tools import figure_tool
        except (ImportError, NotImplementedError):
            return
        try:
            if not bool(self._settings.source("vision", "enabled", True)):
                return
            regions = figure_tool.detect_figure_regions(paper.pdf_path) or []
            result.tool_calls += 1
            max_figures = int(self._settings.source("vision", "max_figures_per_paper", 4))
            for region in regions[:max_figures]:
                try:
                    reading = figure_tool.analyze_figure(
                        paper.pdf_path,
                        region.page,
                        region.bbox,
                        question=f"该图表对任务'{task.question[:80]}'提供了什么证据？",
                        llm=get_llm("vision"),
                    )
                    result.tool_calls += 1
                except NotImplementedError:
                    return
                except Exception as exc:  # noqa: BLE001
                    result.notes.append(f"图表解读失败已跳过: {type(exc).__name__}: {exc}")
                    continue
                if reading is None:
                    continue
                numbers = "、".join(str(n) for n in (reading.numbers or []))
                ev = EvidenceRecord(
                    paper_id=paper.paper_id,
                    task_id=task.task_id,
                    claim_hint=f"图表证据提示（{region.kind}，第 {region.page} 页）",
                    evidence_text=truncate(
                        f"[VLM 解读，须对照原文] {reading.description}；数字：{numbers}", 1200
                    ),
                    page=reading.page if reading.page is not None else region.page,
                    section=None,
                    modality=reading.modality,
                    relevance_score=clamp(float(reading.confidence or 0.5), 0.0, 1.0),
                    source_url=paper.source_url or "",
                    extractor="reader",
                    figure_path=reading.figure_path,
                    created_at=now_iso(),
                )
                result.evidence.append(ev)
                result.card.evidence_ids.append(ev.evidence_id)
        except NotImplementedError:
            return
        except Exception as exc:  # noqa: BLE001 —— 多模态失败不影响文本证据
            result.notes.append(f"图表多模态阅读失败已跳过: {type(exc).__name__}: {exc}")

    def _budget(self, key: str, default: Any = None) -> Any:
        if key in self._budget_overrides:
            return self._budget_overrides[key]
        return self._settings.budget(key, default)
