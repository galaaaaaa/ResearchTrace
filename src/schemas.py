"""数据契约：所有 Agent 与工具只交换这些结构（先于 Prompt 固定）。

核心约束：
- Writer 只能引用进入 Evidence Store 的 EvidenceRecord，每条事实性结论生成 ClaimRecord；
- 引用先写内部键 [paper_id:page]，最终统一渲染为参考文献格式；
- 冲突结论写入 ClaimRecord.status = "conflicted"，不允许静默删除一方。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .utils import new_id

Perspective = Literal["background", "method", "experiment", "application", "critique"]
TaskStatus = Literal["pending", "running", "done", "failed"]
TaskOrigin = Literal["planner", "gap", "repair", "seed"]
Modality = Literal["text", "table", "figure", "formula"]
ClaimStatus = Literal["supported", "conflicted", "unsupported", "metadata_error"]


# --------------------------------------------------------------------------
# 研究定义
# --------------------------------------------------------------------------
class ResearchBrief(BaseModel):
    """Scope Agent 输出：研究范围与判断标准。"""

    objective: str
    out_of_scope: list[str] = Field(default_factory=list)
    time_range: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    disciplines: list[str] = Field(default_factory=list)
    paper_types: list[str] = Field(default_factory=list, description="如 survey/full-paper/preprint")
    core_questions: list[str] = Field(default_factory=list)
    output_format: str = "markdown"
    depth: Literal["quick", "standard", "deep"] = "standard"
    criteria: dict[str, str] = Field(default_factory=dict, description="判断标准：效果/成本/数据规模/可复现性等")
    language: str = "zh"
    clarification_needed: bool = False
    clarification_question: str | None = None
    papers_dir: str | None = None
    assumptions: list[str] = Field(default_factory=list, description="未询问用户时填入的默认假设")


class ResearchTask(BaseModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    question: str
    perspective: Perspective = "background"
    required_evidence: list[str] = Field(default_factory=list, description="成功所需证据类型，如 'ablation 结果'")
    success_criteria: str | None = None
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["search_arxiv", "search_semantic_scholar", "search_web_literature", "traverse_citations", "download_pdf", "read_pdf"]
    )
    max_tool_calls: int = 5
    status: TaskStatus = "pending"
    round_created: int = 0
    origin: TaskOrigin = "planner"
    parent_task_id: str | None = None
    seed_paper_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 论文与证据
# --------------------------------------------------------------------------
class PaperRecord(BaseModel):
    paper_id: str = Field(description="DOI 优先，否则 arXiv ID 或内容哈希")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_url: str = ""
    pdf_path: str | None = None
    citation_count: int | None = None
    is_retracted: bool | None = None
    paper_type: Literal["survey", "full-paper", "preprint", "blog", "other"] = "full-paper"
    retrieval_query: str | None = None
    source_api: str | None = None


class EvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: new_id("ev"))
    paper_id: str
    task_id: str
    claim_hint: str = Field(description="该证据支持的结论提示")
    evidence_text: str = Field(description="原始证据片段，而非模型摘要")
    page: int | None = None
    section: str | None = None
    modality: Modality = "text"
    relevance_score: float = 0.5
    source_url: str = ""
    extractor: str | None = None
    figure_path: str | None = Field(default=None, description="modality 非 text 时绑定的截图路径")
    created_at: str | None = None


class PaperCard(BaseModel):
    """Reader 产出的论文卡片（方法/数据/结论/局限）。"""

    card_id: str = Field(default_factory=lambda: new_id("card"))
    paper_id: str
    motivation: str | None = None
    method: str | None = None
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    results: str | None = None
    limitations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    is_ablation_supported: bool | None = Field(default=None, description="主要结论是否有消融支撑")
    read_status: Literal["ok", "partial", "failed"] = "ok"
    notes: str | None = None


class ClaimRecord(BaseModel):
    claim_id: str = Field(default_factory=lambda: new_id("claim"))
    claim_text: str
    evidence_ids: list[str] = Field(default_factory=list)
    citation_keys: list[str] = Field(default_factory=list, description="形如 [paper_id:page] 的内部引用键")
    section: str | None = None
    confidence: float = 0.5
    status: ClaimStatus = "supported"
    verifier_note: str | None = None


class Finding(BaseModel):
    """子 Agent 返回 Supervisor 的压缩结构化结果（不回传完整搜索历史）。"""

    task_id: str
    agent: str = "researcher"
    summary: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    status: Literal["done", "partial", "failed"] = "done"
    error: str | None = None


# --------------------------------------------------------------------------
# 分析产物
# --------------------------------------------------------------------------
class ComparisonRow(BaseModel):
    paper_id: str
    title: str = ""
    values: dict[str, str] = Field(default_factory=dict, description="列名 → 单元格值")


class ComparisonMatrix(BaseModel):
    """Analyst 产出的统一字段方法对比矩阵。"""

    task_id: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[ComparisonRow] = Field(default_factory=list)
    comparability_warnings: list[str] = Field(
        default_factory=list, description="指标单位/数据集版本/评测设置不一致，不能直接横向比较的提示"
    )
    notes: str | None = None


class ConflictReport(BaseModel):
    """Critic 产出的冲突点：并列呈现，不允许强行统一。"""

    conflict_id: str = Field(default_factory=lambda: new_id("conflict"))
    topic: str
    paper_ids: list[str] = Field(default_factory=list)
    description: str = ""
    possible_causes: list[str] = Field(default_factory=list, description="实验设置/数据/定义差异的解释")
    severity: Literal["high", "medium", "low"] = "medium"
    evidence_ids: list[str] = Field(default_factory=list)


class Gap(BaseModel):
    """Gap Analyzer 检测到的证据缺口。"""

    task_id: str | None = None
    description: str
    severity: Literal["high", "medium", "low"] = "medium"
    reason: Literal[
        "no_evidence", "single_source", "no_experiment_support", "unexplained_conflict", "stale_sources", "low_quality"
    ] = "no_evidence"
    fix_question: str | None = Field(default=None, description="转换成的定向任务问题")


class GapReport(BaseModel):
    sufficient: bool = False
    gaps: list[Gap] = Field(default_factory=list)
    new_tasks: list[ResearchTask] = Field(default_factory=list)


# --------------------------------------------------------------------------
# 写作与核验
# --------------------------------------------------------------------------
class OutlineSection(BaseModel):
    title: str
    key_points: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    perspective: Perspective | None = None


class VerificationCheck(BaseModel):
    check_id: str = Field(default_factory=lambda: new_id("chk"))
    claim_id: str | None = None
    layer: Literal["metadata", "entailment", "coverage", "consistency"]
    label: Literal[
        "entailed", "partial", "contradicted", "unrelated", "ok", "conflict", "citation_missing", "metadata_error"
    ]
    reason: str = ""
    missing_information: str | None = None
    sentence: str | None = Field(default=None, description="coverage 层标注的无引用事实句")


class VerificationReport(BaseModel):
    checks: list[VerificationCheck] = Field(default_factory=list)
    claim_status: dict[str, ClaimStatus] = Field(default_factory=dict)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    metadata_error_claim_ids: list[str] = Field(default_factory=list)
    coverage_missing: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list, description="一致性层发现的结论冲突描述")
    passed: bool = False
    summary: str | None = None


# --------------------------------------------------------------------------
# 工具层产物
# --------------------------------------------------------------------------
class ToolCallLog(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    duration_ms: int = 0
    result_summary: str = ""
    error: str | None = None


class MetadataCheck(BaseModel):
    """元数据核验（Crossref/Semantic Scholar 对照）。"""

    doi_valid: bool | None = None
    title_similarity: float | None = None
    authors_match: bool | None = None
    year_match: bool | None = None
    is_retracted: bool | None = None
    corrected_by: list[str] = Field(default_factory=list)
    source: str = "crossref"
    note: str | None = None


class RetractionStatus(BaseModel):
    paper_id: str
    is_retracted: bool | None = None
    notice: str | None = None
    source: str = "crossref"


class FigureReading(BaseModel):
    """VLM 对图/表/公式截图的结构化解读，必须绑定页码与原图。"""

    figure_path: str
    page: int | None = None
    figure_no: str | None = None
    modality: Modality = "figure"
    description: str = ""
    numbers: list[str] = Field(default_factory=list, description="从图中读出的关键数字")
    caveats: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class PaperDocInfo(BaseModel):
    """parse_pdf 的结构化结果索引（正文细节在返回对象上）。"""

    pdf_path: str
    paper_id: str | None = None
    n_pages: int = 0
    n_chunks: int = 0
    is_scanned: bool = False
    sections: list[str] = Field(default_factory=list)
    sha256: str | None = None
