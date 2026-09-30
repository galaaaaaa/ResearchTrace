"""Writer：提纲生成、逐节写作、结论抽取与引用编号渲染（文档 §7.8）。

- generate_outline：Outline Agent，输入 Brief / 各任务证据覆盖 / 缺口，输出 4-8 节提纲；LLM 失败退 5 节模板；
- write_report：逐节写作（每节一次 chat 调用），写完后用 fast 角色独立抽取 ClaimRecord，
  引用键 [paper_id:page] 与 evidence_id 互相补全；任一环节失败退"结构化事实清单"，流程绝不中断；
- render_final：把内部引用键统一渲染为编号引用 [n]，按首次出现顺序编号，文末追加参考文献。
"""

from __future__ import annotations

import re
from typing import Sequence, get_args

from pydantic import BaseModel, Field

from ..llm import LLMClient, get_fake_llm, get_llm
from ..schemas import (
    ClaimRecord,
    ComparisonMatrix,
    ConflictReport,
    EvidenceRecord,
    Gap,
    OutlineSection,
    PaperCard,
    PaperRecord,
    Perspective,
    ResearchBrief,
    ResearchTask,
)
from ..settings import get_settings
from ..tracing import get_active_tracer
from ..utils import clamp, truncate

try:  # rapidfuzz 是声明依赖；极端环境下退化为标准库相似度
    from rapidfuzz import fuzz

    def _sim(a: str, b: str) -> float:
        return fuzz.partial_ratio(a or "", b or "") / 100.0

except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a or "", b or "").ratio()


VALID_PERSPECTIVES: frozenset[str] = frozenset(get_args(Perspective))
MAJOR_PERSPECTIVES: tuple[str, ...] = ("background", "method", "experiment", "critique")
MIN_SECTIONS = 4
MAX_SECTIONS = 8

# 内部引用键：[paper_id] 或 [paper_id:page]；不吞 markdown 链接 [text](url)，不吞纯数字编号 [1]。
# paper_id 本身可含冒号（如本地 PDF 的 "sha256:<hash>"），故 id 段用非贪婪并允许 ":"，
# 页码固定为结尾的 ":数字]"，保证 [sha256:abc:9] 唯一解析为 id="sha256:abc"、page=9。
_CITE_KEY_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._\-/@:]*?)(?::(\d+))?\](?!\()")
# claims 中 citation_keys 的宽松形式（可能不带冒号页码）
_KEY_IN_BRACKETS_RE = re.compile(r"\[([^\[\]]+?)(?::(\d+))?\]")


class ReportDraft(BaseModel):
    """Writer 输出：报告草稿（内部引用键，未渲染）+ 从中抽取的结论列表。"""

    markdown: str
    claims: list[ClaimRecord] = Field(default_factory=list)


class _SectionDraft(BaseModel):
    title: str = ""
    key_points: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    perspective: str | None = None


class _OutlineDraft(BaseModel):
    sections: list[_SectionDraft] = Field(default_factory=list)


class _ClaimDraft(BaseModel):
    claim_text: str = ""
    citation_keys: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    section: str | None = None
    confidence: float = 0.5


class _ClaimsDraft(BaseModel):
    claims: list[_ClaimDraft] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Prompt 模板（#FAKE:<name> 标记供离线测试的 FakeBackend 注册响应选择）
# --------------------------------------------------------------------------
_OUTLINE_SYSTEM = (
    "你是学术综述提纲规划 Agent。根据研究范围、各子问题的证据覆盖情况与已知证据缺口，规划一份 4-8 节的中文综述提纲："
    "background / method / experiment / critique 等视角应尽量覆盖；证据充足的子问题优先安排章节；标题具体、不空泛。\n"
    "#FAKE:outline"
)

_WRITER_SYSTEM = (
    "你是严谨的学术综述写作 Agent。写作硬约束：\n"
    "1. 每个事实性陈述（数字、比较、作者结论、时间信息）必须紧跟内部引用键，格式 [paper_id:page]，可多个，"
    "如 [paperA:3][paperB:7]；引用键只能取自下方给定证据中出现的键，禁止凭空构造。\n"
    "2. 只能使用给定证据与论文卡片中的信息，禁止引入外部知识或自行推测数据。\n"
    "3. 相关性不得写成因果性，措辞强度必须与证据强度匹配。\n"
    "4. 冲突证据必须并列呈现（如“A 报告 X [p1:3]，但 B 在不同设置下报告 Y [p2:5]”），不得静默删除一方。\n"
    "5. 无证据支撑处使用“目前证据不足”“尚无定论”等不确定性表述。\n"
    "6. 若给出对比矩阵，用 markdown table 渲染嵌入，可增删行列但不得改动数值；同时说明指标口径差异。\n"
    "7. 用中文写作，输出本节正文（不要重复小节标题，不要输出提纲）。\n"
    "#FAKE:writer"
)

_EXTRACT_SYSTEM = (
    "你是引用结论抽取器。从给定报告中抽取所有事实性结论：含数字、百分比、年份、比较（优于/低于/提升/下降）、"
    "作者明确结论的句子；纯过渡句、结构性描述、开放性问题不要抽。每条输出：claim_text（改写为可独立成立的陈述）、"
    "citation_keys（该结论句中出现的 [paper_id:page] 键原样含方括号）、evidence_ids（若证据映射中能对应则填，否则留空）、"
    "section（所属小节标题）、confidence（0-1）。\n"
    "#FAKE:claim_extract"
)


# --------------------------------------------------------------------------
# 内部工具
# --------------------------------------------------------------------------
def _fast_llm_like(llm: LLMClient | None) -> LLMClient:
    """结论抽取用 fast 角色；传入客户端是 fake 时自动同样用 fake（离线不烧 token）。"""
    if llm is not None and getattr(llm, "ptype", "") == "fake":
        return get_fake_llm("fast")
    return get_llm("fast")


def _keywords(brief: ResearchBrief) -> str:
    """从 objective 提取标题关键词（CJK 连续段 + ASCII 词，去虚词）。"""
    tokens: list[str] = []
    for run in re.findall(r"[一-鿿]{2,}", brief.objective or ""):
        run = run.strip("的了在与和及对从被是为把将地得着")
        if len(run) >= 2:
            tokens.append(run)
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", brief.objective or ""):
        if w.lower() not in {"survey", "research", "about", "the", "for", "with", "study"}:
            tokens.append(w)
    return "、".join(tokens[:3])


def _normalize_perspective(value: str | None, title: str) -> str:
    """把 LLM 输出的视角归一化到合法枚举；无法识别时按标题关键词推断。"""
    v = (value or "").strip().lower()
    if v in VALID_PERSPECTIVES:
        return v  # type: ignore[return-value]
    text = f"{title} {value or ''}".lower()
    if any(k in text for k in ("method", "方法", "框架", "算法", "技术路线", "架构")):
        return "method"
    if any(k in text for k in ("experiment", "实验", "评测", "结果", "对比", "比较", "benchmark")):
        return "experiment"
    if any(k in text for k in ("critique", "局限", "争议", "批评", "风险", "缺口", "批判")):
        return "critique"
    if any(k in text for k in ("application", "应用", "落地", "场景")):
        return "application"
    return "background"


def _attach_evidence(
    section: OutlineSection,
    evidence: Sequence[EvidenceRecord],
    task_perspective: dict[str, str],
) -> None:
    """按 任务匹配 > 视角匹配 > 文本相似度 给小节挑选最相关的证据 id。"""
    if not evidence:
        return
    target = f"{section.title} {' '.join(section.key_points)}"
    task_set = set(section.task_ids)

    def score(ev: EvidenceRecord) -> float:
        s = float(ev.relevance_score or 0.0)
        if ev.task_id and ev.task_id in task_set:
            s += 2.0
        elif section.perspective and task_perspective.get(ev.task_id) == section.perspective:
            s += 1.0
        s += _sim(target, f"{ev.claim_hint} {truncate(ev.evidence_text, 200)}")
        return s

    ranked = sorted(evidence, key=score, reverse=True)
    section.evidence_ids = [e.evidence_id for e in ranked[:6]]


def _template_section(
    perspective: str,
    brief: ResearchBrief,
    evidence: Sequence[EvidenceRecord],
    tasks: Sequence[ResearchTask],
) -> OutlineSection:
    """兜底模板小节（标题插入 objective 关键词）。"""
    kw = _keywords(brief)
    titles = {
        "background": f"研究背景与问题定义：{kw}",
        "method": f"主流方法与技术路线：{kw}",
        "experiment": f"实验设置与结果对比：{kw}",
        "application": "典型应用场景与案例",
        "critique": "争议、局限与未来方向",
    }
    points = {
        "background": ["研究问题的来源与重要性", "核心概念与术语界定", "与本综述范围的关系"],
        "method": ["各类方法的分类框架", "代表方法的核心思想", "方法之间的继承与改进关系"],
        "experiment": ["常用数据集与评测指标", "主要实验结果与横向对比", "实验设置的差异与可比性"],
        "application": ["典型应用场景", "落地效果与限制"],
        "critique": ["现有结论之间的冲突", "方法与实验的局限", "尚未解决的开放问题"],
    }
    sec = OutlineSection(
        title=titles.get(perspective, "补充分析"),
        key_points=list(points.get(perspective, [])),
        task_ids=[t.task_id for t in tasks if t.perspective == perspective][:3],
        perspective=perspective,  # type: ignore[arg-type]
    )
    task_map = {t.task_id: t.perspective for t in tasks}
    _attach_evidence(sec, evidence, task_map)
    return sec


def _fallback_outline(
    brief: ResearchBrief,
    tasks: Sequence[ResearchTask],
    evidence: Sequence[EvidenceRecord],
) -> list[OutlineSection]:
    """LLM 失败时的固定 5 节提纲：背景 / 方法 / 实验对比 / 争议与局限 / 结论与证据缺口。"""
    kw = _keywords(brief)
    specs: list[tuple[str, str, list[str]]] = [
        ("background", f"研究背景与问题定义：{kw}", ["研究动机与核心问题", "概念界定"]),
        ("method", f"主流方法与技术路线：{kw}", ["方法分类与代表工作"]),
        ("experiment", f"实验结果与横向对比：{kw}", ["数据集与指标", "结果对比与口径差异"]),
        ("critique", "争议、局限与批判性分析", ["结论冲突与可能解释", "方法局限"]),
        ("critique", "结论与证据缺口", ["当前证据支持的结论", "尚无定论的问题"]),
    ]
    task_map = {t.task_id: t.perspective for t in tasks}
    out: list[OutlineSection] = []
    for p, title, kps in specs:
        sec = OutlineSection(
            title=title,
            key_points=kps,
            task_ids=[t.task_id for t in tasks if t.perspective == p][:3],
            perspective=p,  # type: ignore[arg-type]
        )
        _attach_evidence(sec, evidence, task_map)
        out.append(sec)
    return out


def _reassign_index(sections: list[OutlineSection]) -> int | None:
    """找视角冗余且证据最少的小节下标（用于视角改派），找不到返回 None。"""
    counts: dict[str, int] = {}
    for s in sections:
        if s.perspective:
            counts[s.perspective] = counts.get(s.perspective, 0) + 1
    candidates = [i for i, s in enumerate(sections) if s.perspective and counts.get(s.perspective, 0) > 1]
    if not candidates:
        return None
    return min(candidates, key=lambda i: len(sections[i].evidence_ids))


def _clamp_and_cover(
    sections: list[OutlineSection],
    brief: ResearchBrief,
    evidence: Sequence[EvidenceRecord],
    tasks: Sequence[ResearchTask],
) -> list[OutlineSection]:
    """节数 clamp 到 4-8，并保证 background/method/experiment/critique 四个主要视角全覆盖。"""
    if len(sections) > MAX_SECTIONS:
        sections = sections[:MAX_SECTIONS]
    have = {s.perspective for s in sections if s.perspective}
    for p in MAJOR_PERSPECTIVES:
        if p in have:
            continue
        if len(sections) < MAX_SECTIONS:
            sections.append(_template_section(p, brief, evidence, tasks))
        else:
            idx = _reassign_index(sections)
            if idx is not None:
                sections[idx].perspective = p  # type: ignore[assignment]
    guard = 0
    while len(sections) < MIN_SECTIONS and guard < 8:
        guard += 1
        missing = [p for p in MAJOR_PERSPECTIVES if p not in {s.perspective for s in sections if s.perspective}]
        pick = missing[0] if missing else MAJOR_PERSPECTIVES[len(sections) % len(MAJOR_PERSPECTIVES)]
        sections.append(_template_section(pick, brief, evidence, tasks))
    return sections


# --------------------------------------------------------------------------
# generate_outline
# --------------------------------------------------------------------------
def generate_outline(
    brief: ResearchBrief,
    tasks: Sequence[ResearchTask],
    evidence: Sequence[EvidenceRecord],
    papers: Sequence[PaperRecord],
    *,
    gaps: Sequence[Gap] | None = None,
    llm: LLMClient | None = None,
) -> list[OutlineSection]:
    """根据 Brief、各任务证据覆盖与缺口生成 4-8 节提纲；LLM 失败退 5 节模板，绝不抛异常。"""
    tracer = get_active_tracer()
    llm = llm or get_llm("orchestrator")
    tasks = list(tasks or [])
    evidence = list(evidence or [])

    ev_count: dict[str, int] = {}
    for ev in evidence:
        ev_count[ev.task_id] = ev_count.get(ev.task_id, 0) + 1
    ordered = sorted(tasks, key=lambda t: ev_count.get(t.task_id, 0), reverse=True)  # 有证据的任务优先

    task_lines = "\n".join(
        f"- [{t.perspective}] {truncate(t.question, 120)}（证据 {ev_count.get(t.task_id, 0)} 条）"
        for t in ordered[:12]
    ) or "-（无子任务）"
    question_lines = "\n".join(f"- {q}" for q in (brief.core_questions or [])[:6]) or "-（无）"
    gap_lines = "\n".join(f"- {truncate(g.description, 120)}（严重度 {g.severity}）" for g in (gaps or [])[:8]) or "-（无）"
    paper_lines = "\n".join(
        f"- {p.paper_id}: {truncate(p.title, 80)}（{p.year or '年份未知'}）" for p in (papers or [])[:10]
    ) or "-（暂无论文）"

    prompt = (
        f"研究目标：{brief.objective}\n"
        f"范围外：{'；'.join(brief.out_of_scope) or '无'}\n"
        f"时间范围：{brief.time_range or '不限'}\n\n"
        f"核心问题：\n{question_lines}\n\n"
        f"子问题与证据覆盖（有证据的在前）：\n{task_lines}\n\n"
        f"已知证据缺口：\n{gap_lines}\n\n"
        f"已收集论文（前 10 篇）：\n{paper_lines}\n\n"
        f"请输出 4-8 节提纲（每节：title / key_points / task_ids / perspective）。"
    )

    sections: list[OutlineSection] = []
    try:
        draft = llm.chat_json(prompt, _OutlineDraft, system=_OUTLINE_SYSTEM, label="outline:generate")
        known_tasks = {t.task_id for t in tasks}
        task_map = {t.task_id: t.perspective for t in tasks}
        for sd in draft.sections:
            title = (sd.title or "").strip()
            if not title:
                continue
            sec = OutlineSection(
                title=title[:80],
                key_points=[truncate(k, 120) for k in sd.key_points if k][:5],
                task_ids=[tid for tid in sd.task_ids if tid in known_tasks],
                perspective=_normalize_perspective(sd.perspective, title),
            )
            _attach_evidence(sec, evidence, task_map)
            sections.append(sec)
        sections = _clamp_and_cover(sections, brief, evidence, tasks)
    except Exception as exc:  # noqa: BLE001 —— 提纲生成失败绝不中断流程
        tracer.event("outline_fallback", node="writer", error=truncate(str(exc), 200))
        sections = []

    if not sections:
        sections = _fallback_outline(brief, tasks, evidence)
    tracer.event("outline_ready", node="writer", n_sections=len(sections))
    return sections


# --------------------------------------------------------------------------
# 上下文装配（优先 context_builder，未实现/失败时内联兜底）
# --------------------------------------------------------------------------
def _matrix_to_md(m: ComparisonMatrix) -> str:
    cols = ["论文"] + list(m.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in m.rows:
        cells = [truncate(row.title or row.paper_id, 40)] + [row.values.get(c, "-") for c in m.columns]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    for w in (m.comparability_warnings or [])[:3]:
        lines.append(f"> 注意：{truncate(w, 120)}")
    return "\n".join(lines)


def _fallback_context(
    cards: Sequence[PaperCard],
    evidence: Sequence[EvidenceRecord],
    conflicts: Sequence[ConflictReport],
    comparisons: Sequence[ComparisonMatrix],
    gaps: Sequence[Gap],
    extra: str,
    max_chars: int = 9000,
) -> str:
    """内联兜底上下文：论文卡片 + 证据片段（带 evidence_id / 引用键）+ 冲突 + 对比表 + 缺口。"""
    parts: list[str] = [extra]
    for c in cards[:5]:
        bits = []
        if c.motivation:
            bits.append(f"动机={truncate(c.motivation, 120)}")
        if c.method:
            bits.append(f"方法={truncate(c.method, 160)}")
        if c.datasets:
            bits.append(f"数据={'/'.join(c.datasets[:3])}")
        if c.results:
            bits.append(f"结论={truncate(c.results, 200)}")
        if c.limitations:
            bits.append(f"局限={'；'.join(truncate(x, 80) for x in c.limitations[:3])}")
        parts.append(f"- [{c.paper_id}] " + "；".join(bits))
    for e in evidence[:12]:
        page = f":{e.page}" if e.page else ""
        parts.append(
            f"- 证据({e.evidence_id}) [{e.paper_id}{page}] {truncate(e.claim_hint, 120)}\n"
            f"  原文：{truncate(e.evidence_text, 400)}"
        )
    for c in conflicts[:4]:
        parts.append(f"- 冲突[{c.conflict_id}] {truncate(c.topic, 60)}：{truncate(c.description, 160)}"
                     f"（可能原因：{'；'.join(truncate(x, 60) for x in c.possible_causes[:2])}）")
    for m in comparisons[:2]:
        parts.append(_matrix_to_md(m))
    for g in gaps[:5]:
        parts.append(f"- 缺口：{truncate(g.description, 120)}")
    return truncate("\n".join(p for p in parts if p), max_chars, "")


def _assemble_section_prompt(
    brief: ResearchBrief,
    sec: OutlineSection,
    cards: Sequence[PaperCard],
    evidence: Sequence[EvidenceRecord],
    conflicts: Sequence[ConflictReport],
    comparisons: Sequence[ComparisonMatrix],
    gaps: Sequence[Gap],
    hints: Sequence[str],
) -> str:
    extra = (
        f"【本节标题】{sec.title}\n【本节要点】\n"
        + "\n".join(f"- {k}" for k in sec.key_points)
        + (f"\n【核验修复指令（必须逐条执行）】\n" + "\n".join(f"- {h}" for h in hints) if hints else "")
    )
    context = extra
    try:
        from ..memory.context_builder import build_context

        text = build_context(
            task=None,
            brief=brief,
            evidence=list(evidence),
            papers=[],
            cards=list(cards),
            conflicts=list(conflicts),
            extra=extra,
            token_budget=int(get_settings().budget("context_token_budget", 24000)),
        )
        if text and text.strip():
            context = text
    except Exception:  # NotImplementedError 或任何装配失败 → 内联兜底
        context = _fallback_context(cards, evidence, conflicts, comparisons, gaps, extra)

    # 无论哪条装配路径，都必须显式给出可原样复制的引用键（build_context 的压缩格式不含字面键）
    key_map = "\n".join(
        f"- [{e.paper_id}:{e.page}]（证据 {e.evidence_id}：{truncate(e.claim_hint, 80)}）"
        if e.page
        else f"- [{e.paper_id}]（证据 {e.evidence_id}：{truncate(e.claim_hint, 80)}）"
        for e in evidence[:12]
    )
    keys_block = (
        "\n\n【本节可用的内部引用键（事实性陈述必须原样复制这些键，禁止自造）】\n" + key_map if key_map else ""
    )
    tail = (
        "\n\n【对比矩阵（markdown table，可直接嵌入；数值不得改动）】\n"
        + "\n\n".join(_matrix_to_md(m) for m in comparisons)
        + "\n\n【证据缺口（本节需如实标注为“目前证据不足”）】\n"
        + "\n".join(f"- {truncate(g.description, 120)}" for g in gaps)
        if comparisons or gaps
        else ""
    )
    req = (
        "\n\n【写作要求】只依据上述证据与卡片写作；每个事实性陈述紧跟引用键 [paper_id:page]（从上方键清单原样复制）；"
        "冲突并列呈现；无证据处用不确定性表述；输出本节正文。"
    )
    return truncate(context + keys_block + tail + req, 14000, "")


# --------------------------------------------------------------------------
# write_report
# --------------------------------------------------------------------------
def _related_conflicts(sec: OutlineSection, conflicts: Sequence[ConflictReport]) -> list[ConflictReport]:
    if not conflicts:
        return []
    target = f"{sec.title} {' '.join(sec.key_points)}"
    hits = [c for c in conflicts if _sim(target, f"{c.topic} {c.description}") >= 0.25]
    return hits if hits else (list(conflicts) if len(conflicts) <= 3 else list(conflicts)[:2])


def _related_comparisons(sec: OutlineSection, comparisons: Sequence[ComparisonMatrix]) -> list[ComparisonMatrix]:
    if not comparisons:
        return []
    hits = [m for m in comparisons if m.task_id and m.task_id in sec.task_ids]
    if hits:
        return hits[:2]
    return list(comparisons)[:2] if len(comparisons) <= 2 else []


def _hint_targets_section(hint: str, sec: OutlineSection) -> bool:
    """修复指令形如“【小节名】指令”；无法解析目标时保守注入所有小节。"""
    m = re.match(r"【([^】]*)】", hint or "")
    if not m:
        return True
    target = m.group(1).strip()
    if not target or target in ("全文", "全部", "所有"):
        return True
    return target in sec.title or _sim(target, sec.title) >= 0.6


def _select_section_evidence(sec: OutlineSection, evidence: Sequence[EvidenceRecord]) -> list[EvidenceRecord]:
    """小节未预挂证据时，按标题/要点相似度 + 相关度选 top8。"""
    target = f"{sec.title} {' '.join(sec.key_points)}"

    def score(ev: EvidenceRecord) -> float:
        return float(ev.relevance_score or 0.0) + _sim(target, f"{ev.claim_hint} {truncate(ev.evidence_text, 200)}")

    return sorted(evidence, key=score, reverse=True)[:8]


def _fallback_section_markdown(
    sec: OutlineSection,
    cards: Sequence[PaperCard],
    evidence: Sequence[EvidenceRecord],
    conflicts: Sequence[ConflictReport],
    comparisons: Sequence[ComparisonMatrix],
) -> str:
    """单节兜底：结构化事实清单（每条带引用键），不虚构叙述。"""
    lines: list[str] = ["（本节为证据结构化清单，供后续修复轮重写）", ""]
    for c in cards[:4]:
        page_key = ""
        for e in evidence:
            if e.paper_id == c.paper_id:
                page_key = f" [{e.paper_id}:{e.page}]" if e.page else f" [{e.paper_id}]"
                break
        bits = []
        if c.method:
            bits.append(f"方法：{truncate(c.method, 150)}")
        if c.results:
            bits.append(f"结论：{truncate(c.results, 200)}")
        if c.limitations:
            bits.append(f"局限：{'；'.join(truncate(x, 80) for x in c.limitations[:2])}")
        if bits:
            lines.append(f"- **{truncate(c.paper_id, 40)}**{page_key} " + "；".join(bits))
    for e in evidence[:10]:
        page_key = f"[{e.paper_id}:{e.page}]" if e.page else f"[{e.paper_id}]"
        lines.append(f"- {truncate(e.claim_hint, 120)}——原文：“{truncate(e.evidence_text, 180)}” {page_key}（证据 {e.evidence_id}）")
    for c in conflicts[:3]:
        lines.append(f"- 冲突：{truncate(c.description, 160)}（可能原因：{'；'.join(c.possible_causes[:2])}）")
    for m in comparisons[:1]:
        lines.append("")
        lines.append(_matrix_to_md(m))
    lines.append("")
    lines.append("（当前证据不足以支撑更多叙述性结论。）")
    return "\n".join(lines)


def _normalize_keys(keys: Sequence[str]) -> list[str]:
    out: list[str] = []
    for k in keys or []:
        k = str(k).strip()
        if not k:
            continue
        if not k.startswith("["):
            k = f"[{k}]"
        if _KEY_IN_BRACKETS_RE.fullmatch(k) and k not in out:
            out.append(k)
    return out


# 证据提示的兜底/降级文案（Reader LLM 失败桩、VLM 图表解读提示）：不是真实结论，
# 不得进入结论清单——否则污染四层核验的状态分布，并诱发针对桩文本的修复任务。
_STUB_CLAIM_RE = re.compile(r"^(（LLM 失败|LLM 摘要失败|\[{0,1}VLM 解读|图表证据提示（|表格证据提示（|（本片段为)")


def _is_stub_claim(text: str) -> bool:
    return bool(_STUB_CLAIM_RE.match((text or "").strip()))


def _evidence_for_key(pid: str, page: int | None, evidence: Sequence[EvidenceRecord]) -> EvidenceRecord | None:
    """引用键 [pid:page] → 该论文该页（或最近页）的证据。"""
    cands = [e for e in evidence if e.paper_id == pid]
    if not cands:
        return None
    if page is None:
        return cands[0]
    return min(cands, key=lambda e: abs((e.page or 10**6) - page))


def _match_section(title: str | None, outline: Sequence[OutlineSection]) -> str | None:
    if not title:
        return None
    for s in outline:
        if s.title == title:
            return s.title
    for s in outline:
        if _sim(title, s.title) >= 0.6:
            return s.title
    return title


def _post_claims(
    draft: _ClaimsDraft,
    evidence: Sequence[EvidenceRecord],
    outline: Sequence[OutlineSection],
    max_claims: int,
) -> list[ClaimRecord]:
    """清洗抽取结论：键规范化、citation_keys 与 evidence_ids 互相补全、数量 clamp。"""
    out: list[ClaimRecord] = []
    for cd in draft.claims:
        text = (cd.claim_text or "").strip()
        if len(text) < 4 or _is_stub_claim(text):
            continue
        keys = _normalize_keys(cd.citation_keys)
        ev_ids = [e for e in cd.evidence_ids if any(x.evidence_id == e for x in evidence)]
        # 键 → 证据
        for k in keys:
            m = _KEY_IN_BRACKETS_RE.fullmatch(k)
            if not m:
                continue
            pid, page_s = m.group(1), m.group(2)
            ev = _evidence_for_key(pid, int(page_s) if page_s else None, evidence)
            if ev and ev.evidence_id not in ev_ids:
                ev_ids.append(ev.evidence_id)
        # 证据 → 键
        keyed_pids: set[str] = set()
        for k in keys:
            m = _KEY_IN_BRACKETS_RE.fullmatch(k)
            if m:
                keyed_pids.add(m.group(1))
        for eid in ev_ids:
            ev = next((x for x in evidence if x.evidence_id == eid), None)
            if ev and ev.paper_id not in keyed_pids:
                keys.append(f"[{ev.paper_id}:{ev.page}]" if ev.page else f"[{ev.paper_id}]")
                keyed_pids.add(ev.paper_id)
        out.append(
            ClaimRecord(
                claim_text=text[:500],
                citation_keys=keys,
                evidence_ids=ev_ids,
                section=_match_section(cd.section, outline),
                confidence=clamp(float(cd.confidence if cd.confidence is not None else 0.5), 0.0, 1.0),
                status="supported",  # 默认值，由 Verifier 改判
            )
        )
        if len(out) >= max_claims:
            break
    return out


def _section_of_position(markdown: str, pos: int, outline: Sequence[OutlineSection]) -> str | None:
    head = markdown[:pos]
    best, best_pos = None, -1
    for s in outline:
        p = head.rfind(s.title)
        if p > best_pos:
            best, best_pos = s.title, p
    return best


def _fallback_extract_claims(
    markdown: str,
    evidence: Sequence[EvidenceRecord],
    outline: Sequence[OutlineSection],
    max_claims: int,
) -> list[ClaimRecord]:
    """规则兜底抽取：按正文引用键出现顺序生成结论（claim_text 取证据提示）。"""
    claims: list[ClaimRecord] = []
    used: set[str] = set()
    for m in _CITE_KEY_RE.finditer(markdown or ""):
        pid, page_s = m.group(1), m.group(2)
        ev = _evidence_for_key(pid, int(page_s) if page_s else None, evidence)
        if ev is None or ev.evidence_id in used or _is_stub_claim(ev.claim_hint or ""):
            continue
        used.add(ev.evidence_id)
        claims.append(
            ClaimRecord(
                claim_text=truncate(ev.claim_hint or f"{pid} 的相关结论", 300),
                citation_keys=[f"[{pid}:{page_s}]" if page_s else f"[{pid}]"],
                evidence_ids=[ev.evidence_id],
                section=_section_of_position(markdown, m.start(), outline),
                confidence=clamp(float(ev.relevance_score or 0.5), 0.0, 1.0),
                status="supported",
            )
        )
        if len(claims) >= max_claims:
            break
    if not claims:  # 正文连引用键都没有 → 直接按证据清单生成
        for ev in evidence[:max_claims]:
            key = f"[{ev.paper_id}:{ev.page}]" if ev.page else f"[{ev.paper_id}]"
            claims.append(
                ClaimRecord(
                    claim_text=truncate(ev.claim_hint or f"{ev.paper_id} 的相关结论", 300),
                    citation_keys=[key],
                    evidence_ids=[ev.evidence_id],
                    section=next((s.title for s in outline if ev.evidence_id in s.evidence_ids), None),
                    confidence=clamp(float(ev.relevance_score or 0.5), 0.0, 1.0),
                    status="supported",
                )
            )
    return claims


def _strip_duplicated_header(content: str, title: str) -> str:
    """去掉模型在正文开头重复输出、且与节标题相同的标题行（含其后的空行）。"""
    norm_title = re.sub(r"\s+", "", (title or "")).lower()
    if not norm_title:
        return content
    lines = (content or "").split("\n")
    while lines:
        line = lines[0].strip()
        if not line:
            lines.pop(0)
            continue
        if line.startswith("#") and re.sub(r"\s+", "", line.lstrip("#").strip()).lower() == norm_title:
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip() or content


def write_report(
    brief: ResearchBrief,
    outline: Sequence[OutlineSection],
    evidence: Sequence[EvidenceRecord],
    papers: Sequence[PaperRecord],
    cards: Sequence[PaperCard],
    *,
    comparisons: Sequence[ComparisonMatrix] | None = None,
    conflicts: Sequence[ConflictReport] | None = None,
    gaps: Sequence[Gap] | None = None,
    llm: LLMClient | None = None,
    repair_hints: Sequence[str] | None = None,
    extractor_llm: LLMClient | None = None,
) -> ReportDraft:
    """逐节写作 + 独立抽取结论；任何 LLM 环节失败都退规则兜底，绝不抛异常。"""
    tracer = get_active_tracer()
    llm = llm or get_llm("writer")
    fast = extractor_llm or _fast_llm_like(llm)
    evidence = list(evidence or [])
    outline = list(outline or [])
    if not outline:
        outline = _fallback_outline(brief, [], evidence)
    comparisons = list(comparisons or [])
    conflicts = list(conflicts or [])
    gaps = list(gaps or [])
    repair_hints = list(repair_hints or [])
    max_claims = int(get_settings().budget("max_claims_per_report", 60))

    ev_map = {e.evidence_id: e for e in evidence}
    card_map = {c.paper_id: c for c in (cards or [])}

    sections_md: list[str] = []
    fallback_sections = 0
    for idx, sec in enumerate(outline):
        sec_evidence = [ev_map[i] for i in sec.evidence_ids if i in ev_map] or _select_section_evidence(sec, evidence)
        sec_cards: list[PaperCard] = []
        seen: set[str] = set()
        for e in sec_evidence:
            if e.paper_id in card_map and e.paper_id not in seen:
                seen.add(e.paper_id)
                sec_cards.append(card_map[e.paper_id])
        sec_cards = sec_cards[:5]
        sec_conflicts = _related_conflicts(sec, conflicts)
        sec_comparisons = _related_comparisons(sec, comparisons)
        hints = [h for h in repair_hints if _hint_targets_section(h, sec)]
        need_gaps = gaps and (sec.perspective == "critique" or any(k in sec.title for k in ("缺口", "局限", "争议", "结论")))
        prompt = _assemble_section_prompt(
            brief, sec, sec_cards, sec_evidence, sec_conflicts, sec_comparisons, gaps if need_gaps else [], hints
        )
        try:
            content = llm.chat(prompt=prompt, system=_WRITER_SYSTEM, label=f"writer:section:{idx + 1}")
            if not content or not content.strip():
                raise ValueError("空章节内容")
            content = content.strip()
        except Exception as exc:  # noqa: BLE001 —— 单节失败退兜底，其余节继续
            fallback_sections += 1
            tracer.event("section_fallback", node="writer", section=truncate(sec.title, 60), error=truncate(str(exc), 200))
            content = _fallback_section_markdown(sec, sec_cards, sec_evidence, sec_conflicts, sec_comparisons)
        sections_md.append(f"## {sec.title}\n\n{_strip_duplicated_header(content, sec.title)}")

    markdown = f"# {truncate(brief.objective, 80)}\n\n" + "\n\n".join(sections_md) + "\n"

    # ---- 独立一次结论抽取（fast 角色，与写作分离）----
    claims: list[ClaimRecord] = []
    try:
        ev_hint = "\n".join(f"- {e.evidence_id} → [{e.paper_id}:{e.page if e.page else '-'}]" for e in evidence[:40])
        extract_prompt = (
            f"报告全文：\n{truncate(markdown, 12000)}\n\n证据 id → 引用键映射（用于回填 evidence_ids）：\n{ev_hint}\n\n"
            f"请抽取全部事实性结论（最多 {max_claims} 条）。"
        )
        draft = fast.chat_json(extract_prompt, _ClaimsDraft, system=_EXTRACT_SYSTEM, label="writer:extract_claims")
        claims = _post_claims(draft, evidence, outline, max_claims)
    except Exception as exc:  # noqa: BLE001
        tracer.event("claims_extract_fallback", node="writer", error=truncate(str(exc), 200))
        claims = []
    if not claims:
        claims = _fallback_extract_claims(markdown, evidence, outline, max_claims)

    tracer.event(
        "report_written",
        node="writer",
        n_sections=len(sections_md),
        n_claims=len(claims),
        fallback_sections=fallback_sections,
    )
    return ReportDraft(markdown=markdown, claims=claims)


# --------------------------------------------------------------------------
# render_final
# --------------------------------------------------------------------------
def _fallback_references(known: list[PaperRecord], order: list[str]) -> str:
    lines: list[str] = []
    for p in known:
        n = order.index(p.paper_id) + 1
        if p.authors:
            authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        else:
            authors = "佚名"
        venue = f" {p.venue}," if p.venue else ""
        ident = p.doi or p.source_url or p.arxiv_id or ""
        lines.append(f"[{n}] {authors}. {p.title}.{venue} {p.year or 'n.d.'}. {ident}".rstrip())
    return "\n".join(lines)


def render_final(report_md: str, claims: Sequence[ClaimRecord], papers: Sequence[PaperRecord]) -> str:
    """把 [paper_id:page] / [paper_id] 渲染为编号引用 [n]（同一论文同号），并追加参考文献。

    只渲染正文中实际出现的引用键；未被引用的论文不进参考文献。
    """
    tracer = get_active_tracer()
    paper_map = {p.paper_id: p for p in (papers or [])}
    order: list[str] = []

    def _replace(m: re.Match) -> str:
        pid = m.group(1)
        if pid.isdigit():  # 已是编号引用 [1]，保持原样
            return m.group(0)
        if pid not in order:
            order.append(pid)
        n = order.index(pid) + 1
        return f"[{n}:{m.group(2)}]" if m.group(2) else f"[{n}]"  # 保留页码：[1:9] 可回到 PDF 第 9 页

    text = _CITE_KEY_RE.sub(_replace, report_md or "")
    if not order:  # 正文无键时用 claims 的键兜底编号（至少保证参考文献可生成）
        for c in claims or []:
            for k in _normalize_keys(c.citation_keys):
                m = _KEY_IN_BRACKETS_RE.fullmatch(k)
                if m and m.group(1) not in order:
                    order.append(m.group(1))

    known = [paper_map[pid] for pid in order if pid in paper_map]
    unknown = [pid for pid in order if pid not in paper_map]
    if unknown:
        tracer.event("render_unknown_citations", node="writer", pids=unknown[:10])
    try:
        from ..tools.citation_tool import render_references

        refs = render_references(known) if known else ""
        if not str(refs).strip():
            raise ValueError("空参考文献")
    except Exception:  # NotImplementedError（工具未实现）或渲染失败 → 内联兜底格式
        refs = _fallback_references(known, order)

    tracer.event("report_rendered", node="writer", n_refs=len(known))
    return text.rstrip() + "\n\n## 参考文献\n\n" + str(refs).strip() + "\n"
