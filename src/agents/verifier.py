"""Citation Verifier：四层核验 + 修复路由（文档 §8）。

- L1 元数据核验：claims 涉及论文的 DOI/题名/撤稿（crossref_tool，未实现或网络失败 → unavailable，不判错）；
- L2 引用蕴含核验：claim + 证据原文 → 独立 Judge（entailed/partial/contradicted/unrelated）；
- L3 覆盖率核验：切句 + 规则判事实句，无引用键 → citation_missing（超 30 句可选 fast LLM 辅助二分类）；
- L4 一致性核验：Judge 一次性找矛盾结论组 → conflicted（并列呈现，不触发修复）；
- route_verification / build_repair_tasks / build_repair_hints：修复路由表。

任何单点失败（judge 超时、工具未实现）只降级记录 note，绝不抛异常出模块。
"""

from __future__ import annotations

import re
from typing import Sequence

from pydantic import BaseModel, Field

from ..llm import LLMClient, get_fake_llm, get_llm
from ..schemas import (
    ClaimRecord,
    ClaimStatus,
    EvidenceRecord,
    MetadataCheck,
    PaperRecord,
    ResearchTask,
    VerificationCheck,
    VerificationReport,
)
from ..settings import get_settings
from ..tracing import get_active_tracer
from ..utils import truncate

try:  # rapidfuzz 是声明依赖；极端环境下退化为标准库相似度
    from rapidfuzz import fuzz

    def _sim(a: str, b: str) -> float:
        return fuzz.ratio(a or "", b or "") / 100.0

except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a or "", b or "").ratio()


# judge 标签归一：模型常回中文/同义词/大小写与空白变体（deepseek-flash 实测回 "支持"）。
# 旧版 `v.label in _VALID_LABELS else "unrelated"` 会把这些正确判断静默误杀成 unsupported，
# 曾导致整个 run 7/7 结论全灭（理由明明写着"与结论完全一致"）。
_LABEL_ALIASES = {
    "entailed": "entailed", "entail": "entailed", "supports": "entailed", "supported": "entailed",
    "支撑": "entailed", "支持": "entailed", "蕴含": "entailed", "明确蕴含": "entailed", "被支撑": "entailed",
    "partial": "partial", "partially": "partial", "partiallysupported": "entailed",
    "部分支撑": "partial", "部分支持": "partial", "部分": "partial",
    "contradicted": "contradicted", "contradict": "contradicted", "contradiction": "contradicted",
    "矛盾": "contradicted", "相矛盾": "contradicted", "与结论矛盾": "contradicted",
    "unrelated": "unrelated", "irrelevant": "unrelated", "notrelated": "unrelated",
    "无关": "unrelated", "不相关": "unrelated", "与结论无关": "unrelated",
}


def _norm_label(raw: str) -> str:
    """judge 标签归一：容忍中英文/空白/标点变体；无法识别返回 "unresolved"（按待定处理，不误杀）。"""
    key = re.sub(r"[\s。．·,，;；:：\"']", "", str(raw or "").strip().lower())
    return _LABEL_ALIASES.get(key, "unresolved")
# 状态优先级：一旦判 metadata_error/unsupported，不因后续层降级覆盖
_PRECEDENCE: dict[str, int] = {"supported": 0, "conflicted": 1, "unsupported": 2, "metadata_error": 3}
_KEY_RE = re.compile(r"\[([^\[\]]+?)(?::(\d+))?\]")
# 正文内部引用键（与 writer 一致；兼容 DOI / sha256:<hash> 等含冒号 paper_id——页码固定为结尾 ":数字]"，
# 排除已渲染编号 [1] 与 markdown 链接）
_CITE_KEY_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._\-/@:]*?)(?::(\d+))?\](?!\()")
# 事实句特征：数字 / 百分比 / 年份（被数字覆盖）/ 比较词
_FACTUAL_RE = re.compile(
    r"\d|[％%]|优于|高于|低于|超过|更优|更差|提升|提高|下降|降低|翻倍|显著|超过|lower|better|worse|"
    r"improv|reduc|outperform|achiev|state-of-the-art|sota",
    re.IGNORECASE,
)


class _JudgeVerdict(BaseModel):
    label: str = "entailed"
    reason: str = ""
    missing_information: str | None = None


class _ConflictItem(BaseModel):
    claim_ids: list[str] = Field(default_factory=list)
    description: str = ""


class _ConsistencyVerdict(BaseModel):
    conflicts: list[_ConflictItem] = Field(default_factory=list)


class _CoverageItem(BaseModel):
    index: int = 0
    needs_citation: bool = False


class _CoverageVerdict(BaseModel):
    results: list[_CoverageItem] = Field(default_factory=list)


_JUDGE_SYSTEM = (
    "你是独立的引用核验 Judge。只依据给出的证据原文判断结论是否被支撑，输出 label：\n"
    "entailed（证据明确蕴含）/ partial（证据只支撑较弱版本或不完整）/ contradicted（证据与结论矛盾）/ "
    "unrelated（证据与结论无关）。\n禁止使用你自己的知识补全证据；证据未提及的信息即为不支持。\n"
    "#FAKE:judge"
)

_CONSISTENCY_SYSTEM = (
    "你是结论一致性检查器。给定编号结论列表（每行形如“1. [claim_id] 结论文本”），找出相互矛盾的一组结论"
    "（对同一问题给出相反判断）。只依据结论文本本身判断。输出 conflicts: [{claim_ids: [涉及的 claim_id], "
    "description: 中文矛盾描述}]；无冲突输出空列表。\n#FAKE:consistency"
)

_COVERAGE_SYSTEM = (
    "你是事实句分类器。判断每个编号句子是否为“需要引用支撑的事实性陈述”：含数字/百分比/年份/比较/作者结论"
    "的句子需要；纯过渡句、结构描述、开放性问题不需要。输出 {\"results\": [{\"index\": 序号, "
    "\"needs_citation\": true/false}]}。\n#FAKE:coverage"
)


def _fast_llm_like(llm: LLMClient | None) -> LLMClient:
    """覆盖率辅助分类用 fast 角色；judge 是 fake 时同样用 fake（离线不烧 token）。"""
    if llm is not None and getattr(llm, "ptype", "") == "fake":
        return get_fake_llm("fast")
    return get_llm("fast")


# --------------------------------------------------------------------------
# L3：切句与事实句识别
# --------------------------------------------------------------------------
def _split_sentences(report_md: str) -> list[str]:
    """切句：按 。！？.!? 与换行；跳过标题/表格/代码/已渲染参考文献区。"""
    text = report_md or ""
    if "## 参考文献" in text:
        text = text.split("## 参考文献")[0]
    out: list[str] = []
    for line in text.splitlines():
        t = line.strip()
        if len(t) < 6:
            continue
        # blockquote（writer 的证据边界/证据不足声明框）是元叙述，设计上不带引用键，
        # 参与覆盖率核验只会产生误报并空耗修复轮
        if t.startswith(("#", "|", ">", "```")) or re.match(r"^\[\d+\]", t):
            continue
        for piece in re.split(r"(?<=[。！？!?])|(?<=\.)(?=\s|$)", t):
            p = piece.strip().strip("*").rstrip("。！？.!?；;，, ")
            if len(p) >= 6:
                out.append(p)
    return out


def _is_factual(sentence: str) -> bool:
    return bool(_FACTUAL_RE.search(sentence))


def _has_citation_key(sentence: str) -> bool:
    """句中是否含内部引用键：[paperA:3] / [10.1000/a:5] / 渲染后的 [1:14]（编号:页码）。

    只有裸编号 [9]（指向参考文献列表本身、无页码）不算正文引用——
    旧版把渲染形态 [1:14] 也当裸编号排除，导致带引用的句子全被误报"缺引用"。
    """
    for m in _CITE_KEY_RE.finditer(sentence):
        if m.group(2):  # 带页码：未渲染 [paper:pg] 与已渲染 [n:pg] 都是正文引用
            return True
        if not m.group(1).isdigit():
            return True
    return False


# --------------------------------------------------------------------------
# 主入口：verify
# --------------------------------------------------------------------------
def verify(
    claims: Sequence[ClaimRecord],
    evidence: Sequence[EvidenceRecord],
    papers: Sequence[PaperRecord],
    report_md: str,
    *,
    llm: LLMClient | None = None,
    max_judge_calls: int | None = None,
    metadata_checks: dict[str, MetadataCheck] | None = None,
) -> VerificationReport:
    """四层核验。返回 VerificationReport 并把最终状态回写到各 ClaimRecord。

    metadata_checks：测试注入的 paper_id → MetadataCheck（默认懒导入 crossref_tool.verify_metadata）。
    """
    tracer = get_active_tracer()
    llm = llm or get_llm("judge")
    claims = list(claims or [])
    evidence = list(evidence or [])
    papers = list(papers or [])
    cap = max_judge_calls if max_judge_calls is not None else int(
        get_settings().budget("max_judge_calls_per_verification", 80)
    )

    ev_map = {e.evidence_id: e for e in evidence}
    paper_map = {p.paper_id: p for p in papers}
    claim_by_id = {c.claim_id: c for c in claims}
    status: dict[str, ClaimStatus] = {c.claim_id: (c.status or "supported") for c in claims}
    notes: dict[str, str] = {}
    checks: list[VerificationCheck] = []
    conflict_descs: list[str] = []

    def set_status(cid: str, new: str) -> None:
        old = status.get(cid, "supported")
        if _PRECEDENCE.get(new, 0) >= _PRECEDENCE.get(old, 0):
            status[cid] = new  # type: ignore[assignment]

    # ---- L1 元数据核验 ----
    cited: dict[str, set[str]] = {}
    for c in claims:
        pids = {ev_map[e].paper_id for e in c.evidence_ids if e in ev_map}
        for k in c.citation_keys or []:
            m = _KEY_RE.fullmatch(str(k).strip())
            if m:
                pids.add(m.group(1))
        for pid in pids:
            cited.setdefault(pid, set()).add(c.claim_id)
    for pid, cids in cited.items():
        if metadata_checks is not None and pid in metadata_checks:
            check = metadata_checks[pid]
        else:
            paper = paper_map.get(pid)
            if paper is None:
                continue
            try:
                from ..tools.crossref_tool import verify_metadata

                check = verify_metadata(paper)
            except Exception:  # NotImplementedError（未实现）或网络失败 → 无法判定
                check = MetadataCheck(note="unavailable")
        bad: list[str] = []
        if check is not None and check.doi_valid is False:
            bad.append("DOI 不存在")
        if check is not None and check.title_similarity is not None and check.title_similarity < 0.6:
            bad.append(f"题名相似度过低（{check.title_similarity:.2f}）")
        if check is not None and check.is_retracted:
            bad.append("论文已被撤稿")
        if bad:
            for cid in sorted(cids):
                set_status(cid, "metadata_error")
                notes[cid] = "；".join(bad)
                checks.append(
                    VerificationCheck(claim_id=cid, layer="metadata", label="metadata_error", reason="；".join(bad))
                )
    tracer.event("verify_layer", node="verifier", layer="metadata", n_papers=len(cited))

    # ---- L2 引用蕴含核验 ----
    judged = 0
    for c in claims:
        evs = [ev_map[e] for e in c.evidence_ids if e in ev_map]
        if not evs:
            set_status(c.claim_id, "unsupported")
            checks.append(
                VerificationCheck(claim_id=c.claim_id, layer="entailment", label="unrelated", reason="结论未绑定任何证据")
            )
            continue
        if judged >= cap:
            notes[c.claim_id] = "超出 judge 调用上限，本轮未核验（待定）"
            checks.append(
                VerificationCheck(claim_id=c.claim_id, layer="entailment", label="partial", reason="judge 调用上限，待定")
            )
            continue
        snippet = "\n".join(
            f"[{e.evidence_id}] [{e.paper_id}{f':{e.page}' if e.page else ''}] 原文：{truncate(e.evidence_text, 600)}"
            for e in evs[:5]
        )
        prompt = f"结论：{c.claim_text}\n\n证据原文：\n{snippet}\n\n判断该结论是否被上述证据支撑。"
        try:
            v = llm.chat_json(prompt, _JudgeVerdict, system=_JUDGE_SYSTEM, label=f"judge:{c.claim_id}")
            label = _norm_label(v.label)
            if label == "unresolved":
                # 未知标签按待定处理（partial 语义，不改状态）——误杀成 unsupported 的代价
                # 是整份报告结论全灭并触发无意义的修复轮
                judged += 1
                notes[c.claim_id] = f"judge 标签无法解析（原始值 {v.label!r}），按待定处理"
                checks.append(
                    VerificationCheck(
                        claim_id=c.claim_id, layer="entailment", label="partial",
                        reason=f"标签未识别：{truncate(v.label, 40)}；{truncate(v.reason, 200)}",
                    )
                )
                continue
        except Exception as exc:  # noqa: BLE001 —— judge 失败不崩整个 verify
            judged += 1
            notes[c.claim_id] = f"judge 调用失败，待定：{truncate(str(exc), 120)}"
            checks.append(
                VerificationCheck(claim_id=c.claim_id, layer="entailment", label="partial", reason="judge 调用失败，结论待定")
            )
            continue
        judged += 1
        if label == "entailed":
            checks.append(
                VerificationCheck(claim_id=c.claim_id, layer="entailment", label="entailed", reason=truncate(v.reason, 300))
            )
        elif label == "partial":
            notes[c.claim_id] = f"partial：{v.missing_information or v.reason}"
            checks.append(
                VerificationCheck(
                    claim_id=c.claim_id,
                    layer="entailment",
                    label="partial",
                    reason=truncate(v.reason, 300),
                    missing_information=v.missing_information,
                )
            )
        elif label == "contradicted":
            set_status(c.claim_id, "conflicted")
            checks.append(
                VerificationCheck(claim_id=c.claim_id, layer="entailment", label="contradicted", reason=truncate(v.reason, 300))
            )
        else:
            set_status(c.claim_id, "unsupported")
            checks.append(
                VerificationCheck(claim_id=c.claim_id, layer="entailment", label="unrelated", reason=truncate(v.reason, 300))
            )
    tracer.event("verify_layer", node="verifier", layer="entailment", judged=judged, total=len(claims))

    # ---- L3 覆盖率核验 ----
    sentences = _split_sentences(report_md)
    factual = {i for i, s in enumerate(sentences) if _is_factual(s)}
    if len(sentences) > 30:  # 长报告用 fast LLM 辅助二分类，失败退纯规则
        try:
            fast = _fast_llm_like(llm)
            listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences[:120]))
            verdict = fast.chat_json(f"句子列表：\n{listing}", _CoverageVerdict, system=_COVERAGE_SYSTEM, label="verifier:coverage")
            if not verdict.results:
                raise ValueError("coverage 辅助分类返回空结果，视为失败退规则")
            llm_set = {
                item.index - 1
                for item in verdict.results
                if 0 < item.index <= len(sentences) and item.needs_citation
            }
            # 长报告取"规则∩LLM"双确认：规则正则对含数字的范式定义句误报、
            # LLM 对概括/转述句误报（实测 deepseek 单侧均大量误标），交集才可信。
            # 旧版以规则为准、LLM 只增不减，曾把元叙述全算成缺引用 → 154 句虚高、修复轮空转。
            factual &= llm_set
        except Exception:  # noqa: BLE001 —— 辅助分类失败退纯规则
            pass
    coverage_missing = [sentences[i] for i in sorted(factual) if not _has_citation_key(sentences[i])]
    for s in coverage_missing[:50]:
        checks.append(VerificationCheck(layer="coverage", label="citation_missing", sentence=truncate(s, 200), reason="事实句缺少引用键"))
    tracer.event("verify_layer", node="verifier", layer="coverage", n_sentences=len(sentences), missing=len(coverage_missing))

    # ---- L4 一致性核验 ----
    if len(claims) >= 2:
        subset = claims[:40]
        listing = "\n".join(f"{i + 1}. [{c.claim_id}] {c.claim_text}" for i, c in enumerate(subset))
        try:
            verdict = llm.chat_json(
                f"结论列表：\n{listing}\n\n请找出相互矛盾的结论组。", _ConsistencyVerdict,
                system=_CONSISTENCY_SYSTEM, label="verifier:consistency",
            )
            for item in verdict.conflicts:
                ids = [cid for cid in item.claim_ids if cid in claim_by_id]
                if not ids:
                    continue
                for cid in ids:
                    set_status(cid, "conflicted")
                desc = truncate(item.description or "结论之间存在矛盾", 300)
                checks.append(VerificationCheck(layer="consistency", label="conflict", reason=desc))
                conflict_descs.append(f"{'、'.join(ids)}：{desc}")
        except Exception as exc:  # noqa: BLE001 —— 规则兜底：同文高相似但置信度差异大 → 疑似（仅记 note）
            tracer.event("consistency_fallback", node="verifier", error=truncate(str(exc), 200))
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    a, b = subset[i], subset[j]
                    if _sim(a.claim_text, b.claim_text) >= 0.8 and abs((a.confidence or 0.5) - (b.confidence or 0.5)) >= 0.3:
                        conflict_descs.append(f"疑似冲突（规则，未改判）：{a.claim_id} 与 {b.claim_id} 表述相近但置信度差异大")
    tracer.event("verify_layer", node="verifier", layer="consistency", n_conflicts=len(conflict_descs))

    # ---- 汇总与回写 ----
    unsupported_ids = [cid for cid, st in status.items() if st == "unsupported"]
    metadata_ids = [cid for cid, st in status.items() if st == "metadata_error"]
    n_conflicted = sum(1 for st in status.values() if st == "conflicted")
    n_supported = sum(1 for st in status.values() if st == "supported")
    passed = not metadata_ids and not unsupported_ids and not coverage_missing
    summary = (
        f"四层核验完成：{len(claims)} 条结论中 {n_supported} 条获证据支撑、{n_conflicted} 条存在冲突（并列呈现）、"
        f"{len(unsupported_ids)} 条无证据支撑、{len(metadata_ids)} 条元数据异常；覆盖率缺失 {len(coverage_missing)} 句；"
        f"{'通过' if passed else '未通过'}。"
    )
    for c in claims:
        c.status = status.get(c.claim_id, c.status)
        if c.claim_id in notes:
            c.verifier_note = notes[c.claim_id]

    report = VerificationReport(
        checks=checks,
        claim_status=status,  # type: ignore[arg-type]
        unsupported_claim_ids=unsupported_ids,
        metadata_error_claim_ids=metadata_ids,
        coverage_missing=coverage_missing,
        conflicts=conflict_descs,
        passed=passed,
        summary=summary,
    )
    tracer.event(
        "verification_done", node="verifier", n_claims=len(claims), n_checks=len(checks), passed=passed
    )
    return report


# --------------------------------------------------------------------------
# 修复路由（文档 §8 修复路由表）
# --------------------------------------------------------------------------
def route_verification(report: VerificationReport, *, repair_round: int, max_repair_rounds: int = 2) -> str:
    """有可修复问题（unsupported / metadata_error / citation_missing）且轮次未满 → repair；conflicted 不触发修复。"""
    repairable = bool(report.unsupported_claim_ids or report.metadata_error_claim_ids or report.coverage_missing)
    return "repair" if repairable and repair_round < max_repair_rounds else "finalize"


def build_repair_tasks(
    verification: VerificationReport,
    claims: Sequence[ClaimRecord],
    *,
    max_tasks: int = 3,
) -> list[ResearchTask]:
    """unsupported claims → 定向补检索任务；metadata_error / citation_missing 不生成任务（由 writer 修复）。"""
    claim_by_id = {c.claim_id: c for c in (claims or [])}
    tasks: list[ResearchTask] = []
    for cid in verification.unsupported_claim_ids:
        c = claim_by_id.get(cid)
        text = truncate((c.claim_text if c else "") or "（结论文本缺失）", 200)
        tasks.append(
            ResearchTask(
                question=f"为以下结论定向补充证据：{text}",
                perspective="critique",
                origin="repair",
                required_evidence=["直接实验证据"],
            )
        )
        if len(tasks) >= max_tasks:
            break
    return tasks


def build_repair_hints(verification: VerificationReport, claims: Sequence[ClaimRecord]) -> list[str]:
    """给 Writer 的修复指令（注入对应章节 prompt）。"""
    claim_by_id = {c.claim_id: c for c in (claims or [])}
    hints: list[str] = []

    def _section_of(cid: str) -> str:
        c = claim_by_id.get(cid)
        return (c.section if c and c.section else None) or "全文"

    for cid in verification.unsupported_claim_ids:
        c = claim_by_id.get(cid)
        hints.append(f"【{_section_of(cid)}】删除或降级该结论（无证据支撑）：{truncate(c.claim_text if c else '', 120)}")
    for chk in verification.checks:
        c = claim_by_id.get(chk.claim_id or "")
        if chk.layer == "entailment" and chk.label == "partial" and c is not None:
            hints.append(
                f"【{_section_of(chk.claim_id or '')}】该结论与证据部分匹配，缩小结论范围或补充限定条件："
                f"{truncate(c.claim_text, 120)}"
            )
        elif chk.layer == "metadata" and chk.label == "metadata_error" and c is not None:
            hints.append(
                f"【{_section_of(chk.claim_id or '')}】删除或降级该结论（元数据核验未通过：伪 DOI/撤稿/题名不符）："
                f"{truncate(c.claim_text, 120)}"
            )
    for s in verification.coverage_missing[:10]:
        hints.append(f"【全文】为句子在现有证据中重绑引用（禁止凭空补引用）：{truncate(s, 120)}")
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out[:20]
