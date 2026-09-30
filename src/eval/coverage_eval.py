"""报告覆盖评测：关键点覆盖（关键词命中 + 可选 fast LLM 判定）与 must_cite 论文出现率。

用法：
    python -m src.eval.coverage_eval outputs/reports/x.md --gold data/eval_sets/gold.json [--use-llm]

gold JSON：{"key_points": [...], "must_cite_papers": [paper_id, ...]}。
默认纯关键词匹配（离线、不烧 token）；--use-llm 时用 fast 角色辅助判定关键点覆盖，失败自动退关键词。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from ..llm import LLMClient, get_fake_llm, get_llm

_EN_STOP = frozenset(
    {"the", "and", "for", "with", "from", "that", "this", "which", "into", "onto", "over", "under", "than", "then"}
)
_CJK_RUN_RE = re.compile(r"[一-鿿]{2,}")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_TRIM_CHARS = "的了在与和及对从被是为把将地得着"


def _keywords(point: str) -> list[str]:
    """关键点 → 关键词集合：CJK 连续段（去虚词）+ 长度 >=4 的英文词。"""
    kws: list[str] = []
    for run in _CJK_RUN_RE.findall(point or ""):
        run = run.strip(_TRIM_CHARS)
        if len(run) >= 2:
            kws.append(run)
    for w in _ASCII_WORD_RE.findall(point or ""):
        if w.lower() not in _EN_STOP:
            kws.append(w.lower())
    return kws


def _hit_rate(point: str, report_lower: str) -> float:
    kws = _keywords(point)
    if not kws:
        return 1.0  # 无可提取关键词，不计入惩罚
    hits = sum(1 for k in kws if k in report_lower)
    return hits / len(kws)


def _llm_keypoint_judge(llm: LLMClient, point: str, report_md: str) -> bool | None:
    """fast LLM 判定单个关键点是否被报告覆盖；失败返回 None（退关键词）。"""
    try:
        from pydantic import BaseModel

        class PointVerdict(BaseModel):
            covered: bool = False
            reason: str = ""

        prompt = (
            f"报告（节选）：\n{report_md[:6000]}\n\n关键点：{point}\n\n判断报告是否实质性覆盖该关键点"
            "（有对应论述或证据，而非仅出现字面词）。只输出 JSON。"
        )
        verdict = llm.chat_json(prompt, PointVerdict, system="你是覆盖判定器，只输出 JSON。\n#FAKE:coverage_eval")
        return bool(verdict.covered)
    except Exception:  # noqa: BLE001 —— LLM 失败退关键词
        return None


def evaluate(report_md: str, gold: dict, *, use_llm: bool = False) -> dict:
    """关键点覆盖率（fractional 命中率，可选 LLM 复核）+ must_cite 论文出现率。"""
    report_md = report_md or ""
    report_lower = report_md.lower()
    llm: LLMClient | None = None
    if use_llm:
        llm = get_fake_llm("fast") if get_llm("fast").ptype == "fake" else get_llm("fast")

    points = [str(p) for p in ((gold or {}).get("key_points") or [])]
    rates: list[float] = []
    covered_points: list[str] = []
    missing_points: list[str] = []
    for p in points:
        rate = _hit_rate(p, report_lower)
        covered = rate >= 0.5
        if llm is not None and rate < 0.5:
            judged = _llm_keypoint_judge(llm, p, report_md)  # 仅对关键词未命中的点做 LLM 复核
            if judged:
                covered = True
                rate = max(rate, 0.5)
        (covered_points if covered else missing_points).append(p)
        rates.append(rate)

    must = [str(p) for p in ((gold or {}).get("must_cite_papers") or [])]
    must_hit = [p for p in must if p.lower() in report_lower]
    must_missing = [p for p in must if p.lower() not in report_lower]

    result: dict[str, Any] = {
        "n_key_points": len(points),
        "key_point_coverage": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "covered_key_points": covered_points,
        "missing_key_points": missing_points,
        "n_must_cite": len(must),
        "must_cite_coverage": round(len(must_hit) / len(must), 4) if must else 0.0,
        "missing_must_cite": must_missing,
        "n_chars_report": len(report_md),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="报告覆盖评测（report markdown + gold JSON → 指标）")
    parser.add_argument("report", help="报告 markdown 路径")
    parser.add_argument("--gold", required=True, help="gold JSON 路径（key_points / must_cite_papers）")
    parser.add_argument("--use-llm", action="store_true", help="用 fast LLM 复核关键词未命中的关键点（烧 token）")
    args = parser.parse_args(argv)

    report_md = Path(args.report).read_text(encoding="utf-8")
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    print(json.dumps(evaluate(report_md, gold, use_llm=args.use_llm), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
