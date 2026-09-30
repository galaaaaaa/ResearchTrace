"""引用质量评测：citation_precision / 冲突率 / 平均置信度 / 关键点覆盖。

用法：
    python -m src.eval.citation_eval outputs/audit/x.json [--gold data/eval_sets/gold.json]

audit JSON 需含 claims 列表（ClaimRecord 导出结构：claim_id / claim_text / status / confidence）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:  # rapidfuzz 是声明依赖
    from rapidfuzz import fuzz

    def _match_score(a: str, b: str) -> float:
        return max(fuzz.ratio(a or "", b or ""), fuzz.partial_ratio(a or "", b or "")) / 100.0

except ImportError:  # pragma: no cover
    from difflib import SequenceMatcher

    def _match_score(a: str, b: str) -> float:
        return SequenceMatcher(None, a or "", b or "").ratio()


_KEY_POINT_THRESHOLD = 0.45  # gold 关键点与 supported 结论的模糊匹配阈值


def _claim_list(audit: dict) -> list[dict]:
    for key in ("claims", "claim_records", "claim_status"):
        value = audit.get(key)
        if isinstance(value, list):
            return [c for c in value if isinstance(c, dict)]
    return []


def evaluate(audit: dict, *, gold: dict | None = None) -> dict:
    """从审计 JSON 计算 citation_precision（supported/(supported+unsupported+metadata_error)）等指标。

    gold 可选 {"key_points": [str]}：每个关键点与 supported 结论模糊匹配（>= 0.45 视为覆盖）。
    """
    claims = _claim_list(audit or {})
    n = len(claims)
    counts = {"supported": 0, "conflicted": 0, "unsupported": 0, "metadata_error": 0}
    confidences: list[float] = []
    for c in claims:
        st = str(c.get("status") or "supported")
        counts[st] = counts.get(st, 0) + 1
        try:
            confidences.append(float(c.get("confidence", 0.5)))
        except (TypeError, ValueError):
            pass

    denom = counts["supported"] + counts["unsupported"] + counts["metadata_error"]
    result: dict[str, Any] = {
        "n_claims": n,
        "status_counts": counts,
        "citation_precision": round(counts["supported"] / denom, 4) if denom else 0.0,
        "conflicted_rate": round(counts["conflicted"] / n, 4) if n else 0.0,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
    }

    points = [str(p) for p in ((gold or {}).get("key_points") or [])]
    if points:
        supported_texts = [str(c.get("claim_text") or "") for c in claims if c.get("status") == "supported"]
        covered = 0
        for p in points:
            best = max((_match_score(p, t) for t in supported_texts), default=0.0)
            if best >= _KEY_POINT_THRESHOLD:
                covered += 1
        result.update(
            {
                "n_key_points": len(points),
                "key_points_covered": covered,
                "key_point_coverage": round(covered / len(points), 4),
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="引用质量评测（audit JSON → 指标）")
    parser.add_argument("audit", help="审计 JSON 路径（含 claims 状态）")
    parser.add_argument("--gold", default=None, help="可选 gold JSON 路径（含 key_points）")
    args = parser.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8")) if args.gold else None
    print(json.dumps(evaluate(audit, gold=gold), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
