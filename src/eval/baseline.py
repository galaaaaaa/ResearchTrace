"""消融对照组：单模型直接生成（无检索、无多智能体、无核验）。

用法：python -m src.eval.baseline "研究问题" [--out outputs/reports/baseline_xxx.md]
用于与完整系统对照（文档 §11：单模型直接生成 / 完整系统 的引用质量对比）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..llm import get_llm
from ..settings import get_settings
from ..utils import new_id, now_iso


def direct_generate(query: str, *, llm=None, max_tokens: int = 8192) -> str:
    """单模型直接生成：无 Evidence Store、无 Verifier，引用全凭模型自身。"""
    client = llm or get_llm("writer")
    system = (
        "你是一名科研助手。请针对用户的研究问题直接撰写一篇结构化调研报告（markdown），"
        "包括背景、主要方法、对比、争议与结论。允许你引用你知道的论文（题名+年份即可）。"
    )
    return client.chat(query, system=system, max_tokens=max_tokens, label="baseline:direct")


def main() -> int:
    parser = argparse.ArgumentParser(description="单模型直接生成 baseline")
    parser.add_argument("query")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_id = new_id("baseline")
    text = direct_generate(args.query)
    out = Path(args.out) if args.out else get_settings().reports_dir / f"{run_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + f"\n\n<!-- baseline direct generation {now_iso()} -->\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "out": str(out), "chars": len(text)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
