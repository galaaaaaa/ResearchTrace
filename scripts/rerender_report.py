"""离线重渲染最终报告。

用途：引用渲染逻辑修复（如含冒号 paper_id 的键 → 编号 [n] + 参考文献）后，
无需重跑整条研究管线，直接用审计 JSON 里的 papers/claims 重渲染既有 run 的报告。

用法：
    python scripts/rerender_report.py outputs/audit/<run_id>.json [--force]

报告原文件会备份为 <report>.bak；--force 缺省时若已存在 .bak 则拒绝覆盖。
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.agents.writer import render_final  # noqa: E402
from src.schemas import ClaimRecord, PaperRecord  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    force = "--force" in args
    audit_path = pathlib.Path(args[0])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    report_path = pathlib.Path(audit.get("report_path") or "")
    if not report_path.exists():
        report_path = audit_path.parents[1] / "reports" / f"{audit['run_id']}.md"
    md = report_path.read_text(encoding="utf-8")

    # 拆出正文（参考文献之前）与审计摘要（## 引用审计摘要 起、## 证据缺口 止）
    body, sep, _ = md.partition("## 参考文献")
    if not sep:
        print(f"报告 {report_path} 中未找到 '## 参考文献'，视为非最终报告，中止。")
        return 1
    _, sep2, tail = md.partition("## 引用审计摘要")
    if sep2:
        summary, _, _ = tail.partition("## 证据缺口")
        summary = "## 引用审计摘要" + summary.rstrip()
    else:
        summary = ""

    claims = [ClaimRecord.model_validate(c) for c in audit.get("claims") or []]
    papers = [PaperRecord.model_validate(p) for p in audit.get("papers") or []]
    new_md = render_final(body.rstrip(), claims, papers) + ("\n" + summary.lstrip("\n") if summary else "")

    # 证据缺口披露：从 audit 的 gaps 重建（对 description/fix_question 施加去嵌套 + 截断，
    # 与新版 gap_analyzer 一致），替换旧报告里逐轮嵌套膨胀的文本
    gaps = audit.get("gaps") or []
    if gaps:
        from src.agents.gap_analyzer import _strip_nesting

        lines = [
            f"- [{g.get('severity')}] {_strip_nesting(g.get('description') or '', limit=90)}"
            + (f"（{_strip_nesting(g.get('fix_question') or '', limit=90)}）" if g.get("fix_question") else "")
            for g in gaps
        ]
        new_md += "\n\n## 证据缺口（未解决，明确标注）\n\n" + "\n".join(lines)
        new_md += f"\n\n<!-- run_id: {audit['run_id']} 重渲染于 {__import__('datetime').datetime.now().isoformat(timespec='seconds')} -->\n"

    backup = report_path.with_suffix(".md.bak")
    if backup.exists() and not force:
        print(f"备份 {backup} 已存在；如需覆盖请加 --force。")
        return 1
    backup.write_text(md, encoding="utf-8")
    report_path.write_text(new_md, encoding="utf-8")
    n_keys = new_md.count("[1]")  # 仅作输出提示
    print(f"重渲染完成: {report_path}（原报告备份于 {backup}；claims={len(claims)} papers={len(papers)}）")
    print(f"正文中 [1] 出现 {n_keys} 次；参考文献块：{new_md.split('## 参考文献', 1)[1][:200]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
