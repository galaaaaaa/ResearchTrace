"""把历史研究 run 的证据回填进 Milvus 向量知识库。

用法：
    .venv/bin/python scripts/backfill_kb.py            # 全部 outputs/audit/run_*.json
    .venv/bin/python scripts/backfill_kb.py --audit outputs/audit/run_xxx.json

- 按 run 逐个 upsert（evidence_id 为主键，重复执行幂等）；
- 向量走 embedding 缓存（data/indexes/embeddings.db），二次执行不再计费；
- 需 EMBEDDING_* 配置（.env）；未配置时每 run 计 0 条后退出。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory import vector_store  # noqa: E402
from src.settings import get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", default=None, help="只回填指定审计 JSON（默认全部）")
    args = ap.parse_args()

    settings = get_settings()
    if args.audit:
        files = [Path(args.audit)]
    else:
        files = sorted(settings.audit_dir.glob("run_*.json"))

    if not vector_store.kb_available():
        print("✗ 向量知识库不可用：检查 pymilvus 安装与 .env 的 EMBEDDING_* 配置")
        return 1

    total = 0
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"- 跳过 {f.name}: 读取失败 {e}")
            continue
        evs = d.get("evidence") or []
        titles = {p.get("paper_id"): p.get("title") or "" for p in d.get("papers") or []}
        n = vector_store.upsert_evidence(evs, run_id=str(d.get("run_id") or f.stem), paper_titles=titles)
        total += n
        print(f"- {f.stem}: 证据 {len(evs)} 条，入库 {n} 条（库内共 {vector_store.kb_count()}）")
    print(f"\n完成：共入库 {total} 条，知识库现存 {vector_store.kb_count()} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
