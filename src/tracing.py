"""节点级 Trace：JSONL 追加写，线程安全。

用法：
    set_active_tracer(JsonlTracer(run_id, path))   # app.py 启动时
    get_active_tracer().event("node_end", node="writer", ...)  # 各节点内部
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .utils import now_iso


class NullTracer:
    """默认空 Trace，未启动追踪时不产生任何开销。"""

    def event(self, kind: str, *, node: str | None = None, **payload) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlTracer:
    """逐事件追加写入 data/traces/{run_id}.jsonl，支持回放与消融分析。"""

    def __init__(self, run_id: str, path: str | Path):
        self.run_id = run_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._n = 0
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now_iso(), "run_id": run_id, "kind": "trace_start"}, ensure_ascii=False) + "\n")

    def event(self, kind: str, *, node: str | None = None, **payload) -> None:
        record = {"ts": now_iso(), "run_id": self.run_id, "kind": kind, "seq": self._next_seq()}
        if node is not None:
            record["node"] = node
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _next_seq(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def close(self) -> None:
        self.event("trace_end")


_active_lock = threading.Lock()
_active: JsonlTracer | NullTracer | None = None


def set_active_tracer(tracer: JsonlTracer | NullTracer | None) -> None:
    global _active
    with _active_lock:
        _active = tracer


def get_active_tracer() -> JsonlTracer | NullTracer:
    global _active
    with _active_lock:
        if _active is None:
            _active = NullTracer()
        return _active
