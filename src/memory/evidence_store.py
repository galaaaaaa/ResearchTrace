"""证据记忆：SQLite 持久化（线程安全，支持并行 Researcher 同时写入）。

设计要点：
- 单连接 + ``check_same_thread=False`` + ``threading.RLock`` 串行化全部访问；
- ``PRAGMA journal_mode=WAL``，跨进程读写不互相阻塞（进程内仍由锁串行）；
- payload 一律存 pydantic ``model_dump_json()``，读取用 ``model_validate_json``；
- upsert 采用 INSERT OR REPLACE；evidence 重复写入时保留首次 created_at；
- 运行期错误（除 ``__init__`` 建库失败允许抛出外）经 tracer 记录后按空结果降级，
  不向调用方抛异常。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from src.schemas import ClaimRecord, EvidenceRecord, Finding, PaperCard, PaperRecord
from src.tracing import get_active_tracer
from src.utils import norm_doi, norm_title, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id   TEXT PRIMARY KEY,
    doi        TEXT,
    arxiv_id   TEXT,
    title_norm TEXT,
    payload    TEXT
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    paper_id    TEXT,
    task_id     TEXT,
    payload     TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence(paper_id);
CREATE INDEX IF NOT EXISTS idx_evidence_task  ON evidence(task_id);
CREATE TABLE IF NOT EXISTS cards (
    card_id  TEXT PRIMARY KEY,
    paper_id TEXT UNIQUE,
    payload  TEXT
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    payload  TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    rowid      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT,
    payload    TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_task ON findings(task_id);
"""


def _norm_arxiv(arxiv_id: str | None) -> str | None:
    """arXiv ID 规范化：去空白、去 ``arXiv:`` 前缀、小写。"""
    if not arxiv_id:
        return None
    s = arxiv_id.strip().lower()
    if s.startswith("arxiv:"):
        s = s[len("arxiv:"):]
    return s or None


class EvidenceStore:
    """分层记忆中的证据记忆层：papers / evidence / cards / claims / findings 五张表。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if self.db_path.name != ":memory:":
            parent = self.db_path.parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # 建库失败允许抛出：没有存储层，上层无法工作
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---- 内部工具 --------------------------------------------------------
    def _warn(self, op: str, ex: Exception) -> None:
        get_active_tracer().event("store_error", node="evidence_store", op=op, error=str(ex))

    def _rollback(self) -> None:
        try:
            self._conn.rollback()
        except sqlite3.Error:
            pass

    # ---- papers ----------------------------------------------------------
    def upsert_paper(self, paper: PaperRecord) -> None:
        self.upsert_papers([paper])

    def upsert_papers(self, papers: list[PaperRecord]) -> None:
        """批量写入论文；doi/arxiv_id/title_norm 存规范化键，payload 存完整记录。"""
        if not papers:
            return
        with self._lock:
            try:
                rows = [
                    (
                        p.paper_id,
                        norm_doi(p.doi),
                        _norm_arxiv(p.arxiv_id),
                        norm_title(p.title),
                        p.model_dump_json(),
                    )
                    for p in papers
                ]
                self._conn.executemany(
                    "INSERT OR REPLACE INTO papers(paper_id, doi, arxiv_id, title_norm, payload) VALUES (?,?,?,?,?)",
                    rows,
                )
                self._conn.commit()
            except sqlite3.Error as ex:
                self._rollback()
                self._warn("upsert_papers", ex)

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT payload FROM papers WHERE paper_id = ?", (paper_id,)
                ).fetchone()
            except sqlite3.Error as ex:
                self._warn("get_paper", ex)
                return None
        return PaperRecord.model_validate_json(row[0]) if row else None

    def find_paper(
        self,
        *,
        doi: str | None = None,
        arxiv_id: str | None = None,
        title_norm: str | None = None,
    ) -> PaperRecord | None:
        """按任意非空条件 AND 精确匹配查找论文；doi/arxiv_id 先规范化。

        无任何非空条件时返回 None（无法唯一识别）。
        """
        conds: list[str] = []
        params: list[str] = []
        nd = norm_doi(doi)
        if nd:
            conds.append("doi = ?")
            params.append(nd)
        na = _norm_arxiv(arxiv_id)
        if na:
            conds.append("arxiv_id = ?")
            params.append(na)
        if title_norm:
            conds.append("title_norm = ?")
            params.append(norm_title(title_norm))  # 幂等：重复规范化结果不变
        if not conds:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT payload FROM papers WHERE {' AND '.join(conds)} ORDER BY rowid LIMIT 1",
                    params,
                ).fetchone()
            except sqlite3.Error as ex:
                self._warn("find_paper", ex)
                return None
        return PaperRecord.model_validate_json(row[0]) if row else None

    def all_papers(self) -> list[PaperRecord]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT payload FROM papers ORDER BY rowid").fetchall()
            except sqlite3.Error as ex:
                self._warn("all_papers", ex)
                return []
        return [PaperRecord.model_validate_json(r[0]) for r in rows]

    # ---- evidence --------------------------------------------------------
    def upsert_evidence(self, ev: EvidenceRecord) -> None:
        self.upsert_evidence_many([ev])

    def upsert_evidence_many(self, evs: list[EvidenceRecord]) -> None:
        """批量写入证据；同一 evidence_id 重复写入时保留首次 created_at。"""
        if not evs:
            return
        with self._lock:
            try:
                ids = [e.evidence_id for e in evs]
                placeholders = ",".join("?" * len(ids))
                old_created: dict[str, str] = {}
                for eid, payload in self._conn.execute(
                    f"SELECT evidence_id, payload FROM evidence WHERE evidence_id IN ({placeholders})", ids
                ).fetchall():
                    try:
                        created = (json.loads(payload) or {}).get("created_at")
                    except (json.JSONDecodeError, TypeError):
                        created = None
                    if created:
                        old_created[eid] = created
                rows = []
                for e in evs:
                    kept = old_created.get(e.evidence_id)
                    record = e.model_copy(update={"created_at": kept}) if kept and e.created_at != kept else e
                    rows.append((e.evidence_id, e.paper_id, e.task_id, record.model_dump_json()))
                self._conn.executemany(
                    "INSERT OR REPLACE INTO evidence(evidence_id, paper_id, task_id, payload) VALUES (?,?,?,?)",
                    rows,
                )
                self._conn.commit()
            except sqlite3.Error as ex:
                self._rollback()
                self._warn("upsert_evidence_many", ex)

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT payload FROM evidence WHERE evidence_id = ?", (evidence_id,)
                ).fetchone()
            except sqlite3.Error as ex:
                self._warn("get_evidence", ex)
                return None
        return EvidenceRecord.model_validate_json(row[0]) if row else None

    def evidence_by_ids(self, ids: list[str]) -> list[EvidenceRecord]:
        """按传入顺序返回存在的证据；缺失的 id 静默跳过。"""
        if not ids:
            return []
        with self._lock:
            try:
                placeholders = ",".join("?" * len(ids))
                rows = self._conn.execute(
                    f"SELECT evidence_id, payload FROM evidence WHERE evidence_id IN ({placeholders})", ids
                ).fetchall()
            except sqlite3.Error as ex:
                self._warn("evidence_by_ids", ex)
                return []
        found = {eid: EvidenceRecord.model_validate_json(payload) for eid, payload in rows}
        return [found[i] for i in ids if i in found]

    def evidence_for_task(self, task_id: str) -> list[EvidenceRecord]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT payload FROM evidence WHERE task_id = ? ORDER BY rowid", (task_id,)
                ).fetchall()
            except sqlite3.Error as ex:
                self._warn("evidence_for_task", ex)
                return []
        return [EvidenceRecord.model_validate_json(r[0]) for r in rows]

    def evidence_for_paper(self, paper_id: str) -> list[EvidenceRecord]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT payload FROM evidence WHERE paper_id = ? ORDER BY rowid", (paper_id,)
                ).fetchall()
            except sqlite3.Error as ex:
                self._warn("evidence_for_paper", ex)
                return []
        return [EvidenceRecord.model_validate_json(r[0]) for r in rows]

    def all_evidence(self, limit: int | None = None) -> list[EvidenceRecord]:
        with self._lock:
            try:
                if limit is None:
                    rows = self._conn.execute("SELECT payload FROM evidence ORDER BY rowid").fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT payload FROM evidence ORDER BY rowid LIMIT ?", (int(limit),)
                    ).fetchall()
            except sqlite3.Error as ex:
                self._warn("all_evidence", ex)
                return []
        return [EvidenceRecord.model_validate_json(r[0]) for r in rows]

    # ---- cards -----------------------------------------------------------
    def upsert_card(self, card: PaperCard) -> None:
        self.upsert_cards([card])

    def upsert_cards(self, cards: list[PaperCard]) -> None:
        """写入论文卡片；paper_id 唯一，同 paper_id 的旧卡片（不同 card_id）先删除。"""
        if not cards:
            return
        with self._lock:
            try:
                for c in cards:
                    self._conn.execute(
                        "DELETE FROM cards WHERE paper_id = ? AND card_id <> ?", (c.paper_id, c.card_id)
                    )
                    self._conn.execute(
                        "INSERT OR REPLACE INTO cards(card_id, paper_id, payload) VALUES (?,?,?)",
                        (c.card_id, c.paper_id, c.model_dump_json()),
                    )
                self._conn.commit()
            except sqlite3.Error as ex:
                self._rollback()
                self._warn("upsert_cards", ex)

    def all_cards(self) -> list[PaperCard]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT payload FROM cards ORDER BY rowid").fetchall()
            except sqlite3.Error as ex:
                self._warn("all_cards", ex)
                return []
        return [PaperCard.model_validate_json(r[0]) for r in rows]

    def card_for_paper(self, paper_id: str) -> PaperCard | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT payload FROM cards WHERE paper_id = ?", (paper_id,)
                ).fetchone()
            except sqlite3.Error as ex:
                self._warn("card_for_paper", ex)
                return None
        return PaperCard.model_validate_json(row[0]) if row else None

    # ---- claims ----------------------------------------------------------
    def upsert_claim(self, claim: ClaimRecord) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO claims(claim_id, payload) VALUES (?,?)",
                    (claim.claim_id, claim.model_dump_json()),
                )
                self._conn.commit()
            except sqlite3.Error as ex:
                self._rollback()
                self._warn("upsert_claim", ex)

    def all_claims(self) -> list[ClaimRecord]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT payload FROM claims ORDER BY rowid").fetchall()
            except sqlite3.Error as ex:
                self._warn("all_claims", ex)
                return []
        return [ClaimRecord.model_validate_json(r[0]) for r in rows]

    # ---- findings --------------------------------------------------------
    def add_finding(self, finding: Finding) -> None:
        """追加写入子 Agent 返回的压缩 Finding（append-only，不覆盖）。"""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO findings(task_id, payload, created_at) VALUES (?,?,?)",
                    (finding.task_id, finding.model_dump_json(), now_iso()),
                )
                self._conn.commit()
            except sqlite3.Error as ex:
                self._rollback()
                self._warn("add_finding", ex)

    def findings_for(self, task_id: str) -> list[Finding]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT payload FROM findings WHERE task_id = ? ORDER BY rowid", (task_id,)
                ).fetchall()
            except sqlite3.Error as ex:
                self._warn("findings_for", ex)
                return []
        return [Finding.model_validate_json(r[0]) for r in rows]

    # ---- 统计与生命周期 ---------------------------------------------------
    def stats(self) -> dict:
        out = {"papers": 0, "evidence": 0, "cards": 0, "claims": 0, "findings": 0}
        with self._lock:
            for key, table in (
                ("papers", "papers"),
                ("evidence", "evidence"),
                ("cards", "cards"),
                ("claims", "claims"),
                ("findings", "findings"),
            ):
                try:
                    out[key] = int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except sqlite3.Error as ex:
                    self._warn(f"stats:{table}", ex)
        return out

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
