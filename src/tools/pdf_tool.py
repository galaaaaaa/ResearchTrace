"""PDF 解析与检索：PyMuPDF 本地解析 + 手写 BM25 索引（零网络、零 LLM）。

- parse_pdf：主字号/编号/常见标题词启发式识别章节 → 滑窗分块（sources.pdf 参数）→ 扫描件检测；
- BM25Index：Okapi BM25（中英混合分词），query 返回 [(Chunk, score)] 降序；
- retrieve_chunks：路径 → 解析（{(abspath, mtime): PaperDoc} 模块级缓存）→ BM25 检索；
- extract_title：首页最大字号行启发式提取题名。

所有入口不抛异常：失败降级返回空结果 / None，并写 tracer 事件。
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.schemas import PaperDocInfo
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import hash_file, truncate


# --------------------------------------------------------------------------
# 数据结构（契约由桩固定，签名勿改）
# --------------------------------------------------------------------------
@dataclass
class Section:
    title: str
    page_from: int
    page_to: int


@dataclass
class Chunk:
    chunk_id: str
    page: int
    section: str
    text: str


@dataclass
class PaperDoc:
    pdf_path: str
    paper_id: str | None = None
    n_pages: int = 0
    is_scanned: bool = False
    sha256: str = ""
    sections: list[Section] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    def page_text(self, page: int) -> str:
        """重新打开 PDF 提取指定页文本（1-based）；越界/失败返回空串。"""
        try:
            with pymupdf.open(self.pdf_path) as pdf:
                if not (1 <= page <= pdf.page_count):
                    return ""
                return pdf[page - 1].get_text("text")
        except Exception:
            return ""

    @property
    def info(self) -> PaperDocInfo:
        """结构化索引摘要（正文细节仍在本对象上），供 state / 日志 / Agent 使用。"""
        return PaperDocInfo(
            pdf_path=self.pdf_path,
            paper_id=self.paper_id,
            n_pages=self.n_pages,
            n_chunks=len(self.chunks),
            is_scanned=self.is_scanned,
            sections=[s.title for s in self.sections],
            sha256=self.sha256 or None,
        )


# --------------------------------------------------------------------------
# 章节识别启发式
# --------------------------------------------------------------------------
_HEADING_MAX_LINE = 80      # 标题候选行长度上限（字符）
_HEADING_MAX_CHARS = 100    # 合并后的标题长度上限
_SIZE_DELTA = 1.2           # 高于主字号多少视为标题
_NUMBER_SIZE_TOL = 0.3      # 编号标题允许低于主字号的容差（过滤脚注）
_MERGE_SIZE_TOL = 1.0       # 相邻标题行字号差在该值内才合并（避免标题与首节标题粘连）
_TITLE_SIZE_TOL = 0.5       # 题名行之间的字号抖动容差
_NUMBER_PREFIX_RE = re.compile(r"^\d+(\.\d+){0,3}")
_ALPHA_RE = re.compile(r"[A-Za-z一-鿿]")
_KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "preliminaries",
    "method", "methods", "methodology", "experiments", "experimental results",
    "results", "discussion", "conclusion", "conclusions", "references",
}


def _extract_page(page_dict: dict) -> tuple[list[tuple[str, float]], list[str]]:
    """从 get_text("dict") 结果提取 (行文本, 行最大字号) 列表与段落列表。"""
    lines: list[tuple[str, float]] = []
    paras: list[str] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 只处理文本块
            continue
        block_lines: list[str] = []
        for line in block.get("lines", []):
            spans = [sp for sp in line.get("spans", []) if sp.get("text")]
            if not spans:
                continue
            text = "".join(sp["text"] for sp in spans).strip()
            if not text:
                continue
            size = max(float(sp.get("size", 0.0)) for sp in spans)
            lines.append((text, size))
            block_lines.append(text)
        if block_lines:
            block_text = "\n".join(block_lines)
            paras.extend(p.strip() for p in block_text.split("\n\n") if p.strip())
    return lines, paras


def _is_heading_line(text: str, size: float, main_size: float) -> bool:
    """标题候选：大字号短行 / 编号行 / 常见标题词（大小写不敏感）。"""
    t = text.strip()
    if not t or len(t) > _HEADING_MAX_LINE:
        return False
    if "arxiv:" in t.lower():  # 页眉水印等
        return False
    # 纯数字/符号行（页码、年份）与表格大字号数字标注（"26K"、"17K"）不是标题
    if len(_ALPHA_RE.findall(t)) < 2:
        return False
    low = t.lower().rstrip(":.;,、。 ")
    if low in _KNOWN_HEADINGS or low.startswith("acknowledg"):
        return True
    if main_size and size >= main_size + _SIZE_DELTA:
        return True
    m = _NUMBER_PREFIX_RE.match(t)
    if m and (not main_size or size >= main_size - _NUMBER_SIZE_TOL):
        # 脚注/参考文献条目常用小字号（如 "3That is ..."），字号明显偏小的不算编号标题
        rest = t[m.end():].lstrip(" .、）)")
        # 编号后接大写字母/CJK 才算标题，排除 "2 agents are ..." 之类正文行
        if rest and (rest[0].isupper() or "一" <= rest[0] <= "鿿"):
            return True
    return False


def _detect_headings(pages_lines: list[list[tuple[str, float]]], main_size: float) -> list[tuple[int, str]]:
    """逐页识别标题行并合并相邻同字号标题（跨行标题），返回 [(page_no, title)]。

    进入 References 之后只接受 appendix/supplementary 类标题，
    避免参考文献条目（"1901. Curran Associates, ..."）被误认为章节。
    """
    headings: list[tuple[int, str]] = []
    refs_seen = False
    for pno, lines in enumerate(pages_lines, start=1):
        run: list[str] = []
        run_size = 0.0

        def close_run() -> None:
            nonlocal run, run_size, refs_seen
            if run:
                title = re.sub(r"\s+", " ", " ".join(run)).strip()
                low = title.lower()
                if title and len(title) <= _HEADING_MAX_CHARS and not (
                    refs_seen and not any(k in low for k in ("appendix", "appendices", "supplementary"))
                ):
                    headings.append((pno, title))
                    if low.rstrip(":. ") == "references":
                        refs_seen = True
            run, run_size = [], 0.0

        for text, size in lines:
            if _is_heading_line(text, size, main_size):
                if run and abs(size - run_size) > _MERGE_SIZE_TOL:
                    close_run()  # 字号突变的相邻标题分属两节（如论文题名与首节标题）
                run.append(text)
                run_size = size
            elif run:
                close_run()
        close_run()
    return headings


def _section_title_at(sections: list[Section], page: int) -> str:
    """页 → 该页当时所在的章节标题（最后一个 page_from <= page 的章节）。"""
    title = "fulltext"
    for sec in sections:
        if sec.page_from <= page:
            title = sec.title
        else:
            break
    return title


# --------------------------------------------------------------------------
# 分块
# --------------------------------------------------------------------------
def _split_long_paragraph(para: str, chunk_chars: int) -> list[str]:
    """超长段落（PDF 块常把多个视觉段落并为一块）按行/词边界硬切到 chunk_chars 以内。"""
    if len(para) <= chunk_chars:
        return [para]
    pieces: list[str] = []
    buf = ""
    for line in para.split("\n"):
        if len(line) > chunk_chars:  # 单行仍超长（如断行丢失的电子转换 PDF）→ 按词硬切
            words = line.split(" ")
            for word in words:
                if buf and len(buf) + len(word) + 1 > chunk_chars:
                    pieces.append(buf)
                    buf = word
                else:
                    buf = f"{buf} {word}" if buf else word
            continue
        if buf and len(buf) + len(line) + 1 > chunk_chars:
            pieces.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        pieces.append(buf)
    return pieces


_TEXT_PAGE_CHARS = 3000  # 文本文档伪分页粒度：在线阅读 extract 产物的"页"


def _parse_text_doc(path: Path, *, paper_id: str | None = None) -> PaperDoc:
    """文本文档（.md/.txt）→ PaperDoc：按空行切段落、~3000 字符伪分页、复用 _build_chunks。

    在线阅读（AnySearch extract）产物的读取路径：伪页序号进 Chunk.page，
    [paper_id:页码] 引用与核验链路零改动。
    """
    tracer = get_active_tracer()
    doc = PaperDoc(pdf_path=str(path))
    if not path.exists():
        tracer.event("tool_error", node="pdf_tool", tool="parse_pdf", error=f"文件不存在: {path}")
        return doc
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        tracer.event("tool_error", node="pdf_tool", tool="parse_pdf", error=f"读取失败: {type(exc).__name__}: {exc}")
        return doc
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pages: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for para in paras:
        if cur and cur_len + len(para) > _TEXT_PAGE_CHARS:
            pages.append(cur)
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para)
    if cur:
        pages.append(cur)
    if not pages:
        return doc
    sections = [Section(title="全文", page_from=1, page_to=len(pages))]
    s = get_settings()
    doc.paper_id = paper_id or f"sha256:{hash_file(path)[:12]}"
    doc.sha256 = hash_file(path)
    doc.n_pages = len(pages)
    doc.sections = sections
    doc.chunks = _build_chunks(
        pages,
        sections,
        max(int(s.source("pdf", "chunk_chars", 1400) or 1400), 200),
        max(int(s.source("pdf", "chunk_overlap", 160) or 0), 0),
        max(int(s.source("pdf", "min_chunk_chars", 180) or 0), 0),
    )
    tracer.event("tool_call", node="pdf_tool", tool="parse_text_doc",
                 path=str(path), n_pages=len(pages), n_chunks=len(doc.chunks))
    return doc


def _build_chunks(
    pages_paras: list[list[str]],
    sections: list[Section],
    chunk_chars: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[Chunk]:
    """按页累积段落、滑窗拼到 chunk_chars±，块间重叠 chunk_overlap，尾块过小并入前块。

    chunk.page = 块内首个来源段落的页码（重叠前缀只是检索上下文，不计来源页），
    保证记录的页码永远是块正文的真实起始页。
    """
    raw: list[tuple[int, str]] = []
    parts: list[str] = []
    first_page: int | None = None
    tail = ""

    def emit() -> None:
        nonlocal parts, first_page, tail
        if not parts or first_page is None:
            return
        body = "\n\n".join(parts)
        text = f"{tail}\n\n{body}" if tail else body
        raw.append((first_page, text))
        # 下一块的重叠上下文取自本块正文尾部（不含上一块的重叠前缀，避免叠加漂移）
        new_tail = body[-chunk_overlap:] if chunk_overlap > 0 else ""
        sp = new_tail.find(" ")
        if 0 <= sp < 40:  # 截断处切开单词时丢弃残词
            new_tail = new_tail[sp + 1:]
        tail = new_tail.strip()
        parts, first_page = [], None

    for pno, paras in enumerate(pages_paras, start=1):
        for para in paras:
            for piece in _split_long_paragraph(para, chunk_chars):
                cur = len(tail) + (len("\n\n".join(parts)) if parts else 0) + (2 if tail and parts else 0)
                if parts and cur + len(piece) + 2 > chunk_chars:
                    emit()
                if first_page is None:
                    first_page = pno
                parts.append(piece)
    emit()

    # 尾块低于 min_chunk_chars 时并入前块（保留前块的起始页）
    while len(raw) >= 2 and len(raw[-1][1].strip()) < min_chunk_chars:
        prev_page, prev_text = raw[-2]
        raw[-2] = (prev_page, f"{prev_text}\n\n{raw[-1][1]}")
        raw.pop()

    return [
        Chunk(chunk_id=f"c{i}", page=page, section=_section_title_at(sections, page), text=text)
        for i, (page, text) in enumerate(raw)
    ]


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文/数字连续段（小写）+ CJK 字符逐字。"""
    if not text:
        return []
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    tokens.extend(ch for ch in text if "一" <= ch <= "鿿")
    return tokens


class BM25Index:
    """手写 Okapi BM25：k1=1.5、b=0.75，idf = log(1 + (N-df+0.5)/(df+0.5))。"""

    def __init__(self, chunks: list[Chunk]):
        self.k1 = 1.5
        self.b = 0.75
        self.chunks = list(chunks)
        self._tf: list[dict[str, int]] = []
        self._doc_len: list[int] = []
        df: dict[str, int] = {}
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            freq: dict[str, int] = {}
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
            self._tf.append(freq)
            self._doc_len.append(len(tokens))
            for tok in freq:
                df[tok] = df.get(tok, 0) + 1
        self._df = df
        self._n_docs = len(self.chunks)
        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0

    def query(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        """返回 [(Chunk, score)] 按 score 降序；无有效词或空索引返回 []。"""
        terms = set(tokenize(query))
        if not terms or not self._n_docs or not self._avgdl:
            return []
        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            tf = self._tf[i]
            norm = self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avgdl)
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
                score += idf * f * (self.k1 + 1.0) / (f + norm)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return scored if k is None else scored[:k]


# --------------------------------------------------------------------------
# 公共接口
# --------------------------------------------------------------------------
def make_paper_id_from_pdf(pdf_path: str | Path) -> str:
    """返回 sha256:<文件哈希前 12 位>；读取失败降级为 sha256:unknown。"""
    try:
        return f"sha256:{hash_file(pdf_path)[:12]}"
    except Exception as exc:
        get_active_tracer().event("tool_error", node="pdf_tool", tool="make_paper_id_from_pdf", error=str(exc))
        return "sha256:unknown"


def parse_pdf(pdf_path: str | Path, *, paper_id: str | None = None, max_pages: int | None = None) -> PaperDoc:
    """章节识别（字体/编号启发式）+ 分块（sources.yaml 参数，1-based 页码）+ 扫描件检测。

    后缀为 .md/.txt 时走文本文档路径（在线阅读的 extract 产物）：按 ~3000 字符伪分页，
    页码即伪页序号——[paper_id:页码] 引用契约不变（页指"正文伪页"，非印刷页）。
    """
    tracer = get_active_tracer()
    path = Path(pdf_path)
    if path.suffix.lower() in (".md", ".txt"):
        return _parse_text_doc(path, paper_id=paper_id)
    doc = PaperDoc(pdf_path=str(path))
    if not path.exists():
        tracer.event("tool_error", node="pdf_tool", tool="parse_pdf", error=f"文件不存在: {path}")
        return doc
    try:
        s = get_settings()
        if max_pages is None:
            max_pages = int(s.budget("max_pdf_pages", 80) or 80)
        sha = hash_file(path)
        doc.sha256 = sha
        doc.paper_id = paper_id or f"sha256:{sha[:12]}"

        chunk_chars = max(int(s.source("pdf", "chunk_chars", 1400) or 1400), 200)
        chunk_overlap = max(int(s.source("pdf", "chunk_overlap", 160) or 0), 0)
        min_chunk_chars = max(int(s.source("pdf", "min_chunk_chars", 180) or 0), 0)
        scanned_threshold = float(s.source("pdf", "scanned_text_threshold", 120) or 120)

        pages_lines: list[list[tuple[str, float]]] = []
        pages_paras: list[list[str]] = []
        size_weight: dict[float, int] = {}
        total_chars = 0
        with pymupdf.open(path) as pdf:
            doc.n_pages = pdf.page_count
            limit = max(0, min(pdf.page_count, max_pages))
            for pno in range(limit):
                lines, paras = _extract_page(pdf[pno].get_text("dict"))
                pages_lines.append(lines)
                pages_paras.append(paras)
                total_chars += sum(len(p) for p in paras)
                for text, size in lines:  # 按文本长度加权的众数字号 = 正文主字号
                    key = round(size * 2) / 2
                    size_weight[key] = size_weight.get(key, 0) + len(text)

        main_size = max(size_weight, key=lambda k: size_weight[k]) if size_weight else 0.0

        headings = _detect_headings(pages_lines, main_size)
        if headings:
            sections = []
            for i, (pg, title) in enumerate(headings):
                next_pg = headings[i + 1][0] if i + 1 < len(headings) else limit
                sections.append(Section(title=title, page_from=pg, page_to=max(pg, next_pg)))
        elif limit:
            sections = [Section(title="fulltext", page_from=1, page_to=limit)]
        else:
            sections = []
        doc.sections = sections

        doc.chunks = _build_chunks(pages_paras, sections, chunk_chars, chunk_overlap, min_chunk_chars)
        doc.is_scanned = limit > 0 and total_chars / limit < scanned_threshold
        return doc
    except Exception as exc:
        tracer.event("tool_error", node="pdf_tool", tool="parse_pdf", error=f"{type(exc).__name__}: {exc}")
        return doc


# 模块级解析缓存：{(abspath, mtime): PaperDoc}，避免同一 PDF 重复解析
_doc_cache: dict[tuple[str, float], PaperDoc] = {}
_cache_lock = threading.Lock()


def _resolve_doc(source: "str | Path | PaperDoc") -> PaperDoc | None:
    """source → PaperDoc：已是 PaperDoc 直接用；路径则查缓存或解析。"""
    if isinstance(source, PaperDoc):
        return source
    try:
        path = Path(source)
        if not path.exists():
            get_active_tracer().event("tool_error", node="pdf_tool", tool="retrieve_chunks", error=f"文件不存在: {path}")
            return None
        key = (str(path.resolve()), path.stat().st_mtime)
        with _cache_lock:
            cached = _doc_cache.get(key)
        if cached is not None:
            return cached
        doc = parse_pdf(path)
        with _cache_lock:
            _doc_cache[key] = doc
        return doc
    except Exception as exc:
        get_active_tracer().event("tool_error", node="pdf_tool", tool="retrieve_chunks", error=f"{type(exc).__name__}: {exc}")
        return None


def retrieve_chunks(source: "str | Path | PaperDoc", query: str, *, k: int = 8) -> list[dict]:
    """混合检索（BM25 [+ 向量]），返回 [{"page","section","text","score"}] 降序。

    向量侧可用时对全部分块算稠密相似，与 BM25 序做 RRF 融合（k=60）——
    语义近邻（同义改写/概念性查询）能补词法失配；不可用/失败退纯 BM25，行为不变。
    """
    try:
        doc = _resolve_doc(source)
        if doc is None or not doc.chunks:
            return []
        # 候选池放大到 3k：融合路径需要词法+稠密两路的完整排序信息
        pool = BM25Index(doc.chunks).query(query, k=min(len(doc.chunks), max(3 * k, k)))
        fused = _fuse_with_dense(doc.chunks, pool, query, k)
        ordered = fused if fused is not None else pool[:k]
        score_by_id = {id(chunk): score for chunk, score in pool}
        return [
            {"page": chunk.page, "section": chunk.section, "text": chunk.text,
             "score": round(score_by_id.get(id(chunk), 0.0), 4)}
            for chunk, _rrf in ordered
        ]
    except Exception as exc:
        get_active_tracer().event("tool_error", node="pdf_tool", tool="retrieve_chunks", error=f"{type(exc).__name__}: {exc}")
        return []


def _fuse_with_dense(
    chunks: list, lex_hits: list, query: str, k: int
) -> list | None:
    """稠密序与词法序的 RRF 融合（k=60）；向量侧不可用/失败返回 None（退纯 BM25）。

    稠密序在全部分块上排序——语义近邻即使零词法命中（同义改写/概念性查询）
    也能经 RRF 进入 top-k，这正是混合检索相对纯 BM25 的增量。
    """
    from src.tools import embedding as emb

    if not emb.embedding_available() or not query.strip():
        return None
    try:
        vecs = emb.embed_texts([query] + [c.text for c in chunks])
    except Exception:  # noqa: BLE001 —— embed_texts 自身不抛，防御双重保险
        return None
    if not vecs:
        return None
    q_vec, chunk_vecs = vecs[0], vecs[1:]
    if not q_vec:
        return None
    lex_ids = [id(c) for c, _s in lex_hits][: max(3 * k, k)]
    dense_order = sorted(range(len(chunks)), key=lambda i: -emb.cosine(q_vec, chunk_vecs[i]))
    dense_ids = [id(chunks[i]) for i in dense_order[: max(3 * k, k)]]
    k_rrf = 60
    scores: dict[int, float] = {}
    for ranks in (lex_ids, dense_ids):
        for r, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + r + 1)
    by_id = {id(c): c for c in chunks}
    return [(by_id[cid], scores[cid]) for cid in sorted(scores, key=lambda c: -scores[c])[:k]]


def _clean_title(run: str) -> str | None:
    """过滤 arXiv 水印 / 纯数字 / DOI 行，返回规范化的候选题名。"""
    cand = re.sub(r"\s+", " ", run).strip().strip("\"'“”‘’")
    if len(cand) < 3 or not _ALPHA_RE.search(cand):
        return None
    low = cand.lower()
    if "arxiv:" in low or low.startswith("doi") or re.search(r"\b10\.\d{4,9}/", low):
        return None
    if cand.replace(" ", "").isdigit():
        return None
    return cand


def extract_title(source: "str | Path | PaperDoc") -> str | None:
    """首页最大字号文本行启发式提取题名。

    先剔除无效行（arXiv 水印 / 纯数字 / DOI —— 它们常是页面上字号最大的元素），
    再取剩余行中的最大字号，连续同字号（±抖动容差）行拼接为题名，截 200 字符。
    """
    tracer = get_active_tracer()
    try:
        path = Path(source.pdf_path) if isinstance(source, PaperDoc) else Path(source)
        with pymupdf.open(path) as pdf:
            if pdf.page_count == 0:
                return None
            lines, _ = _extract_page(pdf[0].get_text("dict"))
        if not lines:
            return None
        valid_sizes = [size for text, size in lines if _clean_title(text)]
        if not valid_sizes:
            return None
        max_size = max(valid_sizes)
        runs: list[str] = []
        current: list[str] = []
        prev_idx = -2
        for idx, (text, size) in enumerate(lines):
            if size >= max_size - _TITLE_SIZE_TOL and _clean_title(text):
                if current and idx != prev_idx + 1:
                    runs.append(" ".join(current))
                    current = []
                current.append(text)
                prev_idx = idx
        if current:
            runs.append(" ".join(current))
        for run in runs:
            cand = _clean_title(run)
            if cand:
                return truncate(cand, 200)
        return None
    except Exception as exc:
        tracer.event("tool_error", node="pdf_tool", tool="extract_title", error=f"{type(exc).__name__}: {exc}")
        return None
