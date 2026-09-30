"""PaperQA2 可选适配层（未安装时返回 None 优雅回退）。

paperqa 各版本 API 差异较大（Docs 构造参数、add/query 是否为协程、
Answer/Citation 字段名），因此全部走防御性 getattr + try/except：
任何一步失败 → tracer 记 error → 返回 None，由上层回退到内置 pdf_tool。
本模块自身不做网络请求；实际请求由（若安装的）paperqa 发出。
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from src.tracing import get_active_tracer


def paperqa_available() -> bool:
    """环境中是否安装了可导入的 paperqa（未安装时干净地返回 False）。"""
    try:
        import paperqa  # noqa: F401

        return True
    except Exception:
        return False


def _consume(result: Any) -> Any:
    """部分版本 add/query 返回协程：无事件循环时同步求值。"""
    if inspect.isawaitable(result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(result)
        raise RuntimeError("paperqa 返回协程但当前已在事件循环内，无法同步求值")
    return result


def _first_attr(obj: Any, *names: str) -> Any:
    """按顺序取第一个非 None 的属性（属性访问本身也可能抛异常）。"""
    for name in names:
        try:
            value = getattr(obj, name, None)
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def _citation_to_dict(citation: Any, fallback_path: str = "") -> dict:
    """把不同版本的 Citation 对象压平成统一 dict。"""
    doc = getattr(citation, "doc", None)
    pdf_path = _first_attr(citation, "path") or _first_attr(doc, "path", "name", "dockey")
    page = _first_attr(citation, "page", "page_number", "page_num")
    if page is None:
        locs = _first_attr(citation, "locs", "pages") or []
        if isinstance(locs, (list, tuple)) and locs:
            first = locs[0]
            if isinstance(first, int):
                page = first
            elif isinstance(first, (list, tuple)) and first and isinstance(first[0], int):
                page = first[0]
    text = _first_attr(citation, "text", "content", "chunk") or ""
    score = _first_attr(citation, "score", "relevance")
    return {
        "pdf_path": str(pdf_path) if pdf_path else fallback_path,
        "page": page if isinstance(page, int) else None,
        "text": str(text),
        "score": float(score) if isinstance(score, (int, float)) else None,
    }


def ask_papers(question: str, pdf_paths: list[str], *, max_sources: int | None = None) -> dict | None:
    """成功返回 {"answer": str, "citations": [{"pdf_path","page","text","score"}]}；不可用/失败返回 None。"""
    tracer = get_active_tracer()
    if not paperqa_available():
        tracer.event("tool_skip", node="paperqa_tool", reason="paperqa 未安装")
        return None
    if not question or not question.strip() or not pdf_paths:
        tracer.event("tool_skip", node="paperqa_tool", reason="question 或 pdf_paths 为空")
        return None
    try:
        import paperqa

        docs_cls = getattr(paperqa, "Docs", None)
        if docs_cls is None:
            raise RuntimeError("paperqa.Docs 不可用")

        docs = None
        for kwargs in ({}, {"llm": "default", "summary_llm": "default"}):
            try:
                docs = docs_cls(**kwargs)
                break
            except TypeError:
                continue
        if docs is None:
            raise RuntimeError("无法构造 paperqa Docs 实例")

        for raw_path in pdf_paths:
            try:
                _consume(docs.add(str(raw_path)))
            except TypeError:
                _consume(docs.add(path=str(raw_path)))

        answer = _consume(docs.query(question))
        text = _first_attr(answer, "text", "formatted_answer")
        if not isinstance(text, str) or not text.strip():
            tracer.event("tool_error", node="paperqa_tool", tool="ask_papers", error="Answer 无有效文本")
            return None

        raw_citations = _first_attr(answer, "citations", "sources") or []
        if not isinstance(raw_citations, (list, tuple)):
            raw_citations = []
        citations = []
        for citation in raw_citations:
            try:
                citations.append(_citation_to_dict(citation))
            except Exception:
                continue
        if max_sources is not None and max_sources > 0:
            citations = citations[:max_sources]
        return {"answer": text, "citations": citations}
    except Exception as exc:
        tracer.event("tool_error", node="paperqa_tool", tool="ask_papers", error=f"{type(exc).__name__}: {exc}")
        return None
