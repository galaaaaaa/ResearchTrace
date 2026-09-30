"""在线阅读：AnySearch /v1/extract 抓网页全文 Markdown（无 PDF 直链来源的全文通道）。

设计要点：
- 与 search_web_literature 同一套开关（sources.yaml web_search.enable_for_research，
  默认 false）与 anysearch 密钥；**绝不抛异常**，失败/内容过短返回 None；
- extract 对部分站点会 422 extract_failed（实测 huggingface blog 失败、wikipedia 成功），
  失败即失败，不换协议硬抓；
- 返回前清洗：markdown 链接保留锚文本、丢弃纯导航行（[Jump to content] 类）。
"""
from __future__ import annotations

import re

import httpx

from src.settings import get_settings
from src.tools.web_search import _anysearch_api_key, _provider
from src.tracing import get_active_tracer
from src.utils import truncate

_EXTRACT_URL = "https://api.anysearch.com/v1/extract"
_MIN_CONTENT_CHARS = 500  # 落地页/付费墙壳的特征：能抓到的正文极短
_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")
# 整行只是链接（可逗号/竖线串联多个）→ 导航行，替换锚文本之前就丢弃
_PURE_LINK_LINE_RE = re.compile(r"^(?:\[[^\]]*\]\([^)]*\)[\s,|;]*)+$")


def extract_available() -> bool:
    """在线阅读通道是否可用（同 search_web_literature 的开关与服务商约束）。"""
    return bool(
        get_settings().source("web_search", "enable_for_research", False)
        and _provider() == "anysearch"
        and _anysearch_api_key()
    )


def _clean_markdown(content: str) -> str:
    """链接留锚文本、去纯链接导航行与引用脚注锚（wiki 的 [[14]](url) 残留），压缩多余空行。"""
    lines = [ln for ln in (content or "").splitlines() if not _PURE_LINK_LINE_RE.match(ln.strip())]
    text = _LINK_RE.sub(r"\1", "\n".join(lines))
    text = re.sub(r"\[?\[\d{1,3}\]?", "", text)  # 脚注锚 [14] / 残留的嵌套括号
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _post_extract(url: str) -> dict:
    """调用 extract 端点（内部使用；错误向上抛，由 extract_url 统一退 None）。"""
    timeout = float(get_settings().source("web_search", "request_timeout_s", 60) or 60)
    resp = httpx.post(
        _EXTRACT_URL,
        json={"url": url},
        headers={"Authorization": f"Bearer {_anysearch_api_key()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json() or {}


def extract_url(url: str) -> str | None:
    """抓取网页正文 Markdown（上限约 5 万字符由服务端截断）；失败返回 None。"""
    tracer = get_active_tracer()
    if not extract_available():
        return None
    u = (url or "").strip()
    if not u.startswith(("http://", "https://")):
        return None
    try:
        data = _post_extract(u)
    except Exception as e:  # noqa: BLE001 —— 在线阅读是增强不是依赖
        tracer.event("tool_error", tool="extract_url", error=f"{type(e).__name__}: {truncate(str(e), 160)}")
        return None
    if data.get("code") != 0:
        tracer.event(
            "tool_error", tool="extract_url",
            error=f"code={data.get('code')} {truncate(str(data.get('message') or ''), 120)} url={truncate(u, 80)}",
        )
        return None
    text = _clean_markdown(str((data.get("data") or {}).get("content") or ""))
    if len(text) < _MIN_CONTENT_CHARS:
        tracer.event("tool_skip", tool="extract_url", reason=f"正文过短（{len(text)} 字符），疑似落地页", url=truncate(u, 80))
        return None
    return text
