"""联网搜索（对话 🌐 开关的检索后端）：智谱 BigModel / AnySearch / 阿里云 IQS 三选一。

设计要点：
- 只检索不生成：各服务商统一映射为 [{title, url, content, media, publish_date}]，
  作答仍由对话模型完成——这样能复用 webapp 已验证的“片段放用户消息末尾 + [n] 引用”防线；
- 服务商切换：WEB_SEARCH_PROVIDER=zhipu（默认）| anysearch | iqs（别名 tongxiao）；
  智谱走 /api/paas/v4/web_search（引擎档位 WEB_SEARCH_ENGINE，元/次：search_std 0.01、
  search_pro 0.03、search_pro_sogou/_quark 0.05）；
  AnySearch（anysearch.com，AI Agent 搜索基础设施——**独立公司，与阿里云无关**；
  曾误把两家当成一家、把 anysearch 接到阿里云 IQS 端点，密钥前缀 as_sk_）走
  POST api.anysearch.com/v1/search（max_results 1–10、format=json、REST 无 recency
  过滤；结果无 media/publish_date，media 取 URL hostname；免费额度 1000 次/天；
  匿名模式可用但本模块要求显式配 key——共享出口 IP 下匿名配额不可控）；
  阿里云 IQS“通晓”（信息查询服务）走 UnifiedSearch
  POST cloud-iqs.aliyuncs.com/search/unified（LiteAdvanced 引擎，snippet 自带约 500 字
  摘要，无需再开收费的 summary；密钥创建后约 5 分钟生效）；
- 与学术工具的“绝不抛异常”不同，这里显式分层抛错：未配置密钥 → WebSearchNotConfigured
  （webapp 返回 400 引导配置），重试后仍失败 → WebSearchError（webapp 返回 502）；
- 官方文档：智谱 https://docs.bigmodel.cn/cn/guide/tools/web-search
  （query 建议 ≤70 字符，count 1–50）；
  AnySearch https://www.anysearch.com/docs（code:0/-1 信封，max_results 1–10）；
  IQS https://help.aliyun.com/zh/document_detail/2987411.html
  （query ≤1024 字符，numResults 1–50，错误码表见文档）。
"""
from __future__ import annotations

import os
import re
import threading
from urllib.parse import urlparse

import httpx
import tenacity

from src.schemas import PaperRecord
from src.settings import get_settings
from src.tracing import get_active_tracer
from src.utils import sha256_text, truncate

_API_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
_ENGINES = ("search_std", "search_pro", "search_pro_sogou", "search_pro_quark")
_MAX_QUERY_CHARS = 70  # 智谱官方建议上限，超出可能被拒
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_IQS_API_URL = "https://cloud-iqs.aliyuncs.com/search/unified"
_IQS_QUERY_MAX = 1024  # IQS UnifiedSearch 上限（智谱是 70）
_IQS_ENGINE = "LiteAdvanced"  # 文档当前唯一引擎档；snippet 约 500 字，无需再开 summary（收费）
# 阿里云 IQS 密钥环境变量（按序取第一个非空）；AnySearch(anysearch.com) 是另一家独立服务
_IQS_KEY_ENVS = ("IQS_API_KEY", "TONGXIAO_API_KEY")
# AnySearch（anysearch.com）端点与限额：REST /v1/search，密钥形如 as_sk_…
_ANY_API_URL = "https://api.anysearch.com/v1/search"
_ANY_MAX_RESULTS = 10  # REST 上限 1–10（智谱/IQS 是 50）
_EM_RE = re.compile(r"</?em>")  # IQS snippet 里的关键词高亮标签，喂给模型前去掉
# 智谱 recency 值 → IQS timeRange 值（大小写不同，逐一映射）；AnySearch REST 无对应参数
_IQS_TIME_RANGE = {
    "nolimit": "NoLimit", "oneday": "OneDay", "oneweek": "OneWeek",
    "onemonth": "OneMonth", "oneyear": "OneYear",
}
# AnySearch 密钥环境变量（生态约定名就是 ANYSEARCH_API_KEY）
_ANY_KEY_ENVS = ("ANYSEARCH_API_KEY",)


class WebSearchNotConfigured(RuntimeError):
    """未配置智谱密钥（ZHIPUAI_API_KEY）——调用方应引导配置而非当作服务故障。"""


class WebSearchError(RuntimeError):
    """联网搜索请求失败（网络/HTTP 错误，重试耗尽后仍失败）。"""


_client_lock = threading.Lock()
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    """模块级复用的 httpx.Client（线程安全，跟随重定向）。"""
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                follow_redirects=True,
                headers={"User-Agent": "research-agent/0.1 (web search)"},
            )
        return _client


def _api_key() -> str:
    """密钥：env 名可用 sources.yaml 的 web_search.api_key_env 覆盖，兼容 ZHIPU_API_KEY。"""
    key_env = get_settings().source("web_search", "api_key_env", "ZHIPUAI_API_KEY")
    key = os.environ.get(str(key_env or ""), "").strip() if key_env else ""
    return key or os.environ.get("ZHIPU_API_KEY", "").strip()


def _is_retryable(exc: BaseException) -> bool:
    """仅超时/连接错误与 429/5xx 重试。"""
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503}
    return False


_retry_policy = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_retryable),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)


def _error_detail(exc: BaseException) -> tuple[str, str]:
    """从 HTTPStatusError 里解析错误码与消息（code, message；解析失败返回空串）。

    兼容三种响应体：智谱嵌套 {"error": {code, message}}；阿里云文档写的顶层
    {"code", "message"}；阿里云实测的顶层 {"errorCode", "errorMessage"}（如
    403 Retrieval.InvalidAPIKey 密钥无效——键名与文档不符，只认文档键会全部漏进兜底文案）。
    """
    resp = getattr(exc, "response", None)
    try:
        data = resp.json() or {}  # type: ignore[union-attr]
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("code") or ""), str(err.get("message") or "")
        return (
            str(data.get("code") or data.get("errorCode") or ""),
            str(data.get("message") or data.get("errorMessage") or ""),
        )
    except Exception:  # noqa: BLE001 —— 只影响提示文案，不参与控制流
        return "", ""


def _engine() -> str:
    """智谱搜索引擎档位：WEB_SEARCH_ENGINE 环境变量（或 sources.yaml web_search.engine）。"""
    engine = os.environ.get("WEB_SEARCH_ENGINE", "").strip() or str(
        get_settings().source("web_search", "engine", "search_std")
    )
    return engine if engine in _ENGINES else "search_std"


def _provider() -> str:
    """搜索服务商：WEB_SEARCH_PROVIDER 环境变量（或 sources.yaml web_search.provider）。

    返回 "zhipu"（默认，向后兼容）、"anysearch"（anysearch.com）或
    "iqs"（阿里云 IQS 通晓，别名 tongxiao 归一化）。
    """
    raw = os.environ.get("WEB_SEARCH_PROVIDER", "").strip() or str(
        get_settings().source("web_search", "provider", "zhipu")
    )
    if raw.lower() == "anysearch":
        return "anysearch"
    if raw.lower() in ("iqs", "tongxiao"):
        return "iqs"
    return "zhipu"


def _iqs_api_key() -> str:
    """阿里云 IQS 密钥：兼容 IQS_API_KEY / TONGXIAO_API_KEY（与 AnySearch 密钥无关）。"""
    for name in _IQS_KEY_ENVS:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    return ""


def _anysearch_api_key() -> str:
    """AnySearch（anysearch.com）密钥：ANYSEARCH_API_KEY，控制台创建，形如 as_sk_…。"""
    for name in _ANY_KEY_ENVS:
        key = os.environ.get(name, "").strip()
        if key:
            return key
    return ""


@_retry_policy
def _http_post(payload: dict) -> httpx.Response:
    """带重试的智谱 POST（Authorization 按调用时的密钥生成，便于 .env 后配置）。"""
    timeout = float(get_settings().source("web_search", "request_timeout_s", 20) or 20)
    resp = _get_client().post(
        _API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


@_retry_policy
def _http_post_iqs(payload: dict) -> httpx.Response:
    """带重试的阿里云 IQS POST，Bearer API-KEY（密钥创建后约 5 分钟生效）。"""
    timeout = float(get_settings().source("web_search", "request_timeout_s", 20) or 20)
    resp = _get_client().post(
        _IQS_API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {_iqs_api_key()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


@_retry_policy
def _http_post_anysearch(payload: dict) -> httpx.Response:
    """带重试的 AnySearch（anysearch.com）POST，Bearer as_sk_ 密钥。"""
    timeout = float(get_settings().source("web_search", "request_timeout_s", 20) or 20)
    resp = _get_client().post(
        _ANY_API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {_anysearch_api_key()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


def search_web(query: str, *, count: int = 8, recency: str = "noLimit") -> list[dict]:
    """联网检索，返回 [{title, url, content, media, publish_date}]（各服务商统一契约）。

    - 服务商：WEB_SEARCH_PROVIDER=zhipu（默认）| anysearch（anysearch.com）| iqs（阿里云）；
    - query 压缩空白，空查询直接返回 []（不烧调用费）；各家上限不同（智谱 70 / IQS 1024 /
      AnySearch 未标注），截断与 count 收紧（AnySearch ≤10）在各自实现内完成；
    - 未配置密钥 → WebSearchNotConfigured；网络/HTTP 失败（重试耗尽）→ WebSearchError；
    - 接口正常但无结果 → 返回 []，降级文案由调用方决定。
    """
    q = " ".join((query or "").split())
    if not q:
        return []
    provider = _provider()
    if provider == "anysearch":
        return _search_anysearch(q, count=count, recency=recency)
    if provider == "iqs":
        return _search_iqs(q, count=count, recency=recency)
    return _search_zhipu(q, count=count, recency=recency)


def _search_zhipu(query: str, *, count: int, recency: str) -> list[dict]:
    """智谱 BigModel 联网检索（search_result → 统一契约）。"""
    tracer = get_active_tracer()
    q = query[:_MAX_QUERY_CHARS]
    if not _api_key():
        raise WebSearchNotConfigured(
            "未配置 ZHIPUAI_API_KEY（.env），联网搜索不可用；"
            "或改用阿里云 AnySearch（.env 设 WEB_SEARCH_PROVIDER=anysearch + ANYSEARCH_API_KEY）"
        )
    payload = {
        "search_query": q,
        "search_engine": _engine(),
        "search_intent": False,
        "count": max(1, min(int(count), 50)),
        "search_recency_filter": recency,
        "content_size": "medium",
    }
    try:
        resp = _http_post(payload)
        data = resp.json() or {}
    except Exception as e:  # noqa: BLE001 —— 统一转成可区分的搜索失败
        tracer.event("tool_error", tool="search_web", error=f"{type(e).__name__}: {e}"[:300])
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 429:
            # 智谱把"余额不足"（错误码 1113）也用 429 返回，必须读响应体区分——
            # 余额问题等再久也不会恢复，提示充值而不是等待
            code, message = _error_detail(e)
            if code == "1113" or "余额" in message:
                raise WebSearchError(
                    "搜索接口余额不足（智谱错误码 1113）：请到智谱开放平台（bigmodel.cn）充值或领取资源包"
                ) from e
            # 真限流：智谱搜索 API 频率限制较严（连发几次即触发，窗口约数分钟），提示等待而非报障
            raise WebSearchError("搜索接口限流（429）：请求过于频繁，请等一两分钟再试") from e
        if status == 401:
            raise WebSearchError("搜索接口认证失败（401）：请检查 .env 里的 ZHIPUAI_API_KEY 是否正确") from e
        raise WebSearchError(f"联网搜索请求失败（{status or '网络错误'}）：{truncate(str(e), 160)}") from e
    hits: list[dict] = []
    for it in data.get("search_result") or []:
        if not isinstance(it, dict):
            continue
        content = " ".join(str(it.get("content") or "").split())
        url = str(it.get("link") or "").strip()
        if not content and not url:
            continue
        date_raw = str(it.get("publish_date") or "")
        m = _DATE_RE.search(date_raw)
        hits.append(
            {
                "title": " ".join(str(it.get("title") or "").split()) or truncate(url, 60) or "(无标题)",
                "url": url,
                "content": content,
                "media": " ".join(str(it.get("media") or "").split()),
                "publish_date": m.group(0) if m else (date_raw or None),
            }
        )
    tracer.event("tool_call", tool="search_web", engine=payload["search_engine"], query=q, n_results=len(hits))
    return hits


def _search_anysearch(query: str, *, count: int, recency: str) -> list[dict]:
    """AnySearch（anysearch.com /v1/search）联网检索。

    与另两家的差异（官方 docs + 实测）：
    - 结果只有 title/url/snippet/content，无 media 与 publish_date——media 取 URL
      hostname，publish_date 置 None（对话端引用栏不显示日期）；
    - snippet 优先、缺省退 content（清洗过的全文，截到 500 字与 IQS snippet 等量）；
    - REST 端点不支持 recency 过滤（MCP 侧才有 freshness），入参忽略；
    - 认证失败不静默降级匿名（官方明确 401/403），免费额度 1000 次/天。
    """
    tracer = get_active_tracer()
    if not _anysearch_api_key():
        raise WebSearchNotConfigured(
            "未配置 ANYSEARCH_API_KEY（.env，anysearch.com 控制台创建，形如 as_sk_…），联网搜索不可用"
        )
    payload = {
        "query": query,
        "max_results": max(1, min(int(count), _ANY_MAX_RESULTS)),
        "format": "json",
    }
    try:
        resp = _http_post_anysearch(payload)
        data = resp.json() or {}
    except Exception as e:  # noqa: BLE001 —— 统一转成可区分的搜索失败
        tracer.event("tool_error", tool="search_web", error=f"{type(e).__name__}: {e}"[:300])
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            raise WebSearchError(
                "搜索接口认证失败：请检查 .env 里的 ANYSEARCH_API_KEY 是否有效"
                "（anysearch.com/console/api-keys；无效/过期 key 直接 401/403，不降级匿名）"
            ) from e
        if status == 429:
            raise WebSearchError(
                "搜索接口限流或配额用尽（429）：免费额度 1000 次/天，请稍后再试或到控制台换 key"
            ) from e
        raise WebSearchError(f"联网搜索请求失败（{status or '网络错误'}）：{truncate(str(e), 160)}") from e
    if data.get("code") != 0:
        # HTTP 200 也可能带业务错误：code:-1 + message（与 HTTP 错误共用同一信封）
        raise WebSearchError(
            f"AnySearch 业务错误（code={data.get('code')}）：{truncate(str(data.get('message') or ''), 160)}"
        )
    hits: list[dict] = []
    for it in (data.get("data") or {}).get("results") or []:
        if not isinstance(it, dict):
            continue
        url = str(it.get("url") or "").strip()
        snippet = " ".join(str(it.get("snippet") or "").split())
        content = snippet or truncate(" ".join(str(it.get("content") or "").split()), 500)
        if not content and not url:
            continue
        hits.append(
            {
                "title": " ".join(str(it.get("title") or "").split()) or truncate(url, 60) or "(无标题)",
                "url": url,
                "content": content,
                "media": urlparse(url).netloc.removeprefix("www.") if url else "",
                "publish_date": None,
            }
        )
    tracer.event(
        "tool_call", tool="search_web", engine="anysearch:v1",
        query=payload["query"], n_results=len(hits),
    )
    return hits


def _search_iqs(query: str, *, count: int, recency: str) -> list[dict]:
    """阿里云 IQS 通晓（信息查询服务 UnifiedSearch，LiteAdvanced）联网检索。

    错误码语义比智谱清晰（官方错误表）：
    403 Retrieval.Arrears 余额不足、403 Retrieval.NotActivate 未开通、
    403 Retrieval.TestUserPeriodExpired 测试期（下单后 15 天）已过、
    429 Retrieval.TestUserQueryPerDayExceeded 测试期日限额 1000 次/天、
    429 Retrieval.Throttling.User 真限流、403 Retrieval.InvalidAPIKey /
    404 InvalidAccessKeyId.NotFound 密钥无效（创建后约 5 分钟生效）。
    注意实测错误体键名是 errorCode/errorMessage，不是文档写的 code/message。
    """
    tracer = get_active_tracer()
    if not _iqs_api_key():
        raise WebSearchNotConfigured(
            "未配置 IQS_API_KEY（.env，阿里云 IQS 控制台创建，创建后约 5 分钟生效），联网搜索不可用"
        )
    payload = {
        "query": query[:_IQS_QUERY_MAX],
        "engineType": _IQS_ENGINE,
        "timeRange": _IQS_TIME_RANGE.get((recency or "").strip().lower(), "NoLimit"),
        "contents": {"mainText": False, "markdownText": False, "summary": False, "rerankScore": True},
        "advancedParams": {"numResults": str(max(1, min(int(count), 50)))},
    }
    try:
        resp = _http_post_iqs(payload)
        data = resp.json() or {}
    except Exception as e:  # noqa: BLE001 —— 统一转成可区分的搜索失败
        tracer.event("tool_error", tool="search_web", error=f"{type(e).__name__}: {e}"[:300])
        status = getattr(getattr(e, "response", None), "status_code", None)
        code, message = _error_detail(e)
        if status == 403 and (code == "Retrieval.Arrears" or "recharge" in message.lower()):
            raise WebSearchError(
                "搜索接口余额不足（阿里云错误码 Retrieval.Arrears）：请到阿里云控制台充值后重试"
            ) from e
        if status == 403 and code == "Retrieval.InvalidAPIKey":
            raise WebSearchError(
                "搜索接口认证失败（IQS_API_KEY 无效）：请到阿里云 API-KEY 管理页"
                "（ipaas.console.aliyun.com/api-key）核对；新建密钥约 5 分钟后生效"
            ) from e
        if status == 403 and code == "Retrieval.NotActivate":
            raise WebSearchError(
                "未开通阿里云信息查询服务（IQS）：请到 IQS 控制台（aliyun.com/product/iqs）下单开通"
            ) from e
        if status == 403 and code == "Retrieval.TestUserPeriodExpired":
            raise WebSearchError(
                "IQS 测试期已过（下单后 15 天有效）：请联系阿里云客户经理转正式套餐"
            ) from e
        if status == 429 and code == "Retrieval.TestUserQueryPerDayExceeded":
            raise WebSearchError(
                "IQS 测试期日限额（1000 次/天）已用尽：请明天再试或转正式套餐"
            ) from e
        if status == 429:
            raise WebSearchError("搜索接口限流（429）：请求过于频繁，请等一两分钟再试") from e
        if status in (401, 404):
            raise WebSearchError(
                "搜索接口认证失败：请检查 .env 里的 IQS_API_KEY 是否正确"
                "（IQS 控制台创建后约 5 分钟才生效）"
            ) from e
        raise WebSearchError(f"联网搜索请求失败（{status or '网络错误'}）：{truncate(str(e), 160)}") from e
    hits: list[dict] = []
    for it in data.get("pageItems") or []:
        if not isinstance(it, dict):
            continue
        # snippet 含 <em>关键词</em> 高亮标签，喂给模型前去掉；publishedTime 为 ISO 格式
        content = " ".join(_EM_RE.sub("", str(it.get("snippet") or "")).split())
        url = str(it.get("link") or "").strip()
        if not content and not url:
            continue
        date_raw = str(it.get("publishedTime") or "")
        m = _DATE_RE.search(date_raw)
        hits.append(
            {
                "title": " ".join(str(it.get("title") or "").split()) or truncate(url, 60) or "(无标题)",
                "url": url,
                "content": content,
                "media": " ".join(str(it.get("hostname") or "").split()),
                "publish_date": m.group(0) if m else (date_raw or None),
            }
        )
    tracer.event(
        "tool_call", tool="search_web", engine=f"anysearch:{_IQS_ENGINE}",
        query=payload["query"], n_results=len(hits),
    )
    return hits


# --------------------------------------------------------------------------
# Searcher 正式工具：AnySearch 学术垂直域检索（研究管线用，与自由对话的
# search_web 分开——这里绝不抛异常，未启用/未配 key/失败一律返回 []）
# --------------------------------------------------------------------------
# arXiv 链接里的 id：新式 2305.18290（含 v2 版本号）/ 旧式 cs.CL/0703007
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})", re.IGNORECASE
)
# doi.org 落地页 → DOI 本体（paper_id 约定 DOI 优先，可与 crossref 结果去重合并）
_DOI_URL_RE = re.compile(r"doi\.org/(10\.[^\s?#]+)")


def search_web_literature(
    query: str, *, max_results: int | None = 8, tag: str | None = None
) -> list[PaperRecord]:
    """AnySearch 学术垂直域检索 → list[PaperRecord]（Searcher 工具，失败返回 []）。

    与自由对话的 search_web 的区别：服务研究管线，绝不抛异常；结果映射为论文记录
    参与 searcher 的统一去重合并（arxiv:/DOI 前缀可与其他引擎的同一论文合并补全）：
    - arXiv 链接 → paper_id=arxiv:<id>，source_url 指向 PDF 主站
      （export API 被 429 限流时下载不受牵连——主站与 API 是两台服务）；
    - doi.org 链接 → paper_id=DOI；
    - 其余（技术博客/报告/数据集页）→ paper_id=web:<sha256_12>，abstract 取 snippet，
      无 PDF 可读，Reader 会如实标 partial，供分析层与人类参考。
    tag 默认 sources.yaml web_search.academic_tag（academic.search 跨学科 /
    academic.preprint 预印本 / academic.biomedical 生医；须用 sub-domains 接口的
    sub_domain 原值，加域前缀拼接会 400 Invalid tag——实测坑）。
    """
    tracer = get_active_tracer()
    s = get_settings()
    if not bool(s.source("web_search", "enable_for_research", False)):
        tracer.event("tool_skip", tool="search_web_literature", reason="web_search.enable_for_research 未开启")
        return []
    if _provider() != "anysearch":
        tracer.event("tool_skip", tool="search_web_literature", reason="仅 anysearch 服务商支持学术垂直域")
        return []
    if not _anysearch_api_key():
        tracer.event("tool_skip", tool="search_web_literature", reason="ANYSEARCH_API_KEY 未配置")
        return []
    q = " ".join((query or "").split())
    if not q:
        return []
    payload = {
        "query": q,
        "max_results": max(1, min(int(max_results or 8), _ANY_MAX_RESULTS)),
        "tag": tag or str(s.source("web_search", "academic_tag", "academic.search") or "academic.search"),
        "format": "json",
    }
    try:
        resp = _http_post_anysearch(payload)
        data = resp.json() or {}
    except Exception as e:  # noqa: BLE001 —— 研究管线绝不因检索工具抛异常中断
        tracer.event("tool_error", tool="search_web_literature", error=f"{type(e).__name__}: {truncate(str(e), 200)}")
        return []
    if data.get("code") != 0:
        tracer.event(
            "tool_error", tool="search_web_literature",
            error=f"code={data.get('code')} {truncate(str(data.get('message') or ''), 160)}",
        )
        return []
    papers: list[PaperRecord] = []
    for it in (data.get("data") or {}).get("results") or []:
        if not isinstance(it, dict):
            continue
        url = str(it.get("url") or "").strip()
        title = " ".join(str(it.get("title") or "").split())
        snippet = " ".join(str(it.get("snippet") or "").split())
        if not url or not title:
            continue
        m_ax = _ARXIV_URL_RE.search(url)
        if m_ax:
            ax = m_ax.group(1)
            papers.append(
                PaperRecord(
                    paper_id=f"arxiv:{ax}", title=title, arxiv_id=ax,
                    source_url=f"https://arxiv.org/pdf/{ax}",
                    abstract=truncate(snippet, 1000) or None,
                    paper_type="preprint", retrieval_query=q, source_api="anysearch",
                )
            )
            continue
        m_doi = _DOI_URL_RE.search(url)
        if m_doi:
            papers.append(
                PaperRecord(
                    paper_id=m_doi.group(1), title=title, doi=m_doi.group(1), source_url=url,
                    abstract=truncate(snippet, 1000) or None,
                    paper_type="other", retrieval_query=q, source_api="anysearch",
                )
            )
            continue
        papers.append(
            PaperRecord(
                paper_id=f"web:{sha256_text(url)[:12]}", title=title, source_url=url,
                abstract=truncate(snippet, 1000) or None,
                paper_type="preprint" if url.lower().endswith(".pdf") else "other",
                retrieval_query=q, source_api="anysearch",
            )
        )
    tracer.event("tool_call", tool="search_web_literature", query=q, n_results=len(papers))
    return papers
