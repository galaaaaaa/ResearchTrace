"""web_search（联网检索）离线单测：智谱结果映射 + 429 双语义；AnySearch（anysearch.com）
与阿里云 IQS（通晓）两家服务商的分发/映射/错误码。

不发任何网络请求：monkeypatch _http_post / _http_post_anysearch / _http_post_iqs 返回假 httpx.Response。
"""
from __future__ import annotations

import httpx
import pytest

from src.tools import web_search
from src.tools.web_search import WebSearchError, WebSearchNotConfigured, search_web


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离服务商选择与各家密钥，避免 shell 环境泄漏影响断言。"""
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    for name in web_search._IQS_KEY_ENVS + web_search._ANY_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


class _Resp:
    """假 httpx.Response：只实现 json() 与 status_code（HTTPStatusError 只读这两者）。"""

    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def _status_err(status: int, payload: dict | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", web_search._API_URL)
    return httpx.HTTPStatusError(f"Client error {status}", request=req, response=_Resp(status, payload))


def _enable_anysearch(monkeypatch: pytest.MonkeyPatch, key: str = "as_sk_test") -> None:
    """切到 anysearch（anysearch.com，独立于阿里云）服务商并给密钥。"""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "anysearch")
    monkeypatch.setenv("ANYSEARCH_API_KEY", key)


def _enable_iqs(monkeypatch: pytest.MonkeyPatch, key: str = "ak-test") -> None:
    """切到 iqs（阿里云通晓）服务商并给密钥。"""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "iqs")
    monkeypatch.setenv("IQS_API_KEY", key)


# ---------------------------------------------------------------------------
# 基础行为
# ---------------------------------------------------------------------------
def test_empty_query_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """空查询直接返回 []（不校验密钥、不烧调用费）。"""
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    assert search_web("   ") == []


def test_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZHIPUAI_API_KEY", "")
    with pytest.raises(WebSearchNotConfigured):
        search_web("anything")


# ---------------------------------------------------------------------------
# 智谱：结果映射
# ---------------------------------------------------------------------------
def test_result_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZHIPUAI_API_KEY", "sk-test")

    def fake_post(payload: dict) -> _Resp:
        assert payload["search_query"] == "glm web search"  # 空白已压缩
        assert payload["count"] == 8
        return _Resp(200, {"search_result": [
            {"title": "  T1  ", "link": "https://a", "content": " c1 ", "media": " M ",
             "publish_date": "发布于 2026-09-01 10:00"},
            {"title": "", "link": "", "content": "", "media": "", "publish_date": ""},  # 全空 → 跳过
            {"title": "T3", "link": "https://c", "content": "c3", "media": "", "publish_date": ""},
        ]})

    monkeypatch.setattr(web_search, "_http_post", fake_post)
    hits = search_web("  glm   web  search  ")
    assert len(hits) == 2
    assert hits[0] == {"title": "T1", "url": "https://a", "content": "c1", "media": "M",
                       "publish_date": "2026-09-01"}  # 日期从自由文本中抽出
    assert hits[1]["publish_date"] is None and hits[1]["title"] == "T3"


# ---------------------------------------------------------------------------
# 智谱 429 双语义：余额不足（1113）vs 真限流
# ---------------------------------------------------------------------------
def test_429_balance_insufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    """智谱把余额不足（1113）也用 429 返回——必须读响应体区分，提示充值而非等待。"""
    monkeypatch.setenv("ZHIPUAI_API_KEY", "sk-test")

    def boom(payload: dict) -> _Resp:
        raise _status_err(429, {"error": {"code": "1113", "message": "余额不足或无可用资源包,请充值。"}})

    monkeypatch.setattr(web_search, "_http_post", boom)
    with pytest.raises(WebSearchError, match="余额不足"):
        search_web("q")


def test_429_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZHIPUAI_API_KEY", "sk-test")

    def boom(payload: dict) -> _Resp:
        raise _status_err(429)  # 无 JSON 体（如网关层限流）→ 按真限流处理

    monkeypatch.setattr(web_search, "_http_post", boom)
    with pytest.raises(WebSearchError, match="限流"):
        search_web("q")


def test_401_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZHIPUAI_API_KEY", "sk-wrong")

    def boom(payload: dict) -> _Resp:
        raise _status_err(401, {"error": {"code": "1000", "message": "invalid api key"}})

    monkeypatch.setattr(web_search, "_http_post", boom)
    with pytest.raises(WebSearchError, match="ZHIPUAI_API_KEY"):
        search_web("q")


# ---------------------------------------------------------------------------
# 服务商分发：anysearch（anysearch.com）与 iqs（阿里云）是两家不同的服务
# ---------------------------------------------------------------------------
def test_provider_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认走智谱；anysearch 打到 anysearch.com 端点，iqs/tongxiao 打到阿里云端点。"""
    monkeypatch.setenv("ZHIPUAI_API_KEY", "sk-z")
    monkeypatch.setattr(web_search, "_http_post", lambda p: _Resp(200, {"search_result": []}))
    any_calls: list[dict] = []
    iq_calls: list[dict] = []
    monkeypatch.setattr(
        web_search, "_http_post_anysearch",
        lambda p: any_calls.append(p) or _Resp(200, {"code": 0, "data": {"results": []}}),
    )
    monkeypatch.setattr(
        web_search, "_http_post_iqs",
        lambda p: iq_calls.append(p) or _Resp(200, {"pageItems": []}),
    )

    assert search_web("q") == []
    assert not any_calls and not iq_calls  # 默认智谱
    _enable_anysearch(monkeypatch)
    assert search_web("q") == []
    assert len(any_calls) == 1 and not iq_calls  # anysearch → anysearch.com
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "TONGXIAO")  # 别名归一化到 iqs
    _enable_iqs(monkeypatch)
    assert search_web("q") == []
    assert len(iq_calls) == 1  # iqs/tongxiao → 阿里云 IQS


# ---------------------------------------------------------------------------
# AnySearch（anysearch.com /v1/search）
# ---------------------------------------------------------------------------
def test_anysearch_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "anysearch")
    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
    with pytest.raises(WebSearchNotConfigured, match="ANYSEARCH_API_KEY"):
        search_web("q")


def test_anysearch_result_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """results → 统一契约：media 取 URL hostname、publish_date 恒 None、snippet 优先 content 兜底。"""
    _enable_anysearch(monkeypatch)

    def fake_post(payload: dict) -> _Resp:
        assert payload["query"] == "anysearch 联网 检索"  # 空白已压缩
        assert payload["max_results"] == 8
        assert payload["format"] == "json"
        return _Resp(200, {"code": 0, "data": {"results": [
            {"title": " T1 ", "url": "https://www.arxiv.org/abs/1", "snippet": " 摘要一 ", "content": "全文一"},
            {"title": "T2", "url": "https://b.example.com/2", "content": "全文" * 400},  # 无 snippet → 退 content
            {"title": "T3", "url": "", "snippet": "", "content": ""},  # 全空 → 跳过
        ]}})

    monkeypatch.setattr(web_search, "_http_post_anysearch", fake_post)
    hits = search_web("  anysearch  联网 检索 ", count=8)
    assert len(hits) == 2
    assert hits[0] == {"title": "T1", "url": "https://www.arxiv.org/abs/1", "content": "摘要一",
                       "media": "arxiv.org", "publish_date": None}  # www. 前缀剥掉
    assert len(hits[1]["content"]) == 500 and hits[1]["content"].endswith("…")  # content 截 500
    assert hits[1]["media"] == "b.example.com"


def test_anysearch_count_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """REST max_results 上限 10（智谱/IQS 是 50），count 超出须收紧。"""
    _enable_anysearch(monkeypatch)
    seen: dict[str, int] = {}

    def fake_post(payload: dict) -> _Resp:
        seen["n"] = payload["max_results"]
        return _Resp(200, {"code": 0, "data": {"results": []}})

    monkeypatch.setattr(web_search, "_http_post_anysearch", fake_post)
    assert search_web("q", count=50) == []
    assert seen["n"] == 10


def test_anysearch_401_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """无效 key 直接 401/403（官方明确不降级匿名），提示查 ANYSEARCH_API_KEY。"""
    _enable_anysearch(monkeypatch, key="as_sk_wrong")

    def boom(payload: dict) -> _Resp:
        raise _status_err(401, {"code": -1, "message": "invalid api key"})

    monkeypatch.setattr(web_search, "_http_post_anysearch", boom)
    with pytest.raises(WebSearchError, match="ANYSEARCH_API_KEY"):
        search_web("q")


def test_anysearch_business_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 200 但 code:-1 是业务错误（与 HTTP 错误共用信封），不能当成功空结果。"""
    _enable_anysearch(monkeypatch)

    def fake_post(payload: dict) -> _Resp:
        return _Resp(200, {"code": -1, "message": "upstream unavailable", "error_code": "E_UP"})

    monkeypatch.setattr(web_search, "_http_post_anysearch", fake_post)
    with pytest.raises(WebSearchError, match="AnySearch 业务错误"):
        search_web("q")


# ---------------------------------------------------------------------------
# 阿里云 IQS（通晓 UnifiedSearch）
# ---------------------------------------------------------------------------
def test_iqs_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "iqs")
    monkeypatch.delenv("IQS_API_KEY", raising=False)
    with pytest.raises(WebSearchNotConfigured, match="IQS_API_KEY"):
        search_web("q")


def test_iqs_result_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """pageItems → 统一契约：<em> 高亮剥离、hostname→media、ISO publishedTime→日期。"""
    _enable_iqs(monkeypatch)

    def fake_post(payload: dict) -> _Resp:
        assert payload["query"] == "阿里云 通晓 搜索"  # 空白已压缩
        assert payload["engineType"] == "LiteAdvanced"
        assert payload["advancedParams"]["numResults"] == "8"  # IQS 要求字符串
        assert payload["contents"] == {"mainText": False, "markdownText": False, "summary": False, "rerankScore": True}
        assert payload["timeRange"] == "NoLimit"
        return _Resp(200, {"pageItems": [
            {"title": " T1 ", "link": "https://a", "snippet": " 带<em>高亮</em>的 摘要 ",
             "publishedTime": "2024-12-31T00:00:00+08:00", "hostname": " 站点一 "},
            {"title": "T2", "link": "", "snippet": "", "hostname": ""},  # 全空 → 跳过
        ]})

    monkeypatch.setattr(web_search, "_http_post_iqs", fake_post)
    hits = search_web("  阿里云  通晓 搜索  ", count=8)
    assert hits == [{"title": "T1", "url": "https://a", "content": "带高亮的 摘要",
                     "media": "站点一", "publish_date": "2024-12-31"}]


def test_iqs_query_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """IQS query 上限 1024（智谱是 70），超长截断。"""
    _enable_iqs(monkeypatch)
    seen: dict[str, str] = {}

    def fake_post(payload: dict) -> _Resp:
        seen["query"] = payload["query"]
        return _Resp(200, {"pageItems": []})

    monkeypatch.setattr(web_search, "_http_post_iqs", fake_post)
    assert search_web("字" * 2000, count=3) == []
    assert len(seen["query"]) == 1024


def test_iqs_403_arrears(monkeypatch: pytest.MonkeyPatch) -> None:
    """阿里云顶层错误体 {"code","message"}：Retrieval.Arrears → 提示充值而非等待。"""
    _enable_iqs(monkeypatch)

    def boom(payload: dict) -> _Resp:
        raise _status_err(403, {"code": "Retrieval.Arrears", "message": "Please recharge first."})

    monkeypatch.setattr(web_search, "_http_post_iqs", boom)
    with pytest.raises(WebSearchError, match="余额不足"):
        search_web("q")


def test_iqs_429_daily_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试期日限额（1000 次/天）与真限流是两种 429，文案须区分。"""
    _enable_iqs(monkeypatch)

    def boom(payload: dict) -> _Resp:
        raise _status_err(429, {"code": "Retrieval.TestUserQueryPerDayExceeded", "message": "exceed"})

    monkeypatch.setattr(web_search, "_http_post_iqs", boom)
    with pytest.raises(WebSearchError, match="日限额"):
        search_web("q")


def test_iqs_404_key_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """404 InvalidAccessKeyId.NotFound 按认证失败处理（密钥创建后约 5 分钟生效）。"""
    _enable_iqs(monkeypatch, key="ak-wrong")

    def boom(payload: dict) -> _Resp:
        raise _status_err(404, {"code": "InvalidAccessKeyId.NotFound", "message": "not found"})

    monkeypatch.setattr(web_search, "_http_post_iqs", boom)
    with pytest.raises(WebSearchError, match="IQS_API_KEY"):
        search_web("q")


def test_iqs_403_invalid_apikey_real_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """实测错误体键名是 errorCode/errorMessage（文档写 code/message）：密钥无效须能识别。

    只认文档键名时 _error_detail 返回空码，403 会漏进兜底文案，用户看不到"查密钥"提示。
    """
    _enable_iqs(monkeypatch, key="ak-wrong")

    def boom(payload: dict) -> _Resp:
        raise _status_err(
            403, {"errorCode": "Retrieval.InvalidAPIKey", "errorMessage": "Incorrect APIKey provided."}
        )

    monkeypatch.setattr(web_search, "_http_post_iqs", boom)
    with pytest.raises(WebSearchError, match="IQS_API_KEY"):
        search_web("q")


# ---------------------------------------------------------------------------
# search_web_literature（Searcher 正式工具：学术垂直域 → PaperRecord）
# ---------------------------------------------------------------------------
def _set_research_gate(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    from src.settings import get_settings

    monkeypatch.setitem(get_settings().sources.setdefault("web_search", {}), "enable_for_research", enabled)


def test_web_literature_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """enable_for_research 关闭（默认）→ 不发请求直接 []。"""
    _enable_anysearch(monkeypatch)
    _set_research_gate(monkeypatch, False)
    called: list[dict] = []
    monkeypatch.setattr(
        web_search, "_http_post_anysearch",
        lambda p: called.append(p) or _Resp(200, {"code": 0, "data": {"results": []}}),
    )
    assert web_search.search_web_literature("q") == []
    assert not called


def test_web_literature_zhipu_provider_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 anysearch 服务商（默认 zhipu）不支持学术垂直域 → []。"""
    _set_research_gate(monkeypatch, True)
    assert web_search.search_web_literature("q") == []


def test_web_literature_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """结果映射：arXiv → arxiv:<id>+PDF 主站；doi.org → DOI；其余 → web:<hash>。"""
    _enable_anysearch(monkeypatch)
    _set_research_gate(monkeypatch, True)
    seen: dict[str, Any] = {}

    def fake(payload: dict) -> _Resp:
        seen.update(payload)
        return _Resp(200, {"code": 0, "data": {"results": [
            {"title": " DPO ", "url": "https://arxiv.org/abs/2305.18290", "snippet": " we present DPO "},
            {"title": "DOI 版", "url": "https://doi.org/10.52202/075280-2338", "snippet": "chapter"},
            {"title": "博客", "url": "https://example.com/blog/post", "snippet": " s "},
            {"title": "报告", "url": "https://lab.example.com/report.pdf", "snippet": "r"},
            {"title": "", "url": "", "snippet": ""},  # 无题/无链 → 跳过
        ]}})

    monkeypatch.setattr(web_search, "_http_post_anysearch", fake)
    papers = web_search.search_web_literature("direct preference optimization", max_results=99)
    assert seen["tag"] == "academic.search"
    assert seen["max_results"] == 10  # REST 上限收紧
    assert [p.paper_id for p in papers][:2] == ["arxiv:2305.18290", "10.52202/075280-2338"]
    ax = papers[0]
    assert ax.arxiv_id == "2305.18290" and ax.source_url == "https://arxiv.org/pdf/2305.18290"
    assert ax.paper_type == "preprint" and ax.source_api == "anysearch" and ax.abstract == "we present DPO"
    assert papers[1].doi == "10.52202/075280-2338"
    assert papers[2].paper_id.startswith("web:") and papers[2].paper_type == "other"
    assert papers[3].paper_type == "preprint"  # .pdf 链接


def test_web_literature_error_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP/业务错误 → []（研究管线绝不抛异常，与对话版分层抛错相反）。"""
    _enable_anysearch(monkeypatch)
    _set_research_gate(monkeypatch, True)

    def boom(payload: dict) -> _Resp:
        raise _status_err(429)

    monkeypatch.setattr(web_search, "_http_post_anysearch", boom)
    assert web_search.search_web_literature("q") == []
    monkeypatch.setattr(web_search, "_http_post_anysearch", lambda p: _Resp(200, {"code": -1, "message": "boom"}))
    assert web_search.search_web_literature("q") == []
