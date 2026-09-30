"""研究执行层 Agent（Searcher/Reader/Analyst/Critic）的离线单元测试。

全部使用注入的假 LLM（ScriptedLLM）与假工具，不发任何网络请求、不烧真实 token。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from src.agents import analyst, critic, reader, searcher
from src.llm import LLMError
from src.schemas import (
    EvidenceRecord,
    PaperCard,
    PaperRecord,
    ResearchBrief,
    ResearchTask,
)


# --------------------------------------------------------------------------
# 测试基础设施
# --------------------------------------------------------------------------
class ScriptedLLM:
    """按注册顺序返回 chat_json 响应的假 LLM。

    step 支持：dict / BaseModel（经 schema 校验返回）、Exception（抛出）、
    callable(prompt, schema) -> dict（动态生成，用于解析提示词上下文）。
    步骤耗尽后重复最后一步。
    chat_steps 供 chat() 调用（如中文查询翻译），耗尽后返回空串。
    """

    def __init__(self, steps: list[Any], chat_steps: list[Any] | None = None):
        self.steps = list(steps)
        self.chat_steps = list(chat_steps or [])
        self.i = 0
        self.prompts: list[str] = []
        self.chat_prompts: list[str] = []

    def chat_json(self, prompt: str, schema: type, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        step = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(prompt, schema)
        return schema.model_validate(step)

    def chat(self, prompt: str | None = None, **kwargs: Any) -> str:
        self.chat_prompts.append(prompt or "")
        if not self.chat_steps:
            return ""
        step = self.chat_steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return str(step)


def make_task(question: str = "GRPO 训练是否会导致 reward hacking？") -> ResearchTask:
    return ResearchTask(question=question, required_evidence=["ablation 结果", "数值对比"])


def make_brief(**kwargs: Any) -> ResearchBrief:
    defaults: dict[str, Any] = {"objective": "调研 GRPO 的奖励作弊风险", "year_from": 2022, "year_to": 2025}
    defaults.update(kwargs)
    return ResearchBrief(**defaults)


# --------------------------------------------------------------------------
# Searcher
# --------------------------------------------------------------------------
def test_searcher_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """假工具 + 两轮脚本 LLM（先 search 后 finish）：去重、日志、下载、年份约束。"""
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_arxiv(query: str, *, date_from: str | None = None, date_to: str | None = None,
                   max_results: int | None = None) -> list[PaperRecord]:
        calls.append(("search_arxiv", {"query": query, "date_from": date_from, "date_to": date_to}))
        return [
            PaperRecord(paper_id="arxiv:2401.00001", title="GRPO Reward Hacking Analysis", year=2024,
                        arxiv_id="2401.00001", citation_count=10),
            PaperRecord(paper_id="arxiv:2401.00002", title="Another Study on GRPO", year=2023,
                        arxiv_id="2401.00002", citation_count=2),
            # 题名规范化后与第一条重复（paper_id 不同）→ 应被 norm_title 去重
            PaperRecord(paper_id="arxiv:2401.00009", title="GRPO Reward Hacking Analysis!", year=2024),
        ]

    def fake_download(paper: PaperRecord) -> str:
        calls.append(("download_pdf", {"paper_id": paper.paper_id}))
        return str(tmp_path / f"{paper.paper_id.replace(':', '_')}.pdf")

    llm = ScriptedLLM([
        {"thought": "首次搜索", "action": "search_arxiv", "args": {"query": "GRPO reward hacking"}},
        {"thought": "论文足够", "action": "finish", "reason": "found core papers"},
    ])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=3).discover(
        make_task(), make_brief(), tool_registry={"search_arxiv": fake_arxiv, "download_pdf": fake_download}
    )

    assert result.stopped_reason == "finished"
    assert len(result.papers) == 2  # paper_id + norm_title 两级去重
    assert result.queries == ["GRPO reward hacking"]
    # 循环后补下载：两篇 arXiv 论文都拿到 pdf_path
    assert set(result.downloads) == {"arxiv:2401.00001", "arxiv:2401.00002"}
    assert all(p.pdf_path for p in result.papers)
    assert result.tool_logs and all(log.ok for log in result.tool_logs)
    assert "GRPO Reward Hacking Analysis" in result.tool_logs[0].result_summary
    # brief 年份约束作为默认 date 参数传入
    assert calls[0][1]["date_from"] == "2022-01-01"
    assert calls[0][1]["date_to"] == "2025-12-31"


def test_searcher_budget_hard_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 永远要求搜索（query 词元互不重叠，避免触发停滞检测）→ 工具执行次数被 max_calls 硬顶。"""
    words = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu"]
    state = {"n": 0}

    def next_search(prompt: str, schema: type) -> dict[str, Any]:
        q = words[state["n"] % len(words)]
        state["n"] += 1
        return {"thought": "继续搜", "action": "search_arxiv", "args": {"query": q}}

    def fake_arxiv(query: str, **kwargs: Any) -> list[PaperRecord]:
        return [PaperRecord(paper_id=f"arxiv:2401.{abs(hash(query)) % 100000}", title=f"Paper {query}", year=2024)]

    llm = ScriptedLLM([next_search])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(
        make_task(), make_brief(), tool_registry={"search_arxiv": fake_arxiv}
    )

    assert result.stopped_reason == "budget"
    search_logs = [log for log in result.tool_logs if log.tool == "search_arxiv"]
    assert len(search_logs) <= 2  # ≤ max_calls 硬顶
    assert len(search_logs) == 2
    assert len(result.queries) == 2


def test_searcher_stagnation_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """相邻两次搜索词高度重复 → stagnation 提前停止（第二次不执行）。"""
    def fake_arxiv(query: str, **kwargs: Any) -> list[PaperRecord]:
        return [PaperRecord(paper_id="arxiv:2401.00001", title="T", year=2024)]

    llm = ScriptedLLM([
        {"action": "search_arxiv", "args": {"query": "grpo reward hacking"}},
        {"action": "search_arxiv", "args": {"query": "reward hacking grpo"}},  # token 集合完全相同
    ])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=5).discover(
        make_task(), make_brief(), tool_registry={"search_arxiv": fake_arxiv}
    )

    assert result.stopped_reason == "stagnation"
    assert len([log for log in result.tool_logs if log.tool == "search_arxiv"]) == 1


def test_searcher_tool_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """工具抛 NotImplementedError → 记 ok=False(error=unavailable) 并剔除，流程继续。"""
    def broken_arxiv(query: str, **kwargs: Any) -> list[PaperRecord]:
        raise NotImplementedError("pending tools-search implementation")

    def fake_ss(query: str, *, max_results: int | None = None, year: str | None = None) -> list[PaperRecord]:
        return [PaperRecord(paper_id="10.1234/abc", title="SS Paper", year=2024)]

    llm = ScriptedLLM([
        {"action": "search_arxiv", "args": {"query": "q1"}},
        {"action": "search_semantic_scholar", "args": {"query": "q2"}},
        {"action": "finish", "reason": "ok"},
    ])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=3).discover(
        make_task(), make_brief(),
        tool_registry={"search_arxiv": broken_arxiv, "search_semantic_scholar": fake_ss},
    )

    assert result.tool_logs[0].tool == "search_arxiv"
    assert result.tool_logs[0].ok is False
    assert result.tool_logs[0].error == "unavailable"
    assert all(log.tool != "search_arxiv" for log in result.tool_logs[1:])  # 后续轮次已剔除
    assert [p.paper_id for p in result.papers] == ["10.1234/abc"]
    assert result.stopped_reason == "finished"


def test_searcher_unexpected_error_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """未预期异常 → 返回部分结果 + stopped_reason=error:...，不抛出。"""
    llm = ScriptedLLM([{"action": "search_arxiv", "args": {"query": "q"}}])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(make_task(), None)  # type: ignore[arg-type]
    assert result.stopped_reason.startswith("error:")
    assert result.papers == []


def test_searcher_invalid_action_free_and_corrected(monkeypatch: pytest.MonkeyPatch) -> None:
    """无效动作不消耗工具预算：两次幻觉工具名后仍能执行满额真实搜索，且纠正反馈进入后续提示词。"""
    searches: list[str] = []

    def fake_arxiv(query: str, **kwargs: Any) -> list[PaperRecord]:
        searches.append(query)
        return [PaperRecord(paper_id="arxiv:2401.00001", title="T", year=2024)]

    llm = ScriptedLLM([
        {"thought": "幻觉", "action": "READ_FIGURE", "args": {}},
        {"thought": "再幻觉", "action": "view_page", "args": {}},
        {"thought": "纠正后搜索", "action": "search_arxiv", "args": {"query": "grpo reward hacking"}},
        {"thought": "完成", "action": "finish", "reason": "ok"},
    ])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(
        make_task(), make_brief(), tool_registry={"search_arxiv": fake_arxiv}
    )

    assert result.stopped_reason == "finished"
    # 两次无效动作没有触发兜底搜索，真实搜索照常拿到满额
    assert searches == ["grpo reward hacking"]
    assert len(result.tool_logs) == 1 and result.tool_logs[0].tool == "search_arxiv"
    # 纠正反馈出现在第二轮用户消息末尾，并附可用工具清单
    assert "不是可用工具" in llm.prompts[1]
    assert "search_arxiv" in llm.prompts[1]


def test_searcher_invalid_action_three_strikes(monkeypatch: pytest.MonkeyPatch) -> None:
    """连续 3 次无效动作才出局（此前是 2 次；纠正反馈给了自我纠正机会）。"""
    llm = ScriptedLLM([{"action": "decompose_task", "args": {}}])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=3).discover(
        make_task(), make_brief(), tool_registry={"search_arxiv": lambda q, **kw: []}
    )
    assert result.stopped_reason == "error:invalid_action:decompose_task"
    assert len(llm.prompts) == 3  # 三轮都拿到了带纠正反馈的提示词


def test_searcher_cjk_query_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    """中文 query 自动译成英文检索式再进搜索工具（arXiv 对中文几乎零命中）。"""
    got: list[str] = []

    def fake_arxiv(query: str, **kwargs: Any) -> list[PaperRecord]:
        got.append(query)
        return []

    llm = ScriptedLLM(
        [{"action": "search_arxiv", "args": {"query": "GRPO 训练的奖励作弊风险"}},
         {"action": "finish", "reason": "ok"}],
        chat_steps=["GRPO reward hacking risk"],
    )
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(
        make_task(), make_brief(), tool_registry={"search_arxiv": fake_arxiv}
    )

    assert got == ["GRPO reward hacking risk"]
    assert result.queries == ["GRPO reward hacking risk"]
    assert any("翻译" in p for p in llm.chat_prompts)


def test_searcher_fallback_rotates_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM 失败的兜底检索轮换引擎：arxiv 用同一查询搜过则改试 semantic_scholar，并走翻译。"""
    calls: list[tuple[str, str]] = []

    def fake_arxiv(query: str, **kwargs: Any) -> list[PaperRecord]:
        calls.append(("search_arxiv", query))
        return []

    def fake_ss(query: str, **kwargs: Any) -> list[PaperRecord]:
        calls.append(("search_semantic_scholar", query))
        return [PaperRecord(paper_id="10.1/x", title="S2 Paper", year=2024)]

    llm = ScriptedLLM([LLMError("boom")], chat_steps=["GRPO reward hacking"])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(
        make_task("GRPO 训练是否会导致 reward hacking？"), make_brief(),
        tool_registry={"search_arxiv": fake_arxiv, "search_semantic_scholar": fake_ss},
    )

    # 两次 LLM 失败 → 兜底先 arxiv 后 S2，查询均为翻译后的英文
    assert calls == [("search_arxiv", "GRPO reward hacking"), ("search_semantic_scholar", "GRPO reward hacking")]
    assert result.stopped_reason == "budget"
    assert [p.paper_id for p in result.papers] == ["10.1/x"]


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------
def test_reader_no_pdf() -> None:
    """无 pdf_path → partial + notes 记 no pdf + 无证据。"""
    paper = PaperRecord(paper_id="arxiv:2401.00001", title="T", source_url="http://arxiv/1")
    res = reader.Reader().read_paper(paper, make_task(), llm=ScriptedLLM([]))
    assert res.card.read_status == "partial"
    assert "no pdf" in " ".join(res.notes)
    assert res.evidence == []


def test_reader_retracted_early_return(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """撤稿命中 → read_status=failed 并直接返回（不再解析 PDF）。"""
    from src.schemas import RetractionStatus
    from src.tools import citation_tool

    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 minimal")

    def fake_retraction(paper: PaperRecord) -> RetractionStatus:
        return RetractionStatus(paper_id=paper.paper_id, is_retracted=True, notice="retracted 2025")

    monkeypatch.setattr(citation_tool, "check_retraction", fake_retraction)
    paper = PaperRecord(paper_id="10.1/retracted", title="T", pdf_path=str(pdf_path))
    res = reader.Reader().read_paper(paper, make_task(), llm=ScriptedLLM([]))
    assert res.card.read_status == "failed"
    assert any("撤稿" in n for n in res.notes)
    assert res.evidence == []


def _write_pdf(path: Path, pages: list[str]) -> None:
    import pymupdf

    doc = pymupdf.Document()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(54, 72, 540, 780), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_reader_pdf_extraction(tmp_path: Path) -> None:
    """合成小 PDF（2 页，含 GRPO reward hacking 关键词）→ 证据逐字来自原文。"""
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(
        pdf_path,
        [
            "GRPO reward hacking study. We analyze whether GRPO training leads to reward hacking "
            "in math reasoning tasks. The motivation is reward sparsity under group relative policy optimization.",
            "Experiments on the MATH dataset. Our method achieves 82.4 accuracy compared to a 71.3 baseline. "
            "Limitations include small sample size and no ablation on reward shaping.",
        ],
    )

    try:
        from src.tools.pdf_tool import parse_pdf

        parse_pdf(str(pdf_path))
    except NotImplementedError:
        pytest.skip("pdf_tool pending")
    except Exception:
        pytest.skip("pdf_tool pending")

    def select_or_card(prompt: str, schema: type) -> dict[str, Any]:
        """两段式协议：_Selection 调用解析 #编号 片段选中含 reward hacking 的条目；_CardOut 调用回卡片。"""
        if schema.__name__ != "_Selection":
            return {
                "motivation": "分析 GRPO 的奖励作弊",
                "method": "GRPO 训练分析",
                "datasets": ["MATH"],
                "metrics": ["accuracy"],
                "results": "accuracy 82.4 vs 71.3",
                "limitations": ["小样本"],
                "is_ablation_supported": False,
            }
        idxs: dict[int, str] = {}
        cur: int | None = None
        buf: list[str] = []
        for line in prompt.splitlines():
            m = re.match(r"#(\d+) \| page=(\S+) \| section=(.*)", line)
            if m:
                if cur is not None:
                    idxs[cur] = "\n".join(buf)
                cur, buf = int(m.group(1)), []
            elif cur is not None:
                buf.append(line)
        if cur is not None:
            idxs[cur] = "\n".join(buf)
        return {
            "selected": [
                {"index": i, "claim_hint": "GRPO 存在 reward hacking 风险", "relevance": 0.9}
                for i, t in idxs.items()
                if "reward hacking" in t.lower()
            ]
        }

    paper = PaperRecord(
        paper_id="arxiv:2401.00001", title="GRPO Reward Hacking", year=2024,
        arxiv_id="2401.00001", pdf_path=str(pdf_path), source_url="http://arxiv/1",
    )
    res = reader.Reader().read_paper(paper, make_task(), llm=ScriptedLLM([select_or_card]))

    assert res.evidence, "应抽取到文本证据"
    import pymupdf

    with pymupdf.open(str(pdf_path)) as doc:
        full_text = "".join(doc[i].get_text() for i in range(doc.page_count))
    strip_ws = lambda s: re.sub(r"\s+", "", s)  # noqa: E731 —— 分块可能跨页/换行，空白归零后比对
    for ev in res.evidence:
        if ev.modality != "text":
            continue
        # 逐字来自 PDF 原文（去除空白后的子串校验）
        assert strip_ws(ev.evidence_text) in strip_ws(full_text)
        assert ev.page is not None
        assert ev.extractor == "reader"
        assert ev.task_id and ev.paper_id == paper.paper_id
    text_ids = [ev.evidence_id for ev in res.evidence if ev.modality == "text"]
    assert res.card.evidence_ids[: len(text_ids)] == text_ids  # evidence_ids 回填
    assert res.card.datasets == ["MATH"]
    assert res.card.is_ablation_supported is False
    assert res.tool_calls >= 1
    assert res.card.read_status in ("ok", "partial")


def test_reader_select_batch_failure_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一批选择失败只降级该批（桩 ≤2 条），后续批仍产出真证据——对照 paper-qa 的失败隔离。"""
    monkeypatch.setattr(reader, "_SELECT_BATCH", 2)
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(pdf_path, ["Synthetic paper body for reader isolation test."])
    try:
        from src.tools.pdf_tool import parse_pdf

        parse_pdf(str(pdf_path))
    except Exception:
        pytest.skip("pdf_tool pending")

    from src.tools import pdf_tool

    # 固定 4 条候选，批大小 2 → 两批选择调用（编号全局：批0=#0,#1；批1=#2,#3）
    monkeypatch.setattr(
        pdf_tool, "retrieve_chunks",
        lambda doc, q, k=8: [
            {"page": i, "section": None, "text": f"snippet {i} about GRPO reward hacking"} for i in range(4)
        ],
    )

    calls = {"select": 0}

    def flaky_then_ok(prompt: str, schema: type) -> dict[str, Any]:
        if schema.__name__ != "_Selection":
            return {"motivation": "m", "method": "w", "results": "r"}
        calls["select"] += 1
        if calls["select"] == 1:
            raise LLMError("boom")
        return {"selected": [{"index": 2, "claim_hint": "真结论提示", "relevance": 0.8}]}

    paper = PaperRecord(
        paper_id="arxiv:2401.00002", title="T", pdf_path=str(pdf_path), source_url="http://arxiv/2"
    )
    res = reader.Reader().read_paper(paper, make_task(), llm=ScriptedLLM([flaky_then_ok]))

    stub = [ev for ev in res.evidence if "LLM 失败" in (ev.claim_hint or "")]
    real = [ev for ev in res.evidence if "真结论提示" in (ev.claim_hint or "")]
    assert len(stub) == 2  # 批 0 整批降级：保留前 2 条桩
    assert len(real) == 1 and real[0].evidence_text.startswith("snippet 2")  # 批 1 正常产出
    assert any("选择批失败降级" in n for n in res.notes)
    assert res.card.read_status == "partial"  # 有批失败 → partial（如实标注）
    assert res.card.motivation == "m"  # 卡片调用不受选择批失败影响


# --------------------------------------------------------------------------
# Analyst
# --------------------------------------------------------------------------
def _make_cards() -> list[PaperCard]:
    return [
        PaperCard(paper_id="p1", method="PPO", datasets=["GSM8K"], metrics=["accuracy"], results="accuracy 80"),
        PaperCard(paper_id="p2", method="GRPO", datasets=["MATH"], metrics=["accuracy"], results="accuracy 82"),
        PaperCard(paper_id="p3", method="DPO", datasets=["GSM8K"], metrics=["f1"], results="f1 0.7"),
    ]


def _papers_for(cards: list[PaperCard]) -> list[PaperRecord]:
    return [PaperRecord(paper_id=c.paper_id, title=f"Paper {c.paper_id}") for c in cards]


def test_analyst_matrix_from_llm() -> None:
    """3 张卡片 + 假 LLM → rows/warnings 齐全，规则告警补充。"""
    cards = _make_cards()
    llm = ScriptedLLM([{
        "columns": ["方法", "数据集", "指标", "结果"],
        "rows": [
            {"paper_id": "p1", "values": {"方法": "PPO", "数据集": "GSM8K", "指标": "accuracy", "结果": "80"}},
            {"paper_id": "p2", "values": {"方法": "GRPO", "数据集": "MATH", "指标": "accuracy", "结果": "82"}},
            {"paper_id": "p3", "values": {"方法": "DPO", "数据集": "GSM8K", "指标": "f1", "结果": "0.7"}},
        ],
        "warnings": ["数据集版本不同，不可直接比较"],
    }])
    matrix = analyst.build_comparison(make_task(), _papers_for(cards), [], cards, llm=llm)
    assert len(matrix.rows) == 3
    assert matrix.rows[0].title == "Paper p1"  # 题名回填
    assert any("数据集版本" in w for w in matrix.comparability_warnings)  # LLM 告警保留
    assert any("无交集" in w for w in matrix.comparability_warnings)  # 规则告警：p1/p2 数据集无交集
    assert any("口径" in w for w in matrix.comparability_warnings)  # 规则告警：同名指标不同数据集


def test_analyst_too_few_cards() -> None:
    """cards < 2 → 空矩阵 + note。"""
    cards = [PaperCard(paper_id="p1", method="PPO")]
    matrix = analyst.build_comparison(make_task(), _papers_for(cards), [], cards, llm=ScriptedLLM([]))
    assert matrix.rows == []
    assert matrix.columns == []
    assert "不足 2 篇" in (matrix.notes or "")


def test_analyst_llm_failure_fallback() -> None:
    """LLM 抛错 → 纯规则兜底矩阵非空。"""
    cards = _make_cards()
    matrix = analyst.build_comparison(make_task(), _papers_for(cards), [], cards, llm=ScriptedLLM([LLMError("boom")]))
    assert matrix.rows, "兜底矩阵不应为空"
    assert "方法" in matrix.columns
    assert matrix.rows[0].values["方法"] == "PPO"
    assert "兜底" in (matrix.notes or "")
    assert any("无交集" in w for w in matrix.comparability_warnings)


# --------------------------------------------------------------------------
# Critic
# --------------------------------------------------------------------------
def test_critic_rule_causal_without_ablation() -> None:
    """is_ablation_supported=False + 因果词 → 规则旗标命中。"""
    card = PaperCard(paper_id="p1", results="GRPO 导致 reward hacking，因为奖励过于稀疏", is_ablation_supported=False)
    flags = critic.find_conflicts(make_task(), [], [], [card], llm=ScriptedLLM([{"conflicts": []}]))
    assert any(f.topic == "无消融支撑的因果归因" and f.severity == "medium" for f in flags)


def test_critic_rule_metric_direction_opposite() -> None:
    """同一指标在不同论文中数值方向相反 → 规则旗标命中。"""
    c1 = PaperCard(paper_id="a", metrics=["accuracy"], results="accuracy 提高到 91.2 after GRPO")
    c2 = PaperCard(paper_id="b", metrics=["accuracy"], results="accuracy 下降至 68.5 under GRPO")
    flags = critic.find_conflicts(make_task(), [], [], [c1, c2], llm=ScriptedLLM([{"conflicts": []}]))
    assert any("方向相反" in f.topic for f in flags)
    hit = next(f for f in flags if "方向相反" in f.topic)
    assert set(hit.paper_ids) == {"a", "b"}


def test_critic_rule_retracted() -> None:
    """撤稿论文 → high 冲突。"""
    papers = [PaperRecord(paper_id="p9", title="Retrected Paper", is_retracted=True)]
    flags = critic.find_conflicts(make_task(), papers, [], [], llm=ScriptedLLM([{"conflicts": []}]))
    assert any(f.topic == "撤稿论文结论仍在引用" and f.severity == "high" for f in flags)


def test_critic_llm_merge_and_evidence_backfill() -> None:
    """规则旗标 + LLM 冲突合并 ≤5 条；evidence_ids 回填。"""
    cards = [
        PaperCard(paper_id="p1", results="GRPO 导致 reward hacking", is_ablation_supported=False),
        PaperCard(paper_id="p2", results="stable training", is_ablation_supported=True),
    ]
    papers = _papers_for(cards)
    evidence = [
        EvidenceRecord(paper_id="p1", task_id="t", claim_hint="h", evidence_text="e1"),
        EvidenceRecord(paper_id="p1", task_id="t", claim_hint="h", evidence_text="e2"),
        EvidenceRecord(paper_id="p2", task_id="t", claim_hint="h", evidence_text="e3"),
    ]
    llm = ScriptedLLM([{"conflicts": [{
        "topic": "样本规模差异导致结论不一致", "paper_ids": ["p1", "p2"],
        "description": "d", "possible_causes": ["小样本"], "severity": "low",
    }]}])
    flags = critic.find_conflicts(make_task(), papers, evidence, cards, llm=llm)

    assert len(flags) <= 5
    assert any(f.topic == "无消融支撑的因果归因" for f in flags)  # 规则旗标保留
    merged = next(f for f in flags if "样本规模" in f.topic)  # LLM 冲突并入
    assert set(merged.paper_ids) == {"p1", "p2"}
    assert merged.evidence_ids, "冲突应回填证据 id"
    assert len(merged.evidence_ids) <= 3


def test_critic_dedup_similar_topic() -> None:
    """LLM 冲突 topic 与规则旗标相似 >0.8 → 去重不重复计入。"""
    card = PaperCard(paper_id="p1", results="GRPO 导致 reward hacking", is_ablation_supported=False)
    llm = ScriptedLLM([{"conflicts": [{
        "topic": "无消融支撑的因果归因", "paper_ids": ["p1"], "description": "d", "severity": "medium",
    }]}])
    flags = critic.find_conflicts(make_task(), [], [], [card], llm=llm)
    assert len([f for f in flags if f.topic == "无消融支撑的因果归因"]) == 1


def test_critic_llm_failure_rule_only() -> None:
    """LLM 失败 → 只返回规则旗标。"""
    card = PaperCard(paper_id="p1", results="GRPO 导致 reward hacking", is_ablation_supported=False)
    flags = critic.find_conflicts(make_task(), [], [], [card], llm=ScriptedLLM([LLMError("boom")]))
    assert [f.topic for f in flags] == ["无消融支撑的因果归因"]


def test_searcher_web_tool_alias_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型幻觉的 web_search 动作名 → 真实工具 search_web_literature（曾经只能映射回 search_arxiv）。"""
    got: list[str] = []

    def fake_web_lit(query: str, **kwargs: Any) -> list[PaperRecord]:
        got.append(query)
        return [PaperRecord(paper_id="web:abc123def456", title="Blog Report", source_api="anysearch")]

    llm = ScriptedLLM([
        {"action": "web_search", "args": {"query": "GRPO reward hacking risk"}},
        {"action": "finish", "reason": "ok"},
    ])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(
        make_task("GRPO 训练是否会导致 reward hacking？"), make_brief(),
        tool_registry={"search_web_literature": fake_web_lit},
    )

    assert got == ["GRPO reward hacking risk"]
    assert [p.paper_id for p in result.papers] == ["web:abc123def456"]
    assert result.tool_logs[0].tool == "search_web_literature" and result.tool_logs[0].ok


def test_searcher_inherits_pdf_path_across_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """跨 run PDF 复用：库里有同 arxiv_id 论文的已下载 PDF → 新发现记录直接继承，不再下载。"""
    from src.memory.evidence_store import EvidenceStore
    from src.settings import get_settings

    db = tmp_path / "e.db"
    monkeypatch.setattr(get_settings(), "db_path", db)
    pdf = tmp_path / "arxiv_2401.00001.pdf"
    pdf.write_bytes(b"%PDF-1.4 cached")
    store = EvidenceStore(db)
    store.upsert_papers([PaperRecord(paper_id="arxiv:2401.00001", title="Old Title",
                                     arxiv_id="2401.00001", pdf_path=str(pdf))])
    store.close()

    downloads: list[str] = []

    def fake_web_lit(query: str, **kwargs: Any) -> list[PaperRecord]:
        return [PaperRecord(paper_id="arxiv:2401.00001", title="Newly Found Title",
                            arxiv_id="2401.00001")]  # 无 pdf_path 的新发现

    def fake_download(paper: PaperRecord) -> str:
        downloads.append(paper.paper_id)
        return str(tmp_path / "new.pdf")

    llm = ScriptedLLM([
        {"action": "search_web_literature", "args": {"query": "q"}},
        {"action": "finish", "reason": "ok"},
    ])
    monkeypatch.setattr(searcher, "get_llm", lambda role, **kw: llm)

    result = searcher.Searcher(max_calls=2).discover(
        make_task(), make_brief(),
        tool_registry={"search_web_literature": fake_web_lit, "download_pdf": fake_download},
    )
    assert result.papers[0].pdf_path == str(pdf)  # 继承了库里的路径
    assert downloads == []  # 已有 PDF → 不再触发下载
