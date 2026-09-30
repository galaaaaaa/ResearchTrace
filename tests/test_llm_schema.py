"""_schema_example 的数组元素展开：对象数组曾一律提示 ["<string>"]，模型按错误提示自造字段，
结构化输出校验必败（glm-5.3 实测回 {"id": "#0"}）——Reader 证据批次全灭的主因之一。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.llm import _schema_example


class _Item(BaseModel):
    index: int
    hint: str = ""


class _Outer(BaseModel):
    items: list[_Item] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    single: _Item | None = None


def test_object_array_expanded_to_field_examples() -> None:
    ex = _schema_example(_Outer)
    # 对象数组 → 单元素完整字段示例（字段名可见，模型才知道要输出哪些键）
    assert ex["items"] == [{"index": 0, "hint": "<string>"}]
    assert ex["names"] == ["<string>"]
    assert ex["scores"] == [0.5]
    assert ex["single"] == {"index": 0, "hint": "<string>"}  # 可选对象同样展开
