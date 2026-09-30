"""LLM 接入层：多 provider 工厂 + 结构化 JSON 输出 + token 预算 + 离线 FakeLLM。

- provider：anthropic 兼容 / openai 兼容 / fake（测试）；
- chat_json：提示词请求 JSON → 防御性解析（围栏/平衡括号）→ Pydantic 校验 → 一次修复重试；
- TokenBudget：全局线程安全计数，超限后拒绝继续调用（配合 budgets.yaml）。
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from types import UnionType
from typing import Any, Callable, Literal, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .settings import get_settings
from .tracing import get_active_tracer
from .utils import estimate_tokens, extract_json_block, truncate

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_MARKERS = ("rate", "limit", "timeout", "timed out", "overload", "connection", "unavailable", "429", "500", "502", "503", "504")


class LLMError(RuntimeError):
    pass


class JSONValidationError(LLMError):
    """模型输出无法解析为目标 Schema（含原始文本与校验错误）。"""

    def __init__(self, message: str, raw_text: str = "", errors: str = ""):
        super().__init__(message)
        self.raw_text = raw_text
        self.errors = errors


# --------------------------------------------------------------------------
# Token 预算
# --------------------------------------------------------------------------
class TokenBudget:
    """全局线程安全预算计数。并行 Researcher 共享同一实例。"""

    def __init__(
        self,
        max_calls: int = 500,
        max_input_tokens: int = 6_000_000,
        max_output_tokens: int = 1_500_000,
    ):
        self.max_calls = max_calls
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def spend(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    @property
    def exceeded(self) -> bool:
        return (
            self.calls >= self.max_calls
            or self.input_tokens >= self.max_input_tokens
            or self.output_tokens >= self.max_output_tokens
        )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "max_calls": self.max_calls,
                "exceeded": self.exceeded,
            }


_budget_lock = threading.RLock()  # 可重入：reset_global_budget 内部复用 get_global_budget
_global_budget: TokenBudget | None = None


def get_global_budget() -> TokenBudget:
    global _global_budget
    with _budget_lock:
        if _global_budget is None:
            s = get_settings()
            _global_budget = TokenBudget(
                max_calls=int(s.budget("llm_max_calls", 500)),
                max_input_tokens=int(s.budget("llm_max_input_tokens", 6_000_000)),
                max_output_tokens=int(s.budget("llm_max_output_tokens", 1_500_000)),
            )
        return _global_budget


def reset_global_budget() -> TokenBudget:
    """测试 / 新 run 用：重置并返回新预算。"""
    global _global_budget
    with _budget_lock:
        _global_budget = None
        return get_global_budget()


# --------------------------------------------------------------------------
# Fake 后端（离线测试）
# --------------------------------------------------------------------------
class FakeBackend:
    """注册式假模型：测试在 system prompt 中放 #FAKE:<name> 标记选择响应。"""

    _registry: dict[str, Callable[[str, Type[BaseModel] | None], Any] | str] = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, name: str, response: Callable[[str, Type[BaseModel] | None], Any] | str | dict) -> None:
        with cls._lock:
            cls._registry[name] = response

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._registry.clear()

    def respond(self, system: str | None, prompt: str, schema: Type[BaseModel] | None) -> Any:
        import re

        marker = None
        for m in re.findall(r"#FAKE:([A-Za-z0-9_]+)", system or ""):
            marker = m
        if marker and marker in self._registry:
            entry = self._registry[marker]
            if callable(entry):
                return entry(prompt, schema)
            if isinstance(entry, BaseModel):
                return entry
            if isinstance(entry, str):
                return entry
            return json.dumps(entry, ensure_ascii=False)
        # 无注册时的兜底：按 Schema 生成最小合法实例
        if schema is not None:
            return json.dumps(_default_for_schema(schema), ensure_ascii=False)
        return "（fake 模型响应）"


def _default_for_schema(schema: Type[BaseModel]) -> dict:
    """依据字段类型/默认值构造最小合法 JSON（FakeLLM 兜底）。"""
    from pydantic_core import PydanticUndefined

    out: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        default = field.get_default()
        if default is not None and default is not PydanticUndefined and not callable(default):
            out[name] = _jsonable(default)
            continue
        ann = field.annotation
        origin = getattr(ann, "__origin__", None)
        args = getattr(ann, "__args__", None)
        if ann is str:
            out[name] = name
        elif ann is int:
            out[name] = 1
        elif ann is float:
            out[name] = 0.5
        elif ann is bool:
            out[name] = False
        elif origin is Literal:
            first = args[0] if args else None
            out[name] = first if isinstance(first, (str, int, float, bool)) else None
        elif origin in (list,):
            out[name] = []
        elif origin in (dict,):
            out[name] = {}
        elif origin is UnionType or str(ann).startswith("typing.Optional") or str(ann).startswith("typing.Union"):
            non_none = [a for a in (args or []) if a is not type(None)]
            if non_none and isinstance(non_none[0], type) and issubclass(non_none[0], BaseModel):
                out[name] = _default_for_schema(non_none[0])
            elif non_none and getattr(non_none[0], "__origin__", None) is Literal:
                lit_args = getattr(non_none[0], "__args__", [])
                out[name] = lit_args[0] if lit_args else None
            else:
                out[name] = None
        elif isinstance(ann, type) and issubclass(ann, BaseModel):
            out[name] = _default_for_schema(ann)
        else:
            out[name] = None
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------
# 客户端缓存与构建
# --------------------------------------------------------------------------
_client_cache: dict[tuple, Any] = {}
_client_lock = threading.Lock()
_extra_body_ok: dict[str, bool] = {}


def _build_client(provider_cfg: dict, model: str, timeout_s: float):
    ptype = provider_cfg.get("type", "fake")
    if ptype == "fake":
        return FakeBackend()
    if ptype == "anthropic":
        import os

        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get(provider_cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN"), "")
        base_url = provider_cfg.get("base_url") or None
        return ChatAnthropic(
            model=model or "glm-5.3",
            api_key=api_key or "not-set",
            base_url=base_url,
            timeout=timeout_s,
            max_retries=1,
        )
    if ptype == "openai":
        import os

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=os.environ.get(provider_cfg.get("api_key_env", "OPENAI_API_KEY"), "not-set"),
            base_url=provider_cfg.get("base_url") or None,
            timeout=timeout_s,
            max_retries=1,
        )
    raise LLMError(f"未知 provider 类型: {ptype}")


def _get_client(provider_name: str, provider_cfg: dict, model: str, timeout_s: float):
    key = (provider_name, model, timeout_s)
    with _client_lock:
        if key not in _client_cache:
            _client_cache[key] = _build_client(provider_cfg, model, timeout_s)
        return _client_cache[key]


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {str(exc)[:300]}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _extract_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _extract_usage(message: Any) -> tuple[int, int]:
    um = getattr(message, "usage_metadata", None)
    if isinstance(um, dict) and um.get("input_tokens") is not None:
        return int(um.get("input_tokens") or 0), int(um.get("output_tokens") or 0)
    rm = getattr(message, "response_metadata", {})
    usage = (rm or {}).get("usage") or {}
    return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0), int(
        usage.get("output_tokens") or usage.get("completion_tokens") or 0
    )


def _image_blocks(images: list[tuple[bytes, str]], ptype: str) -> list[dict]:
    blocks = []
    for data, media_type in images:
        b64 = base64.b64encode(data).decode("ascii")
        if ptype == "anthropic":
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}})
        else:
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}})
    return blocks


# --------------------------------------------------------------------------
# LLMClient
# --------------------------------------------------------------------------
class LLMClient:
    """面向角色的客户端。所有 Agent 通过 role 获取，不直接触碰 provider 细节。"""

    def __init__(self, role: str, *, budget: TokenBudget | None = None):
        s = get_settings()
        self.role = role
        self.cfg = s.role_config(role)
        self.provider_name = self.cfg["provider"]
        self.provider_cfg = s.provider_config(self.provider_name)
        self.ptype = self.provider_cfg.get("type", "fake")
        # 防御性归一：剥掉模型名尾部的会话标记（如 Claude Code 的 "glm-5.3[1m]"）。
        # 该标记是 harness 约定不属于任何 API——曾原样抄进 .env 导致智谱端点 1211
        # "模型不存在"，所有 LLM 调用 400、Reader 证据批次全灭降级为桩。
        self.model = re.split(r"\[", self.cfg.get("model") or "", maxsplit=1)[0].strip()
        self.budget = budget or get_global_budget()
        self.timeout_s = float(self.provider_cfg.get("request_timeout_s", 180))

    # ---- 基础调用 ----
    def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict] | None = None,
        system: str | None = None,
        images: list[tuple[bytes, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "",
    ) -> str:
        if self.budget.exceeded:
            raise LLMError(f"LLM 预算已耗尽: {self.budget.snapshot()}")

        tracer = get_active_tracer()
        t0 = time.time()

        if self.ptype == "fake":
            backend = _get_client(self.provider_name, self.provider_cfg, self.model, self.timeout_s)
            raw = backend.respond(system, prompt or json.dumps(messages, ensure_ascii=False), None)
            answer = raw if isinstance(raw, str) else json.dumps(_jsonable(raw), ensure_ascii=False)
            self.budget.spend(estimate_tokens(prompt or ""), estimate_tokens(answer))
            tracer.event("llm_call", role=self.role, model="fake", label=label, output_tokens=estimate_tokens(answer))
            return answer

        client = _get_client(self.provider_name, self.provider_cfg, self.model, self.timeout_s)
        msgs: list[dict] = list(messages or [])
        if prompt is not None:
            content: Any = prompt
            if images:
                blocks = _image_blocks(images, self.ptype)
                text_part = [{"type": "text", "text": prompt}] if self.ptype == "anthropic" else [{"type": "text", "text": prompt}]
                content = blocks + text_part
            msgs = msgs + [{"role": "user", "content": content}]
        kwargs = dict(
            temperature=self.cfg.get("temperature", 0.2) if temperature is None else temperature,
            max_tokens=int(self.cfg.get("max_tokens", 4096) if max_tokens is None else max_tokens),
        )

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                bound = client
                if self.ptype == "anthropic" and _extra_body_ok.get(self.provider_name, True):
                    try:
                        bound = client.bind(extra_body={"thinking": {"type": "disabled"}})
                    except Exception:
                        _extra_body_ok[self.provider_name] = False
                        bound = client
                message = bound.invoke(msgs, **kwargs)
                answer = _extract_text(message)
                in_tok, out_tok = _extract_usage(message)
                if in_tok == 0:
                    in_tok, out_tok = estimate_tokens(json.dumps(msgs, ensure_ascii=False)), estimate_tokens(answer)
                self.budget.spend(in_tok, out_tok)
                tracer.event(
                    "llm_call",
                    role=self.role,
                    model=self.model,
                    label=label,
                    attempt=attempt,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    duration_ms=int((time.time() - t0) * 1000),
                )
                if not answer:
                    raise LLMError(f"模型返回空文本 (role={self.role})")
                return answer
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _extra_body_ok.get(self.provider_name, True) and self.ptype == "anthropic":
                    _extra_body_ok[self.provider_name] = False  # 可能是 extra_body 不被支持
                if not _is_retryable(exc) or attempt == 2:
                    raise LLMError(f"LLM 调用失败 (role={self.role}): {type(exc).__name__}: {exc}") from exc
                time.sleep(2 ** (attempt + 1))
        raise LLMError(f"LLM 调用失败 (role={self.role}): {last_exc}")

    # ---- 结构化输出 ----
    def chat_json(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
        images: list[tuple[bytes, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        label: str = "",
        fake_value: Any = None,
    ) -> T:
        json_system = (system or "") + "\n你必须只输出一个合法 JSON 对象（UTF-8，无注释、无多余文本、不用代码围栏），字段严格符合给定 Schema。"
        schema_hint = json.dumps(_schema_example(schema), ensure_ascii=False)

        if self.ptype == "fake":
            backend = _get_client(self.provider_name, self.provider_cfg, self.model, self.timeout_s)
            marker_match = None
            import re

            for m in re.findall(r"#FAKE:([A-Za-z0-9_]+)", json_system):
                marker_match = m
            if fake_value is not None:
                data = fake_value
                if isinstance(data, BaseModel):
                    return data if isinstance(data, schema) else schema.model_validate(data.model_dump())
                return schema.model_validate(data)
            raw = backend.respond(json_system, prompt, schema)
            data = raw if not isinstance(raw, str) else extract_json_block(raw)
            return schema.model_validate(data)

        full_prompt = f"{prompt}\n\n输出 JSON Schema 示例（键与类型必须一致，值可按内容填充）：\n{schema_hint}"
        raw = self.chat(
            full_prompt,
            system=json_system,
            images=images,
            temperature=0.0 if temperature is None else temperature,
            max_tokens=max_tokens,
            label=label or f"{self.role}:json",
        )
        data = extract_json_block(raw)
        try:
            if data is None:
                raise ValueError("未能从输出中提取 JSON")
            return schema.model_validate(data)
        except (ValidationError, ValueError) as first_err:
            # 一次修复重试：把错误信息发回给模型
            repair_prompt = (
                f"你之前的输出不是合法的目标 JSON。错误：{truncate(str(first_err), 500)}\n"
                f"之前的输出：\n{truncate(raw, 2000)}\n\n"
                f"请重新输出，只输出一个合法 JSON 对象，Schema 示例：\n{schema_hint}\n\n原始任务：\n{prompt}"
            )
            raw2 = self.chat(repair_prompt, system=json_system, temperature=0.0, label=(label or self.role) + ":repair")
            data2 = extract_json_block(raw2)
            try:
                if data2 is None:
                    raise ValueError("修复后仍未提取到 JSON")
                return schema.model_validate(data2)
            except (ValidationError, ValueError) as second_err:
                raise JSONValidationError(
                    f"结构化输出解析失败 (role={self.role}, schema={schema.__name__})",
                    raw_text=raw2,
                    errors=str(second_err),
                ) from second_err


def _schema_example(schema: Type[BaseModel]) -> dict:
    """给模型的 Schema 提示：字段名 → 类型占位描述。"""
    example: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        ann = field.annotation
        desc = field.description or ""
        origin = getattr(ann, "__origin__", None)
        if ann is str:
            example[name] = f"<string{'; ' + desc if desc else ''}>"
        elif ann is int:
            example[name] = 0
        elif ann is float:
            example[name] = 0.5
        elif ann is bool:
            example[name] = False
        elif origin is list:
            # 数组元素类型决定示例形状：对象数组必须展开成完整字段示例——
            # 旧版一律给 ["<string>"]，list[BaseModel] 被描述成字符串数组，
            # 模型按错误提示自造字段（实测 glm-5.3 回 {"id": "#0"}）导致校验必败，
            # 这是结构化输出一次通过率低的主因
            args = getattr(ann, "__args__", None)
            elem = args[0] if args else str
            if isinstance(elem, type) and issubclass(elem, BaseModel):
                example[name] = [_schema_example(elem)]
            elif elem is int:
                example[name] = [0]
            elif elem is float:
                example[name] = [0.5]
            else:
                example[name] = [f"<string{'; ' + desc if desc else ''}>"]
        elif origin is dict:
            example[name] = {"<key>": "<string>"}
        else:
            args = getattr(ann, "__args__", None)
            if args and isinstance(args[0], str):
                example[name] = args[0]  # Literal 首个值
            elif args:
                example[name] = _schema_example(args[0]) if isinstance(args[0], type) and issubclass(args[0], BaseModel) else None
            else:
                example[name] = "<value>"
    return example


def get_llm(role: str, *, budget: TokenBudget | None = None) -> LLMClient:
    return LLMClient(role, budget=budget)


def get_fake_llm(role: str = "fast", *, budget: TokenBudget | None = None) -> LLMClient:
    """强制 fake 后端的客户端：离线单元测试 / CI / --fake 演示模式。"""
    client = LLMClient(role, budget=budget)
    client.provider_name = "fake"
    client.provider_cfg = {"type": "fake"}
    client.ptype = "fake"
    client.model = "fake"
    return client
