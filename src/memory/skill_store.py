"""程序性记忆：Skill 的加载、校验、粗匹配与候选落盘。

遵循“生成候选—离线验证—人工启用”闭环：add_candidate 只写候选文件，
不直接修改线上策略；match() 只命中 status=enabled 的 Skill。
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.settings import PROJECT_ROOT
from src.tracing import get_active_tracer

_CJK_RUN = re.compile(r"[一-鿿]+")
_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-]+")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_\-]*$")


def _tokens(text: str) -> set[str]:
    """中英混合粗分词：ASCII 词（≥2 字符）+ CJK 二元组（长度 1 的 run 保留单字）。"""
    if not text:
        return set()
    tokens = set(_ASCII_WORD.findall(text.lower()))
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.add(run)
        else:
            tokens.update(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class Skill(BaseModel):
    name: str
    version: str = "0.1"
    status: str = "candidate"           # candidate | enabled | disabled
    trigger: str = ""
    actions: list[str] = Field(default_factory=list)
    evaluation: list[str] = Field(default_factory=list)
    created_from: list[str] = Field(default_factory=list)
    notes: str | None = None


class SkillStore:
    """skills/*.yaml 的加载与候选管理；坏文件跳过并 tracer 告警，不中断加载。"""

    def __init__(self, skills_dir: str | Path | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir is not None else PROJECT_ROOT / "skills"
        self.candidates_dir = self.skills_dir / "candidates"
        self._lock = threading.Lock()
        self._skills: list[Skill] = []

    def load(self) -> list[Skill]:
        """扫描 skills/*.yaml 与 skills/candidates/*.yaml，逐文件校验，坏文件跳过。"""
        skills: list[Skill] = []
        paths: list[Path] = []
        if self.skills_dir.is_dir():
            paths += sorted(self.skills_dir.glob("*.yaml")) + sorted(self.skills_dir.glob("*.yml"))
        if self.candidates_dir.is_dir():
            paths += sorted(self.candidates_dir.glob("*.yaml")) + sorted(self.candidates_dir.glob("*.yml"))
        for path in paths:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if not isinstance(data, dict):
                    raise ValueError("YAML 顶层必须是映射")
                skills.append(Skill.model_validate(data))
            except Exception as ex:
                get_active_tracer().event("skill_load_warning", node="skill_store", file=str(path), error=str(ex))
        with self._lock:
            self._skills = skills
        return list(skills)

    def skills(self) -> list[Skill]:
        return self.load()

    def enabled(self) -> list[Skill]:
        return [s for s in self.skills() if s.status == "enabled"]

    def candidates(self) -> list[Skill]:
        return [s for s in self.skills() if s.status == "candidate"]

    def match(self, query: str) -> list[Skill]:
        """对 enabled Skill 的 trigger 做关键词粗匹配：分词后与 query 词重叠 ≥1 即命中。"""
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        return [s for s in self.enabled() if _tokens(s.trigger) & query_tokens]

    def add_candidate(self, skill: Skill) -> Path:
        """把 Skill 写入 skills/candidates/{name}.yaml（status 强制为 candidate），返回文件路径。"""
        name = (skill.name or "").strip()
        if not _SAFE_NAME.match(name):
            raise ValueError(f"非法 Skill 名称: {skill.name!r}")
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        data = skill.model_dump(exclude={"status"})
        data["status"] = "candidate"
        path = self.candidates_dir / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return path
