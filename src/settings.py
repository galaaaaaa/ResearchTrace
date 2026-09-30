"""全局配置：加载 .env 与 configs/*.yaml，支持 ${VAR} 与 ${VAR:-default} 插值。"""

from __future__ import annotations

import functools
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Path | None = None, *, override: bool = False) -> None:
    """极简 .env 加载：不覆盖已有环境变量。"""
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def _interpolate(value: Any) -> Any:
    """递归解析配置中的 ${VAR} / ${VAR:-default}。"""

    def repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        env = os.environ.get(name)
        if env is not None and env != "":
            return env
        return default if default is not None else ""

    if isinstance(value, str):
        return _ENV_VAR.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _interpolate(data)


class Settings:
    """聚合 models / budgets / sources 三份配置与路径约定。"""

    def __init__(self) -> None:
        load_dotenv()
        self.models: dict = _load_yaml(PROJECT_ROOT / "configs" / "models.yaml")
        self.budgets: dict = _load_yaml(PROJECT_ROOT / "configs" / "budgets.yaml")
        self.sources: dict = _load_yaml(PROJECT_ROOT / "configs" / "sources.yaml")

        data_dir = Path(os.environ.get("RA_DATA_DIR") or PROJECT_ROOT / "data")
        self.data_dir = data_dir
        self.papers_dir = data_dir / "papers"
        self.indexes_dir = data_dir / "indexes"
        self.traces_dir = data_dir / "traces"
        self.eval_sets_dir = data_dir / "eval_sets"
        self.reports_dir = PROJECT_ROOT / "outputs" / "reports"
        self.audit_dir = PROJECT_ROOT / "outputs" / "audit"
        for d in (self.papers_dir, self.indexes_dir, self.traces_dir, self.eval_sets_dir, self.reports_dir, self.audit_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.db_path = self.indexes_dir / "evidence.sqlite3"

    # ---- 便捷访问 ----
    def budget(self, key: str, default: Any = None) -> Any:
        defaults = self.budgets.get("defaults", {})
        return defaults.get(key, default)

    def source(self, section: str, key: str, default: Any = None) -> Any:
        return self.sources.get(section, {}).get(key, default)

    def role_config(self, role: str) -> dict:
        roles = self.models.get("roles", {})
        base = dict(roles.get(role) or roles.get("_default") or {})
        base.setdefault("provider", self.models.get("default_provider", "fake"))
        base.setdefault("model", "")
        base.setdefault("temperature", 0.2)
        base.setdefault("max_tokens", 4096)
        return base

    def provider_config(self, name: str) -> dict:
        providers = self.models.get("providers", {})
        cfg = providers.get(name)
        if cfg is None:
            # 未配置任何真实 provider 时回退到 fake，保证系统可运行、可测试
            cfg = {"type": "fake"}
        return cfg


_lock = threading.Lock()
_cached: Settings | None = None


def get_settings() -> Settings:
    global _cached
    with _lock:
        if _cached is None:
            _cached = Settings()
        return _cached


def reset_settings() -> None:
    """测试用：强制重新加载配置。"""
    global _cached
    with _lock:
        _cached = None


def enable_fake_mode() -> None:
    """把所有角色切到 fake provider（离线演示 / CI 冒烟，零网络零 token）。"""
    s = get_settings()
    s.models.setdefault("providers", {})["fake"] = {"type": "fake"}
    for role, cfg in (s.models.get("roles") or {}).items():
        if isinstance(cfg, dict):
            cfg["provider"] = "fake"
    s.models["default_provider"] = "fake"


@functools.lru_cache(maxsize=None)
def _load_yaml_cached(path_str: str) -> dict:
    return _load_yaml(Path(path_str))
